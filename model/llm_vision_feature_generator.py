from .multi_dim_block.RP3D_Diag_image_text import RadNet_VisEncoder
from .multi_dim_block.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from .multi_dim_block.unetr_decoder_clip import UnetrDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

def l2norm(t):
    return F.normalize(t, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

class Text_Encoder():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-2023')
        self.model = AutoModel.from_pretrained('FremyCompany/BioLORD-2023')

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = False

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self,sentences):

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


class LLM_Vision_Encoder(nn.Module):
    def __init__(self, config):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()

        self.hid_dim = config.vision_dim 
        self.query_dim = config.query_dim

        self.encoder = RadNet_VisEncoder(size = config.crop_size[0], depth = config.crop_size[-1], hid_dim = self.hid_dim)
        
        self.bridger = nn.Sequential(
                nn.Linear(self.hid_dim, self.query_dim),
                nn.GELU(),
                nn.Linear(self.query_dim, self.query_dim),
                nn.GELU()
        )

        self.decoder = UnetrDecoder()

        # # segmentation layer
        self.avg_pool_ls = [
            nn.AvgPool2d(16),
            nn.AvgPool2d(8),
            nn.AvgPool2d(4),
            nn.AvgPool2d(2)
            ]
            
        # # multi-scale latent feature are projected to query_dim before query decoder
        self.projection_layer = nn.Sequential(
            nn.Linear(3904, config.biolord_dim),
            nn.GELU(),
            nn.Linear(config.biolord_dim, config.biolord_dim),
            nn.GELU()
            )

        # # positional encoding
        pos_embedding = PositionalEncoding2D(config.biolord_dim)(torch.zeros(1, 8, 8, config.biolord_dim)) # b h/p w/p d/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w c -> (h w) b c')   # n b dim
        
        # # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=config.biolord_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(config.biolord_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)

        self.qwen2biolord = nn.Linear(config.query_dim, config.biolord_dim)
        
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(config.biolord_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU()
        )

    def forward(self, images, marks, patch_num, device, freeze_vision=False):

        image_embeds = []
        for b in range(len(images)):
            image = torch.tensor(images[b][0][0][:,None,:,:,:])
            image = image.to(device) # z_p*w_p*h_p, 1, 256,256,32
            

            mark = torch.tensor(marks[b][0]) # z_p*w_p*h_p 32
            if len(mark.shape) > 2:
                mark = mark.squeeze()
            mark = mark.to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                cur_patch_num = patch_num[b][0]
                encoder_features, vit_patch = self.encoder(image,mark,cur_patch_num)
            
                proj_image_feature = self.bridger(vit_patch.view(-1, self.hid_dim))     
                proj_image_feature = rearrange(proj_image_feature, '(z x y d) h-> (z d) x y h',z=cur_patch_num[0],x=cur_patch_num[1],y=cur_patch_num[2])    
                proj_image_feature = rearrange(proj_image_feature, 'z x y h-> (z x y) h', x=cur_patch_num[1],y=cur_patch_num[2]) 
                image_embeds.append(proj_image_feature) # z_p*w_p*h_p 32 2048 
        
        return torch.concat(image_embeds,0)

    def decode_mask(self, images, marks, patch_num, text_embeddings, device, freeze_vision=False):

        pred_masks = []
        for b in range(len(images)):
            masks_for_sample = []            

            image = torch.tensor(images[b][0][0][:,None,:,:,:])
            image = image.to(device) # z_p*w_p*h_p, 1, 256,256,32
            mark = torch.tensor(marks[b][0]) # z_p*w_p*h_p 32
            if len(mark.shape) > 2:
                mark = mark.squeeze()
            mark = mark.to(device)
            cur_patch_num = patch_num[b][0]

            # cur_patch_num = [2,2,max_cnt//4]
            if freeze_vision:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        encoder_features, vit_patch = self.encoder(image,mark,cur_patch_num)
            else:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  
                    encoder_features, vit_patch = self.encoder(image,mark,cur_patch_num)
            
            # vit_patch z_p*w_p*h_p 32 2048
            
            if freeze_vision:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):  
                        decoder_lastlayer_feature = self.decoder(encoder_features, vit_patch, None)
            else:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  
                    decoder_lastlayer_feature = self.decoder(encoder_features, vit_patch, None)
            
            label_feature = self.qwen2biolord(text_embeddings)
            label_feature = label_feature[None,:,:]
            image_embedding, pos = self.vision_backbone_forward(encoder_features, freeze_vision)

            B = image_embedding.shape[1]
            queries = label_feature.repeat(B,1,1).to(image_embedding[0].device)
            logits = self.train_forward(queries, image_embedding, pos, decoder_lastlayer_feature, cur_patch_num, freeze_vision)
            masks_for_sample.append(logits)
            masks_for_sample = torch.stack(masks_for_sample)
            pred_masks.append(masks_for_sample)
        return pred_masks

    def get_label_feature(self, label_dict):
        label_feature_dict = {}
        for task_name in label_dict:
            labels = label_dict[task_name]
            label_feature_dict[task_name] = self.text_encoder.encode(labels)
        return label_feature_dict
    
    def vision_backbone_forward(self, latent_embedding_ls, freeze_vision):

        image_embedding = []

        # latent_embedding_ls
        cnt = 0
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):

            if freeze_vision:
                with torch.no_grad():
                    tmp = avg_pool(latent_embedding)
            else:
                tmp = avg_pool(latent_embedding)
            
            image_embedding.append(tmp)   # B ? H/P W/P D/P
            cnt+=1
        image_embedding.append(latent_embedding_ls[-1])

        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)

        image_embedding = rearrange(image_embedding, 'b d h w -> b h w d')

        if freeze_vision:
            with torch.no_grad():
                image_embedding = self.projection_layer(image_embedding)   # B H/P W/P D/P Dim
        else:
            image_embedding = self.projection_layer(image_embedding)

        image_embedding = rearrange(image_embedding, 'b h w dim -> (h w) b dim') # (H/P W/P D/P) B Dim

        # add pe to image embedding
        if freeze_vision:
            with torch.no_grad():
                pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P D/P) 1 Dim
        else:
            pos = self.pos_embedding.to(latent_embedding_ls[-1].device)
            
        return image_embedding, pos

    def train_forward(self, queries, image_embedding, pos, decoder_lastlayer_feature, patch_num, freeze_vision):

        _, B, _ = image_embedding.shape # (h w d) B query_dim
        
        # query decoder
        _, N, _ = queries.shape    # N is the num of query B N query_dim

        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim NOTE:By default, attention in torch is not batch_first

        if freeze_vision:
            with torch.no_grad():
                mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
                mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
                # Dot product
                last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
                last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        else:
            mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            # Dot product
            last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
            last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        
        logits = [torch.einsum('bchw,bnc->bnhw', decoder_lastlayer_feature, last_mask_embedding)]

        predict_logits = rearrange(logits[0], '(b z x y d) c h w -> b c (x h) (y w) (z d)', b=1, z = patch_num[0],x=patch_num[1],y=patch_num[2])

        return predict_logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == '__main__':
    text_encoder = Text_Encoder()
    sentences = ["Cat scratch injury", "Cat scratch disease", "Bartonellosis"]
    embeddings = text_encoder.encode(sentences)
    print("Sentence embeddings:",embeddings.shape)

    model = LLM_Vision_Encoder([256,256,32],2048,256,1536)
    module1_params = sum(p.numel() for p in model.encoder.parameters())
    module2_params = sum(p.numel() for p in model.bridger.parameters())

    print(f"Encoder Parameter: {module1_params}")
    print(f"Bridger Parameter: {module2_params}")