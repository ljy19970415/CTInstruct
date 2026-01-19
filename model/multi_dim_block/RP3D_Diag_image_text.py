from .resnet2D import resnet50_2D
from torch import nn
import torch
from einops import rearrange
from .transformer_clip.attention import Transformer, ContinuousPositionBias

class RadNet_VisEncoder(nn.Module):
    def __init__(self, size, depth, hid_dim, dim_head=64, heads=8, spatial_depth=4, temporal_depth=4, augment=False):
        super().__init__()
        self.size = size
        self.depth = depth
        self.resnet2D = resnet50_2D()
        self.vision_dim = 2048

        transformer_kwargs = dict(
            dim = self.vision_dim,
            dim_head = dim_head, # 64
            heads = heads, # 8
            attn_dropout = 0., # 0.
            ff_dropout = 0., # 0.
            peg = True,
            peg_causal = True,
        )
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs) # depth 4
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs) # depth 4
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = self.vision_dim, heads = heads)

    def forward(self, image_x, marks, patch_num):

        # image_x p c h w d
        # marks p d
        # device = image_x.

        B = image_x.shape[0]

        image_x = image_x.repeat(1,3,1,1,1)

        image_x = rearrange(image_x, 'b c h w d -> (b d) c h w')

        skips, res_x = self.resnet2D(image_x) # (B d) (2048*h'*w')

        output_x = res_x
        
        output_x = rearrange(output_x, '(z w h d) v ->(z d) w h v', z=patch_num[0],w=patch_num[1],h=patch_num[2])
        output_x = output_x[None,:] # 1 t w h v

        video_shape = tuple(output_x.shape[:-1]) # b t h w

        output_x = rearrange(output_x, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(patch_num[1], patch_num[2], device = output_x.device)
        
        tokens = self.enc_spatial_transformer(output_x, attn_bias = attn_bias, video_shape = video_shape)
        
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=1, h = patch_num[1] , w = patch_num[2])

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        if len(marks.shape) == 1:
            marks = marks[None,:]
        attn_mask = rearrange(marks, '(t w h) d -> (w h) (t d)',w=patch_num[1], h = patch_num[2])

        tokens = self.enc_temporal_transformer(tokens, self_attn_mask = attn_mask.bool(), video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) (z t) d -> (b z h w) t d', z = patch_num[0], h = patch_num[1], w = patch_num[2])

        return skips, tokens

    def seg_forward(self, image_x, marks):

        # image_x p c h w d
        # marks p d
        # device = image_x.
        B = image_x.shape[0]
        image_x = image_x.repeat(1,3,1,1,1)

        image_x = rearrange(image_x, 'b c h w d -> (b d) c h w') # 8 32 1 256 256

        skips, res_x = self.resnet2D(image_x) # (B d) (2048*h'*w')

        output_x = res_x
        
        output_x = rearrange(output_x, '(z w h d) v ->(z d) w h v', z=patch_num[0],w=patch_num[1],h=patch_num[2])
        output_x = output_x[None,:] # 1 t w h v

        video_shape = tuple(output_x.shape[:-1]) # b t h w

        output_x = rearrange(output_x, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(patch_num[1], patch_num[2], device = output_x.device)
        tokens = self.enc_spatial_transformer(output_x, attn_bias = attn_bias, video_shape = video_shape)
        
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', t = self.depth, h = patch_num[1] , w = patch_num[2])

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        attn_mask = rearrange(marks, '(t w h) d -> (w h) (t d)',w=patch_num[1], h = patch_num[2])

        tokens = self.enc_temporal_transformer(tokens, self_attn_mask = attn_mask.bool(), video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) (z t) d -> (b z h w) t d', z = patch_num[0], h = patch_num[1], w = patch_num[2])

        return skips, tokens

def resample_image(image, tS, tD):
    # image d c h w
    depth = image.shape[0]
    height = image.shape[-2]
    width = image.shape[-1]
    assert height == width
    size = height

    if size == tS and depth <= tD:
        output_tensor = image
    else:
        output_tensor = torch.nn.functional.interpolate(image, size=(tS, tS), mode='bilinear', align_corners=False)
        if depth > tD:
            step = depth / tD
            indices = indices = torch.arange(0, depth, step).long()
            indices = indices[:tD]
            output_tensor = output_tensor[indices]
    return output_tensor

def getimage(size,depth):
    mark = torch.zeros(depth)

    image_fuse = torch.zeros((1,size,size,depth), dtype=torch.float32)

    image_datas = np.load(image_path) # x,y,z
    image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0)
    image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
    image_tensor = resample_image(image_tensor, size, depth)
    image_tensor = rearrange(image_tensor, "d c h w -> c h w d")

    mark[:image_tensor.shape[-1]] = 1

    if image_tensor.max()-image_tensor.min() == 0:
        print("warning: 0")
        image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
    image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
    image_fuse[:,:,:,0:image.shape[-1]] = image

    # image 
    return image_fuse, mark