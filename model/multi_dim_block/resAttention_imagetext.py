import torch
from torch import nn
import torch.nn.functional as F
from .transformer_encoder import TransformerEncoder


class Attention1D(nn.Module):
    def __init__(self, hid_dim=2048, max_depth=32, nhead=8, num_layers=6, pool='cls', batch_first=True):
        super().__init__()
        
        self.transformer_encoder = TransformerEncoder(d_model=hid_dim, d_ff=hid_dim, n_heads=nhead, n_layers=num_layers, dropout=0.)

        self.pool = pool
        self.batch_first = batch_first
        if pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim, dtype=torch.float32))
        
        
    def forward(self, x, mask):
        B, S, D = x.shape # B depth (2048*h'*w')
        if self.pool == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  
            x = torch.cat((cls_tokens, x), dim=1)
            mask = F.pad(mask, (1, 0), "constant", 1)
        elif self.pool == 'mean':
            pass
        else:
            raise ValueError('pool type must be either cls (cls token) or mean (mean pooling)')
        

        if not self.batch_first:
            x = x.permute(1, 0, 2)
            mask = mask.permute(1, 0)


        x, attn_x, score_x = self.transformer_encoder(x, mask.to(torch.bool))

        attn_x = attn_x.mean(dim=1) # B d+1 d+1
        attn_x = attn_x[:,0,:] # B 1 d+1
        score_x = score_x.mean(dim=1) # B d+1 d+1
        score_x = score_x[:,0,:] # B d+1

        if not self.batch_first:
            # not run here
            x = x.permute(1, 0, 2)

        if self.pool == 'cls':
            # run here
            out_x = x[:,0,:] # B 2048
        else:
            out_x = x.mean(dim=1)

        return out_x, attn_x, score_x, x