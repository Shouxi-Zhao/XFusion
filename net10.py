## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import math
from pickle import NONE
from turtle import forward
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torchvision.models as models

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Fustormer -----------------------
class Fustormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = True        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Fustormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # print("input", inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        # print(inp_enc_level1.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # print(inp_enc_level1.shape)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        # print(inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # print(inp_enc_level2.shape)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        # print(inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        # print(inp_enc_level3.shape)

        inp_enc_level4 = self.down3_4(out_enc_level3) 
        # print(inp_enc_level4.shape)       
        # latent = self.latent(inp_enc_level4) 
        # print(latent.shape)
                        
        inp_dec_level3 = self.up4_3(self.latent(inp_enc_level4) )
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


class MultiScaleEncoder(nn.Module):
    def __init__(self, embed_dim, 
            num_blocks = [4,6,6,8], 
            heads = [1,2,4,8],
            ffn_expansion_factor = 2.66,
            bias = False,
            LayerNorm_type = 'WithBias'):
        super(MultiScaleEncoder, self).__init__()

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(embed_dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(embed_dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(embed_dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(embed_dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(embed_dim*2**2)) ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(embed_dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        

    def forward(self, x):
        # print("input", x.shape)

        out_enc_level1 = self.encoder_level1(x)
        # print(out_enc_level1.shape)
        
        # inp_enc_level2 = self.down1_2(out_enc_level1)
        # print(inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        # print(out_enc_level2.shape)

        # inp_enc_level3 = self.down2_3(out_enc_level2)
        # print(inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2)) 
        # print(out_enc_level3.shape)

        # inp_enc_level4 = self.down3_4(out_enc_level3) 
        # print(inp_enc_level4.shape) 
        out_enc_level4 = self.encoder_level4(self.down3_4(out_enc_level3))
        # print(out_enc_level4.shape)

        return (out_enc_level1, out_enc_level2, out_enc_level3, out_enc_level4) 

class MultiScaleDecoder(nn.Module):
    def __init__(self, dim, 
            num_blocks = [4,6,6,8], 
            num_refinement_blocks = 4,
            heads = [1,2,4,8],
            ffn_expansion_factor = 2.66,
            bias = False,
            LayerNorm_type = 'WithBias'):
        super(MultiScaleDecoder, self).__init__()

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        # self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*2**0), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
    
        self.refinement =  nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        
    def forward(self, encode_level1, encode_level2, encode_level3, encode_level4): # [1, 192, 30, 40]
        x = self.up4_3(encode_level4)
        x = torch.cat([x, encode_level3], dim=1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x) # [1, 96, 60, 80]
        # print(out_dec_level3.shape)

        x = self.up3_2(x)
        x = torch.cat([x, encode_level2],dim=1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x) # [1, 48, 120, 160]
        # print(out_dec_level2.shape)

        x = self.up2_1(x)
        x = torch.cat([x, encode_level1],dim=1)
        # inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        x = self.decoder_level1(x) # [1, 24 *2, 240, 320]
        # print(out_dec_level1.shape)

        x = self.refinement(x)
        
        return x


    # def forward_maxfusion(self, encode_level1, encode_level2, encode_level3, encode_level4): # [1, 192, 30, 40]
    #     inp_dec_level3 = self.up4_3(self.tensor_maxfusion(in_level4_list))
    #     inp_dec_level3 = torch.cat([inp_dec_level3, self.tensor_maxfusion(in_level3_list)], dim=1)
    #     inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
    #     out_dec_level3 = self.decoder_level3(inp_dec_level3) # [1, 96, 60, 80]
    #     # print(out_dec_level3.shape)

    #     inp_dec_level2 = self.up3_2(out_dec_level3)
    #     inp_dec_level2 = torch.cat([inp_dec_level2, self.tensor_maxfusion(in_level2_list)],dim=1)
    #     inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
    #     out_dec_level2 = self.decoder_level2(inp_dec_level2) # [1, 48, 120, 160]
    #     # print(out_dec_level2.shape)

    #     inp_dec_level1 = self.up2_1(out_dec_level2)
    #     inp_dec_level1 = torch.cat([inp_dec_level1, self.tensor_maxfusion(in_level1_list)],dim=1)
    #     inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
    #     out_dec_level1 = self.decoder_level1(inp_dec_level1) # [1, 24, 240, 320]
    #     # print(out_dec_level1.shape)
        
    #     return self.output(out_dec_level1)

    # def forward(self, encode_level1, encode_level2, encode_level3, encode_level4):
    #     return self.forward_convfusion(encode_level1, encode_level2, encode_level3, encode_level4)
        # return self.forward_maxfusion(in_level1_list, in_level2_list, in_level3_list, in_level4_list)

class AttentionFusion(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4,num_blocks=4):
        super(AttentionFusion, self).__init__()
        self.fusion_attention = nn.Sequential(*[TransformerBlock(dim=dim_in, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias") for i in range(num_blocks)])
        self.fusion_out = nn.Conv2d(dim_in, dim_out, kernel_size=1) # 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        return self.fusion_out(self.fusion_attention(x))

class MultiFocusTransNet(nn.Module):
    def __init__(self, in_channels=3, embed_dim=24, out_channels=3, num_fusion=3):
        super(MultiFocusTransNet, self).__init__()
        self.feasure_fusion_type = "max" # conv attention max
        self.patch_embed = OverlapPatchEmbed(in_channels, embed_dim)

        self.encoder = MultiScaleEncoder(embed_dim = embed_dim)
        self.decoder = MultiScaleDecoder(dim=embed_dim)

        self.skip_conv = nn.Conv2d(num_fusion * embed_dim, int(embed_dim*2), kernel_size=1, bias=False)
        self.output = nn.Conv2d(int(embed_dim*2), out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if self.feasure_fusion_type == "conv":
            self.feasure_fusion_level4 = nn.Conv2d(int(num_fusion * embed_dim*2**3), int(embed_dim*2**3), kernel_size=3, padding=1, bias=False)
            self.feasure_fusion_level3 = nn.Conv2d(int(num_fusion * embed_dim*2**2), int(embed_dim*2**2), kernel_size=3, padding=1, bias=False)
            self.feasure_fusion_level2 = nn.Conv2d(int(num_fusion * embed_dim*2**1), int(embed_dim*2**1), kernel_size=3, padding=1, bias=False)
            self.feasure_fusion_level1 = nn.Conv2d(int(num_fusion * embed_dim*2**0), int(embed_dim*2**0), kernel_size=3, padding=1, bias=False)
        elif self.feasure_fusion_type == "max" :
            self.feasure_fusion_level4 = None
            self.feasure_fusion_level3 = None
            self.feasure_fusion_level2 = None
            self.feasure_fusion_level1 = None
        elif self.feasure_fusion_type == "attention" :
            self.feasure_fusion_level4 = AttentionFusion(int(num_fusion * embed_dim*2**3), int(embed_dim*2**3), num_heads=8, num_blocks=2)
            self.feasure_fusion_level3 = AttentionFusion(int(num_fusion * embed_dim*2**2), int(embed_dim*2**2), num_heads=4, num_blocks=2)
            self.feasure_fusion_level2 = AttentionFusion(int(num_fusion * embed_dim*2**1), int(embed_dim*2**1), num_heads=2, num_blocks=2)
            self.feasure_fusion_level1 = AttentionFusion(int(num_fusion * embed_dim*2**0), int(embed_dim*2**0), num_heads=1, num_blocks=2)
        else:
            self.feasure_fusion_level4 = None
            self.feasure_fusion_level3 = None
            self.feasure_fusion_level2 = None
            self.feasure_fusion_level1 = None

    def tensor_maxfusion(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_convfusion(self, tensors, level):
        if level == 4:
            return self.feasure_fusion_level4(torch.cat(tensors, dim =1))
        elif level == 3:
            return self.feasure_fusion_level3(torch.cat(tensors, dim =1))
        elif level == 2:
            return self.feasure_fusion_level2(torch.cat(tensors, dim =1))
        elif level == 1:
            return self.feasure_fusion_level1(torch.cat(tensors, dim =1))
        else:
            assert False, level

    def tensor_attentionfusion(self, tensors, level):
        if level == 4:
            return self.feasure_fusion_level4(torch.cat(tensors, dim =1))
        elif level == 3:
            return self.feasure_fusion_level3(torch.cat(tensors, dim =1))
        elif level == 2:
            return self.feasure_fusion_level2(torch.cat(tensors, dim =1))
        elif level == 1:
            return self.feasure_fusion_level1(torch.cat(tensors, dim =1))
        else:
            assert False, level

    def level_feasure_fusion(self, encode_level_list, level):
        if self.feasure_fusion_type == "max":
            return self.tensor_maxfusion(encode_level_list)
        elif self.feasure_fusion_type == "conv":
            return self.tensor_convfusion(encode_level_list, level)
        elif self.feasure_fusion_type == "attention":
            return self.tensor_attentionfusion(encode_level_list, level)

    def forward(self, *tensors):
        # Feature extraction
        length = len(tensors)
        tensor_embeds = []
        for tensor in tensors:
            tensor_embeds.append(self.patch_embed(tensor))

        tensor_encodes = []
        for tensor in tensor_embeds:
            tensor_encodes.append(self.encoder(tensor))

        encode_level1 = []
        encode_level2 = []
        encode_level3 = []
        encode_level4 = []

        for i in range(length):
            encode_level1.append(tensor_encodes[i][0])
            encode_level2.append(tensor_encodes[i][1])
            encode_level3.append(tensor_encodes[i][2])
            encode_level4.append(tensor_encodes[i][3])

        del tensor_encodes

        out = self.decoder(
            self.level_feasure_fusion(encode_level1, level=1), 
            self.level_feasure_fusion(encode_level2, level=2), 
            self.level_feasure_fusion(encode_level3, level=3), 
            self.level_feasure_fusion(encode_level4, level=4))

        out = out + self.skip_conv(torch.cat(tensor_embeds, dim=1))
        out = self.output(out)
        return out

def test():
    net = MultiFocusTransNet(in_channels=3, embed_dim=24,out_channels=3)
    net.eval()
    x1 = torch.randn(1, 3, 240, 320)
    x2 = torch.randn(1, 3, 240, 320)
    x3 = torch.randn(1, 3, 240, 320)
    y = net(x1, x2, x3)
    print("out", y.shape)
    # y = net(x1, x2)
    # print('out', y.shape)
    print(sum(p.numel() for p in net.parameters()))
    # torch.Size([1, 10, 32, 32])

def test_MDTA():
    net = Attention(dim = 16, num_heads=4, bias=False)
    # x = torch.randn(1, 32, 120, 160)
    x = torch.randn(1, 16, 480, 640)
    # x = torch.randn(1, 32, 480, 560)
    y = net(x)
    print(y.shape)
    print(sum(p.numel() for p in net.parameters()))

def test_GDFN():
    net = FeedForward(dim = 16, ffn_expansion_factor = 2.66, bias=False)
    # x = torch.randn(1, 32, 120, 160)
    x = torch.randn(1, 16, 480, 560)
    # x = torch.randn(1, 32, 480, 560)
    y = net(x)
    print(y.shape)
    print(sum(p.numel() for p in net.parameters()))

def test_size():
    net = nn.Sequential(*[TransformerBlock(dim=128, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias") for i in range(4)])
    net.eval()
    x = torch.randn(1, 128, 120, 160)
    y = net(x)
    print("out", y.shape)
    print(sum(p.numel() for p in net.parameters()))

def test_MultiScaleEncoder():
    net = MultiScaleEncoder(in_channels=3, embed_dim=24)
    net.eval()
    x = torch.randn(1, 3, 240, 320)
    y = net(x)
    # print("out", y.shape)
    print(sum(p.numel() for p in net.parameters()))

def test_MultiScaleDecoder():
    net = MultiScaleDecoder(dim=24, out_channels=3)
    net.eval()

    net_encoder = MultiScaleEncoder(in_channels=3, embed_dim=24)
    x_in = torch.randn(1,3,240,320)
    y_in = torch.randn(1,3,240,320)
    z_in = torch.randn(1,3,240,320)

    x_out = net_encoder(x_in)
    y_out = net_encoder(y_in)
    z_out = net_encoder(z_in)

    y = net([x_out[0],y_out[0],z_out[0]], [x_out[1],y_out[1],z_out[1]], [x_out[2],y_out[2],z_out[2]], [x_out[3],y_out[3],z_out[3]])

    # x2 = torch.randn(1, 192, 30, 40)
    # x2 = torch.randn(1, 96, 60, 80)
    # x3 = torch.randn(1, 48, 120, 160)
    # x4 = torch.randn(1, 24, 240, 320)

    # y1 = torch.randn(1, 192, 30, 40)
    # y2 = torch.randn(1, 96, 60, 80)
    # y3 = torch.randn(1, 48, 120, 160)
    # y4 = torch.randn(1, 24, 240, 320)

    # z1 = torch.randn(1, 192, 30, 40)
    # z2 = torch.randn(1, 96, 60, 80)
    # z3 = torch.randn(1, 48, 120, 160)
    # z4 = torch.randn(1, 24, 240, 320)
    # y = net([z1,y1,x1], [z2,y2,x2], [z3,y3,x3], [z4,y4,x4])
    print("out", y.shape)
    print(sum(p.numel() for p in net.parameters()))


if __name__ == "__main__":
    test()
    # test_MDTA()
    # test_GDFN()
    # test_size()
    # test_MultiScaleEncoder()
    # test_MultiScaleDecoder()