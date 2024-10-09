## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

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
        print("input", inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        print(inp_enc_level1.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        print(inp_enc_level1.shape)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        print(inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        print(inp_enc_level2.shape)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        print(inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        print(inp_enc_level3.shape)

        inp_enc_level4 = self.down3_4(out_enc_level3) 
        print(inp_enc_level4.shape)       
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



class RearrangeTensor(nn.Module):
    def __init__(self, str_rearrange, **axes_lengths):
        super().__init__()
        self.str_rearrange = str_rearrange
        self.args = axes_lengths

    def forward(self, x):
        # print(self.args)
        # print(**self.args)
        return rearrange(x,self.str_rearrange, **self.args)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
 
    def forward(self, x):
        return self.double_conv(x)
 
 
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
 
 
    def forward(self, x):
        return self.maxpool_conv(x)
 
 
class Up(nn.Module):
    """Upscaling then double conv"""
 
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
 
 
        self.conv = DoubleConv(in_channels, out_channels)
 
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
 
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
 
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 
 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
 
    def forward(self, x):
        return self.conv(x)

 
class MultiFocusTransNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels=1, bilinear=True, **args):
        super(MultiFocusTransNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
 
 
        self.inc = DoubleConv(in_channels, 32)   
        self.down1 = Down(32, 64)              # 1 480 640
        self.transencoder_0 = nn.Sequential(*[TransformerBlock(dim=64, 
                            num_heads=4, 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type="WithBias") for i in range(2)])


        self.down2 = Down(64, 128)             # 1 240 320  - > 120 160
        self.transencoder_1 = nn.Sequential(*[TransformerBlock(dim=128, 
                            num_heads=4, 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type="WithBias") for i in range(4)])

        self.down3 = Down(128, 256)             # 1 120 160 - > 60 80
        self.transencoder_2 = nn.Sequential(*[TransformerBlock(dim=256, 
                            num_heads=4, 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type="WithBias") for i in range(4)])

        self.down4 = Down(256, 256)             # 1 60 80 - > 30 40
        self.transencoder_3 = nn.Sequential(*[TransformerBlock(dim=256, 
                            num_heads=4, 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type="WithBias") for i in range(6)])


        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, out_channels)

        # self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # self.last_conv = DoubleConv(out_channels, out_channels)

    def forward(self, input):
        # print(input.shape)
        x1 = self.inc(input)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print("x2", x2.shape)
        x3 = self.down2(x2)
        # print("x3", x3.shape)
        x4 = self.down3(x3)
        # print("x4", x4.shape)
        x5 = self.down4(x4)
        # print("x5", x5.shape)

        x5 = x5 + self.transencoder_3(x5)

        x = self.up1(x5, self.transencoder_2(x4))
        # print("up1", x.shape)
        x = self.up2(x, self.transencoder_1(x3))
        # print("up2", x.shape)
        x = self.up3(x, self.transencoder_0(x2))
        # print("up3", x.shape)
        x = self.up4(x, x1)
        # print("up4", x.shape)
        last = self.outc(x)
        # print("last ", last.shape)

        # last = last + self.skip_conv(input)
        # # print("skip", last.shape)
        # last = self.last_conv(last)
        return last
 

 
def test():
    net = MultiFocusTransNet(9, 3)
    net.eval()
    x = torch.randn(1, 9, 480, 640)
    y = net(x)
    print("out", y.shape)
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

if __name__ == "__main__":
    test()
    # test_MDTA()
    # test_GDFN()
