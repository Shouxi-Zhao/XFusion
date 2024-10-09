
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
import math
from einops.einops import rearrange
from numpy.core.fromnumeric import partition
from numpy.core.records import array


def print(*args):
    pass


def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    """ 
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

class TransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
       

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class TransEmbeddings(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, layer_norm_eps, dropout_prob):
        super().__init__()
        print('embedding in out ', max_position_embeddings, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.LayerNorm = TransLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        input_shape = input_ids.size()
    
        seq_length = input_shape[1]
        device = input_ids.device
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[:2])
       
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        ## 最后xshape (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward( self, hidden_states ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 注意力加权
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class TransSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = TransLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TransAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.selfatten = TransSelfAttention(num_attention_heads, hidden_size, dropout_prob)
        self.output = TransSelfOutput(hidden_size, layer_norm_eps, dropout_prob)

    def forward(self, hidden_states):
        self_outputs = self.selfatten(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output

class TransIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act] ## relu 

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TransOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = TransLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransLayer(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.attention = TransAttention(num_attention_heads, hidden_size, layer_norm_eps, dropout_prob)
        self.intermediate = TransIntermediate(hidden_size, intermediate_size, hidden_act='gelu')
        self.output = TransOutput(intermediate_size, hidden_size, layer_norm_eps, dropout_prob)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        # return attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MultiTransLayer(nn.Module):
    def __init__(self, num_layers, num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.multi_layers =  nn.ModuleList([TransLayer(num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, dropout_prob) for _ in range(num_layers)])

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.multi_layers):
            hidden_states = layer_module(hidden_states)
        return hidden_states

class InputDense(nn.Module):
    def __init__(self, in_size, hidden_size, layer_norm_eps):
        super(InputDense, self).__init__()    
        self.dense = nn.Linear(in_size, hidden_size)
        self.transform_act_fn = ACT2FN['gelu']
        self.LayerNorm = TransLayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TransModel(nn.Module):
    """
    input shape (1, 64, 256, 256) with patch size(8, 8) get input(1, 256*256, 64)
    in_size: 64 (1, 256*256, 64)
    out_size: 64 (1, 256*256, 64)
    max_position_embeddings: 256*256
    hidden_size: 64 -> 128 
    intermediate_size: 128 -> 160 
    """
    def __init__(self, in_size, out_size, max_position_embeddings, hidden_size, intermediate_size, num_layers=4, num_attention_heads=6, layer_norm_eps= 1e-12, dropout_prob=0.1):
        super(TransModel, self).__init__()
        self.indense = InputDense(in_size, hidden_size, layer_norm_eps)
        self.embeddings = TransEmbeddings( max_position_embeddings, hidden_size, layer_norm_eps, dropout_prob)
        self.encoder = MultiTransLayer(num_layers, num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, dropout_prob)
        
        self.outdense = nn.Linear(hidden_size, out_size)
        # if not(hidden_size == out_size):
        #     self.outdense = nn.Linear(hidden_size, out_size)
        # else:
        #     self.outdense = None

    def forward(self, x):  
        print('TransModel2d input_ids ', x.shape)

        x = self.indense(x)
        print('dense_out', x.shape)

        x = self.embeddings( x )
        print('embedding_output', x.shape)
        
        x = self.encoder(x)
        print('encoder_layers', x.shape)

        if self.outdense:
            x = self.outdense(x)
            print('outdense output', x.shape)

        return x

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
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 
 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
 
    def forward(self, x):
        return self.conv(x)

 
class MultiFocusTransNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes=1, bilinear=True, **args):
        super(MultiFocusTransNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
 
        self.inc = DoubleConv(n_channels, 32)   
        self.down1 = Down(32, 64)              # 1 480 640
        self.down2 = Down(64, 128)             # 1 240 320  - > 120 160

        self.transencoder_1 = nn.Sequential(
            RearrangeTensor('b c (n1 h) (n2 w) -> b (n1 n2 c) (h w)', h=30, w=40),
            TransModel(
                in_size     = 1200, 
                out_size    = 1200, 
                max_position_embeddings= 128 * 16, 
                hidden_size = 1280, 
                intermediate_size= 1536,
                num_layers  = 1,
                num_attention_heads= 4),
            RearrangeTensor('b (n1 n2 c) (h w) -> b c (n1 h) (n2 w)', c=128, h=30, w=40, n1=4, n2=4),
        )

        self.down3 = Down(128, 256)             # 1 120 160 - > 60 80

        self.transencoder_2 = nn.Sequential(
            RearrangeTensor('b c (n1 h) (n2 w) -> b (n1 n2 c) (h w)', h=30, w=40),
            TransModel(
                in_size     = 1200, 
                out_size    = 1200, 
                max_position_embeddings= 256 * 4, 
                hidden_size = 1280, 
                intermediate_size= 1536,
                num_layers  = 1,
                num_attention_heads= 4),
            RearrangeTensor('b (n1 n2 c) (h w) -> b c (n1 h) (n2 w)', c=256, h=30, w=40, n1=2, n2=2),
        )

        self.down4 = Down(256, 256)             # 1 60 80 - > 30 40
        
        self.transencoder_3 = nn.Sequential(
            RearrangeTensor('b c h w -> b c (h w)', h=30, w=40),
            TransModel(
                in_size     = 1200, 
                out_size    = 1200, 
                max_position_embeddings= 1280, 
                hidden_size = 1280, 
                intermediate_size= 1536,
                num_layers  = 4,
                num_attention_heads= 4),
            RearrangeTensor('b c (h w) -> b c h w', c=256, h=30, w=40),
        )

        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        print(x.shape)
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)
        x5 = self.down4(x4)
        print(x5.shape)

        x5 = self.transencoder_3(x5)

        x = self.up1(x5, self.transencoder_2(x4))
        print(x.shape)
        x = self.up2(x, self.transencoder_1(x3))
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        logits = self.outc(x)
        return logits
 
class UTransformer(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UTransformer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    
 
    def forward(self, x):
        print(x.shape)
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)
        x5 = self.down4(x4)
        print(x5.shape)
        x = self.up1(x5, x4)
        print(x.shape)
        x = self.up2(x, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        logits = self.outc(x)
        return logits
 
 
def test():
    net = MultiFocusTransNet(3, 1)
    x = torch.randn(1, 3, 480, 640)
    y = net(x)
    print(y.size())
    print(sum(p.numel() for p in net.parameters()))
    # torch.Size([1, 10, 32, 32])

def test_loss():
    net = MultiFocusTransNet(3,10)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss2d()
    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.MSELoss()

    x = torch.randn(4, 3, 480, 640)
    y_label = torch.randn(4, 480, 640).long()
    y = net(x)
    print(y.size())
    loss = criterion(y, y_label)
    print(loss.item())

if __name__ == "__main__":
    test()
    # test_loss()
