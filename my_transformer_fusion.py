import math
from einops.einops import rearrange
from numpy.core.fromnumeric import partition
from numpy.core.records import array

import torch
from torch import nn

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

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.decoder = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
         # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
 
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.decoder(x)

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        print('decoder 2d.....')
        print(x.shape)
        x = self.decoder_1(x)
        print(x.shape)
        x = self.decoder_2(x)
        print(x.shape)
        x = self.decoder_3(x)
        print(x.shape)
        x = self.decoder_4(x)
        print(x.shape)
        x = self.final_out(x)
        print(x.shape)
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


class MultiFocusTransNet(nn.Module): 
    def __init__(self, input_shape, patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10], features=[512, 256, 128, 64]):
        super().__init__()
        self.input_shape = input_shape # (b, c, h, w)
        b, c, h, w = self.input_shape[:]
        px1, px2, px3 = patch_sizes_x[:]
        py1, py2, py3 = patch_sizes_y[:]

        # (b, c, h, w) => (b, t1, p1*p1) => (b, c, h, w)

        self.encoder1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
        )

        self.rearrange12 = RearrangeTensor('b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1=px1,p2=py1)

        emb = int(h*w*32/4/px1/py1)
        # print(emb, h, w, px1, py1)
        self.encoder2 = nn.Sequential(
                    
                    TransModel(
                        in_size     = px1*py1, 
                        out_size    = px1*py1, 
                        max_position_embeddings= emb,  # int(h*w*64/4/px1/py2), 
                        hidden_size = px1*py1, 
                        intermediate_size= px1*py1*2,
                        num_layers  = 1,
                        num_attention_heads= 8),
                    RearrangeTensor('b (h w c) (p1 p2) -> b c (h p1) (w p2)', c=32,p1=px1,p2=py1,h=int(h/2/px1)),
        )

        self.encoder3 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
        )

        self.rearrange34 = RearrangeTensor('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=px2,p2=py2)

        emb = int(h*w/4/4/px2/py2) 
        # print(emb, h, w, px2, py2)
        self.encoder4 = nn.Sequential(
                    
                    TransModel(
                        in_size     = px2*py2*64, 
                        out_size    = px2*py2*64, 
                        max_position_embeddings= emb, #int(h*w/4/4/px2/py2), 
                        hidden_size = px2*py2*64, 
                        intermediate_size= px2*py2*64*2,
                        num_layers  = 1,
                        num_attention_heads= 8),
                    RearrangeTensor('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=64,p1=px2,p2=py2,h=int(h/4/px2)),
        )

        self.encoder5 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
        )

        # self.rearrange56 = RearrangeTensor('b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1=px3,p2=py3)

        # emb = int(h*w*128/4/4/4/px3/py3), 
        # self.encoder6 = nn.Sequential(
                    
        #             TransModel(
        #                 in_size     = px3*py3, 
        #                 out_size    = px3*py3, 
        #                 max_position_embeddings= emb, #int(h*w*256/4/4/4/px3/py3), 
        #                 hidden_size = px3*py3, 
        #                 intermediate_size= px3*py3*2,
        #                 num_layers  = 1,
        #                 num_attention_heads= 8),
        #             RearrangeTensor('b (h w c) (p1 p2) -> b c (h p1) (w p2)', c=128,p1=px3,p2=py3,h=int(h/8/px3)),
        # )

        self.encoder7 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
        )

        self.rearrange78 = RearrangeTensor('b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1=px3,p2=py3)

        emb = int(h*w*256/4/4/4/4/px3/py3)
        self.encoder8 = nn.Sequential(
                    
                    TransModel(
                        in_size     = px3*py3, 
                        out_size    = px3*py3, 
                        max_position_embeddings= emb, #int(h*w*256/4/4/4/px3/py3), 
                        hidden_size = px3*py3, 
                        intermediate_size= px3*py3*2,
                        num_layers  = 1,
                        num_attention_heads= 4),
                    RearrangeTensor('b (h w c) (p1 p2) -> b c (h p1) (w p2)', c=256,p1=px3,p2=py3,h=int(h/16/px3)),
        )

        self.encoder9 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
        )

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 16)
        self.up5_1 = Up(17, 1)
        self.up5_2 = Up(17, 1)
        self.up5_3 = Up(17, 1)

        self.con_out = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=1, padding=0),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=False),
        )

        # self.up5 = nn.Sequential(
        #             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        #             nn.Conv2d(32, 1, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(1),
        #             nn.ReLU(inplace=True),
        # )

    def forward(self, xa):
        print(xa.shape)
        x1 = self.encoder1(xa)
        print('x1', x1.shape)
        x = self.rearrange12(x1)
        print(x.shape)
        x2 = self.encoder2(x)
        print('x2', x2.shape)
        x3 = self.encoder3(x2)
        print('x3', x3.shape)
        x = self.rearrange34(x3)
        print(x.shape)
        x4 = self.encoder4(x)
        print('x4', x4.shape)
        x5 = self.encoder5(x4)
        print('x5', x5.shape)
        # x = self.rearrange56(x5)
        # print(x.shape)
        # x6 = self.encoder6(x)
        # print('x6', x6.shape)
        x7 = self.encoder7(x5)
        print('x7', x7.shape)
        x = self.rearrange78(x7)
        print(x.shape)
        x8 = self.encoder8(x)
        print('x8', x8.shape)
        x9 = self.encoder9(x7)
        print('x9', x9.shape)

        x = self.up1(x9, x7)
        print('x9, x7 -> x',x.shape)
        x = self.up2(x, x5)
        print('x x5 -> x', x.shape)
        x = self.up3(x, x3)
        print('x x3 -> x', x.shape)
        x = self.up4(x, x1)
        print('x x1 -> x', x.shape)
        out1 = self.up5_1(x, xa[:,0,:,:][:,None])
        out2 = self.up5_2(x, xa[:,1,:,:][:,None])
        out3 = self.up5_3(x, xa[:,2,:,:][:,None])
        out = out1+out2+out3
        out = self.con_out(out)
        print('out', out.shape)
        return out


def test_TransLayerNorm():
    t = TransLayerNorm(1024)
    # input = torch.randn(1, 10, 1024)
    input = torch.randn(1, 3, 256, 1024)
    out = t(input)

    print(out.shape) # torch.Size([1, 10, 1024]) torch.Size([1, 3, 256, 1024])

def test_TransEmbeddings():
    t = TransEmbeddings(max_position_embeddings = 8, hidden_size = 1024, layer_norm_eps = 1e-12, dropout_prob = 0.1)
    # input = torch.randn(1, 10, 1024)
    input = torch.randn(1, 3, 1024)
    out = t(input)

    print(out.shape) # torch.Size([1, 10, 1024])  torch.Size([1, 3, 1024])


def test_TransSelfAttention():
    t = TransSelfAttention(num_attention_heads= 8, hidden_size=1024, dropout_prob=0.1)
    # input = torch.randn(1, 10, 1024)
    input = torch.randn(1, 48, 1024)
    out = t(input)

    print(out.shape) # torch.Size([1, 10, 1024])  torch.Size([1, 48, 1024])

def test_TransAttention():
    t = TransAttention(num_attention_heads= 8, hidden_size=1024, layer_norm_eps= 1e-12, dropout_prob=0.1)
    # input = torch.randn(1, 10, 1024)
    input = torch.randn(1, 48, 1024)
    out = t(input)

    print(out.shape) # torch.Size([1, 10, 1024])  torch.Size([1, 48, 1024])

def test_TransLayer():
    t = TransLayer(num_attention_heads= 8, hidden_size=1024, intermediate_size=1680, layer_norm_eps= 1e-12, dropout_prob=0.1)
    # input = torch.randn(1, 10, 1024)
    input = torch.randn(1, 48, 1024)
    out = t(input)

    print(out.shape) # torch.Size([1, 10, 1024])  torch.Size([1, 48, 1024])

    # multi_layers =  nn.ModuleList([TransLayer(num_attention_heads= 8, hidden_size=1024, intermediate_size=1680, layer_norm_eps= 1e-12, dropout_prob=0.1) for _ in range(7)])

    # all_encoded_layers = []
    # for i, layer_module in enumerate(multi_layers):
        
    #     layer_output = layer_module(input)
    #     input = layer_output
    #     all_encoded_layers.append(input)
    #     print(input.shape)

def test_InputDense():
    t = InputDense(in_size = 256, hidden_size=1024, layer_norm_eps= 1e-12 )
    input = torch.randn(1, 4, 32, 256) # last shape size should be same
    out = t(input)

    print(out.shape) # torch.Size([1, 4, 32, 1024]) last shape size should be same

def test_TransModel():
    from einops import rearrange

    # t0 = TransModel(512, 512, 768, 1024, 1568, 4, 8)
    # input = torch.randn(1,3, 512)
    # print(t0(input).shape)

    # t0 = TransModel(1024, 1024, 768, 1024, 1568, 4, 8)
    # input = torch.randn(1, 8, 1024)
    # print(t0(input).shape)

    # p = 32
    # t = TransModel(
    #     in_size = p*p,
    #     out_size= p*p,
    #     max_position_embeddings = int(512*512*3 / p / p), 
    #     hidden_size = p*p*2, 
    #     intermediate_size =p*p*3,
    #     num_layers=2,
    #     num_attention_heads=8)

    # input = torch.randn(1, 3, 512, 512)
    # # in_patchs = rearrange(input, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 32, p2 = 32)
    # in_patchs = rearrange(input, 'b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1 = p, p2 = p)
    # print(in_patchs.shape) # torch.Size([1, 768, 1024])
    # out = t(in_patchs)

    # print(out.shape) # torch.Size([1, 768, 1024])

    # out = rearrange(out, 'b (h w c) (p1 p2) -> b c (h p1) (w p2)', c = 3, p1 = p, p2 = p, h=512//p)
    # print (out.shape) # torch.Size([1, 3, 512, 512])

    cn = 128
    px = 28
    py = 21
    t = TransModel(
        in_size = px*py,
        out_size= px*py,
        max_position_embeddings = int(128*140*210/px/py), 
        hidden_size = px*py, 
        intermediate_size =px*py*2,
        num_layers=2,
        num_attention_heads=6)

    input = torch.randn(1, cn, 140, 210)
    # in_patchs = rearrange(input, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 32, p2 = 32)
    in_patchs = rearrange(input, 'b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1 = px, p2 = py)
    print(in_patchs.shape) # torch.Size([1, 768, 1024])
    out = t(in_patchs)

    print(out.shape) # torch.Size([1, 768, 1024])

    out = rearrange(out, 'b (h w c) (p1 p2) -> b c (h p1) (w p2)', p1 = px, p2 = py, h=280//px, w = 420//py)
    print (out.shape) # torch.Size([1, 3, 512, 512])
    
def test_Decoder2D():
    t = Decoder2D(in_channels= 1024,out_channels=1, features=[512, 256, 128, 64])
    input = torch.randn(1, 1024, 24, 24)
    out = t(input)

    print(out.shape) # torch.Size([1, 1, 384, 384])

def test_multi_input(parten:str, **axes_lengths):
    t = RearrangeTensor(parten, **axes_lengths)
    input = torch.randn(1, 1024, 24, 24)
    out = t(input)
    print(out.shape)

    # print(parten)
    # print(axes_lengths)

    # input = torch.randn(1, 1024, 24, 24)
    # out = rearrange(input, parten, **axes_lengths)
    # print(out.shape)

def test_MultiFocusTransNet():
    # input = torch.randn(1, 3, 512, 512)
    input = torch.randn(1, 3, 480, 640)

    t = MultiFocusTransNet(input.shape)
    
    out = t(input)
    print(out.shape)


if __name__ == '__main__':
    # test_TransLayerNorm()
    # test_TransEmbeddings()
    # test_TransSelfAttention()
    # test_TransAttention()
    # test_TransLayer()
    # test_InputDense()
    # test_TransModel()
    # test_Decoder2D()

    test_MultiFocusTransNet()

    # test_multi_input('b c (h p1) (w p2)-> b (h w) (p1 p2 c)', p1=3, p2=3)