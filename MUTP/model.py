"""
An Lao
"""
import math
import logging
import random
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from urbanclip_model import Residual, ParallelTransformerBlock, CrossAttention, LayerNorm

_logger = logging.getLogger(__name__)

pi = 3.1415926535

def print_check(text, image, bilinear):
    print('Check value:')
    if text is not None:
        print('text:\n', text[0][0])
    if image is not None:
        print('image:\n', image[0][0])
    if bilinear is not None:
        print('bilinear:\n', bilinear[0][0])
    print('-' * 50)

class TextPositionEmbed(nn.Module):
    def __init__(self, seq_len, d_model=128, dropout=0.):
        super(TextPositionEmbed, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, text):
        text = text + Variable(self.pe[:,:text.size(1)], requires_grad=False)

        return self.dropout(text)

class ImagePatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, d_model=128, in_channels=3):
        super(ImagePatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)  # (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv_layer = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        B, C, H, W = image.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        image = self.conv_layer(image).flatten(2).transpose(1, 2)  # (B, H*W, D)
        return image

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(dropout),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model),
                                          nn.Dropout(dropout))

    def forward(self, x):
        return self.feed_forward(x)

class Image2TextGate(nn.Module):
    def __init__(self, n, d_model):
        super(Image2TextGate, self).__init__()
        self.n = n
        self.avg_pool = nn.AvgPool1d(kernel_size=n)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.select_para = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))

    def forward(self, image):
        B, N, C = image.shape
        assert N == self.n
        image = image * torch.view_as_complex(self.select_para)
        image = image.permute(0, 2, 1)  # (B, C, N)
        image = self.avg_pool(image.real)  # (B, C, 1)
        image = self.conv_layer(image)  # (B, C, 1)
        image = image.permute(0, 2, 1)  # (B, 1, C)
        return image

class Text2ImageGate(nn.Module):
    def __init__(self, s, d_model):
        super(Text2ImageGate, self).__init__()
        self.s = s
        self.avg_pool = nn.AvgPool1d(kernel_size=s)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.select_para = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))

    def forward(self, text):
        text = text * torch.view_as_complex(self.select_para)  # (B, S, C)
        text = text.permute(0, 2, 1)
        text = self.avg_pool(text.real)  # (B, C, 1)
        text = self.conv_layer(text)  # (B, C, 1)
        text = text.permute(0, 2, 1)  # (B, 1, C)
        return text

class ImageFrequencySelection(nn.Module):
    def __init__(self, s, d_model):
        super(ImageFrequencySelection, self).__init__()

        self.text_gate = Text2ImageGate(s, d_model)

    def forward(self, image, text):
        """
        image: (B, N, C)  N=h*w  in frequency domain
        """
        text_gate = self.text_gate(text)
        image = image * text_gate
        return image

class TextFrequencySelection(nn.Module):
    def __init__(self, n, d_model):
        super(TextFrequencySelection, self).__init__()

        self.image_gate = Image2TextGate(n, d_model)

    def forward(self, text, image):
        image_gate = self.image_gate(image)
        text = text * image_gate
        return text

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super(AddNorm, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(d_model, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x)
        x_ = x
        x = self.dropout(x)
        x = self.feed_forward(x) + x_
        x = self.norm2(x)
        return x

class FtLayer(nn.Module):
    def __init__(self, d_model, s, n, num_filter=2, dropout=0.,use_bank=True):
        super(FtLayer, self).__init__()
        self.s = s
        self.n = n
        self.use_bank = use_bank
        self.num_filter = num_filter

        self.text_weight = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))
        self.text_filter_bank = nn.Parameter(torch.randn(num_filter, s, d_model, 2, dtype=torch.float32))

        self.image_weight = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))
        self.image_filter_bank = nn.Parameter(torch.randn(num_filter, n, d_model, 2, dtype=torch.float32))

        # --- learnable frequency‑domain masks (real, broadcast across batch) ---
        # Shapes: (1, n, d_model) for image, (1, s, d_model) for text
        # Values will be squashed to (0, 1) range via sigmoid during forward pass
        self.image_freq_mask = nn.Parameter(torch.ones(1, n, d_model))
        self.text_freq_mask  = nn.Parameter(torch.ones(1, s, d_model))

        self.text_frequency_select = TextFrequencySelection(n, d_model)
        self.image_frenquency_select = ImageFrequencySelection(s, d_model)

        self.text_add_norm = AddNorm(d_model, dropout)
        self.image_add_norm = AddNorm(d_model, dropout)

    def filter(self, x, length, filter_bank, weight):
        if self.use_bank:
            power = (x * x) / length
            Y = []
            for k in range(self.num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * self.num_filter))
                Y.append(power * filter_bank[k] * cos)
            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        else:
            x = x * weight

        return x

    def forward(self, text, image, spatial_size=None):
        x_text = text
        B, S, D = text.shape
        assert S // 2 + 1 == self.s

        x_image = image
        B, N, C = image.shape
        assert N // 2 + 1 == self.n
        # if spatial_size:
        #     a, b = spatial_size
        # else:
        #     a = b = int(math.sqrt(N))

        # fft
        _text = torch.fft.rfft(text, dim=1, norm='ortho')
        _image = torch.fft.rfft(image, dim=1, norm='ortho')

        # frequency filter
        _text = self.filter(_text, self.s, torch.view_as_complex(self.text_filter_bank),
                            torch.view_as_complex(self.text_weight))
        _image = self.filter(_image, self.n, torch.view_as_complex(self.image_filter_bank),
                             torch.view_as_complex(self.image_weight))

        # apply learnable frequency masks
        _text  = _text  * torch.sigmoid(self.text_freq_mask)
        _image = _image * torch.sigmoid(self.image_freq_mask)


        # frequency select
        _text = self.text_frequency_select(_text, _image)
        _image = self.image_frenquency_select(_image, _text)

        # ifft
        text = torch.fft.irfft(_text, n=S, dim=1, norm='ortho')
        image = torch.fft.irfft(_image, n=N, dim=1, norm='ortho')
        # image = image.view(B, N, C)

        # add & norm
        text = self.text_add_norm(text + x_text)
        image = self.image_add_norm(image + x_image)

        return text, image

class FtBlock(nn.Module):
    def __init__(self, d_model, s, n, num_layer=1, num_filter=2, dropout=0.):
        """
        :param d_model:
        :param s: seq_len / 2 + 1
        :param h:
        :param w:
        :param n:
        """
        super(FtBlock, self).__init__()
        self.ft = nn.ModuleList([FtLayer(d_model, s, n, num_filter, dropout) for _ in range(num_layer)])

    def forward(self, text, image):
        for ft_layer in self.ft:
            text, image = ft_layer(text, image)

        return text, image

class Fusion(nn.Module):
    def __init__(self, d_model, act_layer=torch.tanh):
        super(Fusion, self).__init__()

        self.text_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.image_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.fusion_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.act_layer = act_layer

    def forward(self, text, image):
        alpha = self.js_div(text, image)

        fusion = torch.matmul(text, self.text_weight) + torch.matmul(image, self.image_weight)
        f = (1-alpha) * fusion + alpha * text + alpha * image

        return f

    @staticmethod
    def js_div(p, q):
        """
        Function that measures JS divergence between target and output logits:
        """
        M = (p + q) / 2
        kl1 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        gamma = 0.5 * kl1 + 0.5 * kl2
        return gamma

class MLP(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim, num_class, act_layer=nn.ReLU, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_dim, outputs_dim)
        self.norm2 = nn.LayerNorm(outputs_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(outputs_dim, num_class)

    def forward(self, x, need_tsne_data=False):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act_layer(x)
        features = None
        if need_tsne_data:
            features = x.detach().cpu().numpy()
        x = self.fc3(x)
        return x, features

class FSRU(nn.Module):
    def __init__(self, W, vocab_size, d_text, seq_len, img_size, patch_size, d_model,
                 num_filter, num_class, num_layer, dropout=0., mlp_ratio=4.):
        super(FSRU, self).__init__()

        # Text
        self.text_embed = nn.Embedding(vocab_size, d_text)
        self.text_embed.weight = nn.Parameter(torch.from_numpy(W))
        self.text_encoder = nn.Sequential(nn.Linear(d_text, d_model),
                                          nn.LayerNorm(d_model),
                                          TextPositionEmbed(seq_len, d_model, dropout))
        s = seq_len // 2 + 1

        # Image
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        self.fusion = Fusion(d_model)

        self.mlp = MLP(d_model, int(mlp_ratio*d_model), d_model, num_class, dropout=dropout)

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
            # trunc_normal_(m.weight, std=.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image):
        text = text.long()
        text = self.text_embed(text)  # (batch, seq, dim)
        text = self.text_encoder(text)

        image = image.to(torch.float32)
        image = self.img_patch_embed(image)
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)

        text = torch.max(text, dim=1)[0]
        image = torch.max(image, dim=1)[0]

        f = self.fusion(text, image)  # (batch, d_model)

        outputs = self.mlp(f)

        return text, image, outputs, f

def truncated_normal_fill(shape, mean=0., std=1., limit=2.):
    num_examples = 8
    tmp = torch.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)

def _init_weights(m, init_std=0.01):
    for key, val in m.named_parameters():
        if "weight" in key or "bias" in key:
            val.data.copy_(truncated_normal_fill(val.data.shape, std=init_std))

# FSRU模型集成UrbanCLIP组件
class FSRU_new(nn.Module):
    def __init__(self, W, vocab_size, d_text, seq_len, img_size, patch_size, d_model,
                 num_filter, num_class, num_layer, dropout=0., mlp_ratio=4.,
                 disable_tr: bool = False, disable_mim: bool = False):
        super(FSRU_new, self).__init__()
        self.disable_tr = disable_tr
        self.disable_mim = disable_mim

        # 文本
        self.text_embed = nn.Embedding(vocab_size, d_text)
        self.text_embed.weight = nn.Parameter(torch.from_numpy(W),)
        self.text_encoder = nn.Sequential(nn.Linear(d_text, d_model),
                                          nn.LayerNorm(d_model),
                                          TextPositionEmbed(seq_len, d_model, dropout))
        # text refinement block (can be disabled via disable_tr)
        self.text_refine = AddNorm(d_model, dropout)
        s = seq_len // 2 + 1

        # 图像
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        # 融合和MLP
        self.fusion = Fusion(d_model)
        self.mlp = MLP(d_model, int(mlp_ratio * d_model), d_model, num_class, dropout=dropout)

        # UrbanCLIP组件
        self.multimodal_layers = nn.ModuleList([
            nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=d_model, dim_head=64, heads=8, ff_mult=4)),
                Residual(CrossAttention(dim=d_model, dim_head=64, heads=8, parallel_ff=True, ff_mult=4))
            ]) for _ in range(num_layer)
        ])
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, W.shape[0])
        )

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image, need_tsne_data=False):
        text = text.long()
        text = self.text_embed(text)  # (batch, seq, dim)
        text_embed = text
        text = self.text_encoder(text)
        if not self.disable_tr:
            text = self.text_refine(text)

        image = image.to(torch.float32)
        image = self.img_patch_embed(image)
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)

        # 融合和MLP分支
        text_fusion = torch.max(text, dim=1)[0]
        image_fusion = torch.max(image, dim=1)[0]
        f = self.fusion(text_fusion, image_fusion)  # (batch, d_model)
        outputs, features = self.mlp(f, need_tsne_data)

        if need_tsne_data:
            return text_fusion, image_fusion, outputs, f, features, text_embed

        # 多模态层和to_logits分支（可关断）
        if not self.disable_mim:
            for attn_ff, cross_attn in self.multimodal_layers:
                text = attn_ff(text)
                text = cross_attn(text, image)
        logits = self.to_logits(text)

        return text_fusion, image_fusion, outputs, f, logits, text_embed


class FSRU_new_without_fusion(nn.Module):
    def __init__(self, W, vocab_size, d_text, seq_len, img_size, patch_size, d_model,
                 num_filter, num_class, num_layer, dropout=0., mlp_ratio=4.,
                 disable_tr: bool = False, disable_mim: bool = False):
        super(FSRU_new_without_fusion, self).__init__()
        self.disable_tr = disable_tr
        self.disable_mim = disable_mim

        # 文本
        self.text_embed = nn.Embedding(vocab_size, d_text)
        self.text_embed.weight = nn.Parameter(torch.from_numpy(W),)
        self.text_encoder = nn.Sequential(nn.Linear(d_text, d_model),
                                          nn.LayerNorm(d_model),
                                          TextPositionEmbed(seq_len, d_model, dropout))
        # text refinement block (can be disabled via disable_tr)
        self.text_refine = AddNorm(d_model, dropout)
        s = seq_len // 2 + 1

        # 图像
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        # 融合和MLP
        self.fusion = Fusion(d_model)
        self.mlp = MLP(2*d_model, int(mlp_ratio * 2 * d_model), d_model, num_class, dropout=dropout)

        # UrbanCLIP组件
        self.multimodal_layers = nn.ModuleList([
            nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=d_model, dim_head=64, heads=8, ff_mult=4)),
                Residual(CrossAttention(dim=d_model, dim_head=64, heads=8, parallel_ff=True, ff_mult=4))
            ]) for _ in range(num_layer)
        ])
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, W.shape[0])
        )

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image, need_tsne_data=False):
        text = text.long()
        text = self.text_embed(text)  # (batch, seq, dim)
        text_embed = text
        text = self.text_encoder(text)
        if not self.disable_tr:
            text = self.text_refine(text)

        image = image.to(torch.float32)
        image = self.img_patch_embed(image)
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)

        # 融合和MLP分支
        text_fusion = torch.max(text, dim=1)[0]
        image_fusion = torch.max(image, dim=1)[0]
        f = torch.cat((text_fusion, image_fusion), dim=1)
        # outputs = self.mlp(f)
        outputs, features = self.mlp(f, need_tsne_data)

        if need_tsne_data:
            return text_fusion, image_fusion, outputs, f, features, text_embed

        # 多模态层和to_logits分支（可关断）
        if not self.disable_mim:
            for attn_ff, cross_attn in self.multimodal_layers:
                text = attn_ff(text)
                text = cross_attn(text, image)
        logits = self.to_logits(text)

        return text_fusion, image_fusion, outputs, f, logits, text_embed

def main():
    # Define model parameters
    model_params = {
        'W': torch.randn(10000, 512).numpy(),  # Example pre-trained word vectors
        'vocab_size': 10000,
        'd_text': 512,
        'seq_len': 128,
        'img_size': 256,
        'patch_size': 16,
        'd_model': 512,
        'num_filter': 2,
        'num_class': 10,
        'num_layer': 6,
        'dropout': 0.1,
        'mlp_ratio': 4.0
    }

    # Create the FSRU model
    model = FSRU(**model_params)

    # Print the model structure
    print(model)

    # 定义模型参数
    model_params = {
        'W': torch.randn(10000, 512).numpy(),  # 示例预训练词向量
        'vocab_size': 10000,
        'd_text': 512,
        'seq_len': 128,
        'img_size': 256,
        'patch_size': 16,
        'd_model': 512,
        'num_filter': 2,
        'num_class': 10,
        'num_layer': 6,
        'dropout': 0.1,
        'mlp_ratio': 4.0
    }

    # 创建FSRU_new模型
    model = FSRU_new(**model_params)

    # 输出模型结构
    print(model)

if __name__ == "__main__":
    main()