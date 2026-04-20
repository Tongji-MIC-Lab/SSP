# Modified from mmdet/models/plugins/msdeformattn_pixel_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from mmcv.cnn import (ConvModule, caffe2_xavier_init, xavier_init)
from mmcv.runner import BaseModule, ModuleList
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.core.anchor import MlvlPointGenerator
from mmdet.models.utils import SinePositionalEncoding

from .tranformer_decoder import FeedForward, MultiHeadAttention, PosEncoding
from .backbones.util import Fold, Unfold, Attention, LayerNorm2d
from einops import rearrange, repeat

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)

        weight = torch.zeros(dim, 1, kernel_size, kernel_size)
        for i in range(dim):
            for j in range(kernel_size):
                weight[i, 0, j, j] = 1
                
        self.conv_constant = nn.Parameter(weight)
        self.conv_constant.requires_grad = False
        
    def forward(self, x):
        return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
        # return x + self.conv_constant(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x

class AdaptiveSelection(nn.Module):
    def __init__(self, embed_dim, heads, mlp_expand=4, attn_expand=4, dropout=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        # img_size: 448, 480
        self.scale_rate = [64,16,4]
        self.scale_size = [8, 4, 2]
        self.ln = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(len(self.scale_rate))])
        self.mlps = nn.ModuleList([nn.Linear(self.scale_rate[i], self.scale_rate[i]) for i in range(len(self.scale_rate))])
        self.pixel_shuffles = nn.ModuleList([nn.PixelShuffle(self.scale_size[i]) for i in range(len(self.scale_size))])
        self.relu = nn.ReLU(inplace=True)
        self.fusion = nn.Conv2d(embed_dim // 64 + embed_dim // 16 + embed_dim // 4, 1, kernel_size=3, padding=1)
        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, multiscale_feats):
        # scale_factor: 8,4,2
        feats = []
        for i in range(len(multiscale_feats)):
            b, c, h, w = multiscale_feats[i].shape
            spatial_info = rearrange(multiscale_feats[i], 'b c h w -> b (h w) c')
            spatial_info = self.ln[i](spatial_info)
            gamma = self.ln[i].weight / sum(self.ln[i].weight)
            spatial_info = rearrange(spatial_info * gamma, 'b (h w) c -> b c h w', h=h, w=w)
            spatial_info = torch.sum(spatial_info, dim=1) # b h w
            spatial_info = spatial_info.repeat_interleave(self.scale_size[i], dim=-1)
            spatial_info = spatial_info.repeat_interleave(self.scale_size[i], dim=-2)
            spatial_info = spatial_info.unsqueeze(1)

            feat = self.pixel_shuffles[i](multiscale_feats[i])
            b, c, h, w = feat.shape # b c/r h w
            feat = feat.reshape(b, c, self.scale_rate[i], (h*w) // self.scale_rate[i]).transpose(-2,-1)
            feat = self.relu(self.mlps[i](feat)).transpose(-2,-1)
            feat = feat.reshape(b, c, h*w).reshape(b, c, h, w)
            feat = spatial_info * feat
            feats.append(feat)

        x = self.fusion(torch.cat(feats, dim=1))
        x = torch.sigmoid(x)

        return x


class SAttnLayer(nn.Module):
    def __init__(self, embed_dim, heads, mlp_expand=4, attn_expand=4, dropout=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.s2p_cross_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)
        self.mlp = FeedForward(embed_dim, embed_dim*mlp_expand, dropout, act='relu')
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)     

        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, pixel_feats, superpixel_feats):

        pixel_feats = pixel_feats + self.s2p_cross_attn(q=pixel_feats, k=superpixel_feats, v=superpixel_feats)[0]
        pixel_feats = self.norm1(pixel_feats)

        pixel_feats = pixel_feats + self.mlp(pixel_feats)
        pixel_feats = self.norm2(pixel_feats)

        return pixel_feats


class VSAttnLayer(nn.Module):
    def __init__(self, embed_dim, heads, mlp_expand=4, attn_expand=4, dropout=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.n_iter = 1
        self.stoken_size = [2,4,8,16]
        self.poses, self.poses_norms, self.stoken_refines, self.norms, self.norms_mlp, self.mlps = [], [], [], [], [], []
        self.unfold, self.fold = Unfold(3), Fold(3)
        self.v_norm1 = norm_layer(embed_dim)
        self.v_norm2 = norm_layer(embed_dim)
        self.mlp = FeedForward(embed_dim, embed_dim*mlp_expand, dropout, act='relu')

        for idx in range(len(self.stoken_size)):
            pose = ResDWC(embed_dim)
            poses_norms = LayerNorm2d(embed_dim)
            attn = Attention(embed_dim, num_heads=heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
            layernorm = LayerNorm2d(embed_dim)
            layernorm_mlp = LayerNorm2d(embed_dim)
            self.poses.append(pose)
            self.poses_norms.append(poses_norms)
            self.stoken_refines.append(attn)
            self.norms.append(layernorm)
            self.norms_mlp.append(layernorm_mlp)
            # self.mlps.append(mlp)

        self.poses = nn.ModuleList(self.poses)
        self.poses_norms = nn.ModuleList(self.poses_norms)
        self.stoken_refines = nn.ModuleList(self.stoken_refines)
        self.norms = nn.ModuleList(self.norms)
        self.norms_mlp = nn.ModuleList(self.norms_mlp)

        self.l2s_cross_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)

        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, superpixel_features, lang, prompts=None, lang_mask=None, vis_pos=None, vis_padding_mask=None, return_attn=False, **kwargs):
        
        ori_shapes = list()
        padding_shapes = list()
        stoken_features = list()
        affinity_matrixs = list()
        stoken_feature_shapes = list()

        for stage in range(len(self.stoken_size)):
            pixel_feature = superpixel_features[stage]
            pixel_feature = self.poses_norms[stage](self.poses[stage](pixel_feature))
            # super pixel features
            B, C, H0, W0 = pixel_feature.shape
            h, w = self.stoken_size[stage], self.stoken_size[stage]
            ori_shapes.append([H0, W0])
            pad_l = pad_t = 0
            pad_r = (w - W0 % w) % w
            pad_b = (h - H0 % h) % h
            if pad_r > 0 or pad_b > 0:
                pixel_feature = F.pad(pixel_feature, (pad_l, pad_r, pad_t, pad_b))
                
            _, _, H, W = pixel_feature.shape
            padding_shapes.append([H, W])
            
            hh, ww = H//h, W//w
            
            stoken_feature = F.adaptive_avg_pool2d(pixel_feature, (hh, ww)) # (B, C, hh, ww)
            pixel_feature = pixel_feature.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
            
            with torch.no_grad():
                for idx in range(self.n_iter):
                    stoken_feature = self.unfold(stoken_feature) # (B, C*9, hh*ww)
                    stoken_feature = stoken_feature.transpose(1, 2).reshape(B, hh*ww, C, 9)
                    scale = pixel_feature.shape[-1] ** -0.5
                    affinity_matrix = pixel_feature @ stoken_feature * scale # (B, hh*ww, h*w, 9)
                    affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                    affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                    affinity_matrix_sum = self.fold(affinity_matrix_sum)
                    stoken_feature = pixel_feature.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    stoken_feature = self.fold(stoken_feature.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)
                    stoken_feature = stoken_feature/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)

                stoken_feature = self.stoken_refines[stage](stoken_feature)
                stoken_features.append(stoken_feature.flatten(2).permute(0, 2, 1))

        stoken_features = torch.cat(stoken_features, dim=1) # (B, hw*3, C)
        stoken_features = stoken_features + self.l2s_cross_attn(q=stoken_features, k=lang, v=lang, attention_mask=lang_mask)[0]
        stoken_features = self.v_norm1(stoken_features)

        stoken_features = stoken_features + self.mlp(stoken_features)
        stoken_features = self.v_norm2(stoken_features)

        return stoken_features

class VLMSDeformAttnLayer(nn.Module):
    def __init__(self, embed_dim, heads, num_levels, 
                 num_points=4, im2col_step=16, 
                 mlp_expand=4, attn_expand=4,
                 dropout=0., norm_layer=nn.LayerNorm, with_gamma=False, init_value=1.0):
        super().__init__()
        self.vis_self_attn = MultiScaleDeformableAttention(embed_dim, num_heads=heads, num_levels=num_levels, num_points=num_points, 
                                                           im2col_step=im2col_step, dropout=dropout, batch_first=True)
        self.p2l_cross_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)
        self.l2p_cross_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                 dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)
        self.lang_self_attn = MultiHeadAttention(embed_dim, embed_dim, embed_dim*attn_expand, num_heads=heads, 
                                                dropout=dropout, clamp_min_for_underflow=True, clamp_max_for_overflow=True)

        self.vis_mlp = FeedForward(embed_dim, embed_dim*mlp_expand, dropout, act='relu')
        self.lang_mlp = FeedForward(embed_dim, embed_dim*mlp_expand, dropout, act='relu')

        self.with_gamma = with_gamma
        if with_gamma:
            self.gamma_v2l = nn.Parameter(init_value*torch.ones((1, 1, embed_dim)), requires_grad=True)
            self.gamma_l2v = nn.Parameter(init_value*torch.ones((1, 1, embed_dim)), requires_grad=True)

        self.v_norm1 = norm_layer(embed_dim)
        self.v_norm2 = norm_layer(embed_dim)
        self.v_norm3 = norm_layer(embed_dim)
        self.v_drop3 = nn.Dropout(dropout)

        self.l_norm1 = norm_layer(embed_dim)
        self.l_norm2 = norm_layer(embed_dim)
        self.l_norm3 = norm_layer(embed_dim)
        self.l_drop1 = nn.Dropout(dropout)
        self.l_drop2 = nn.Dropout(dropout)
        self.l_drop3 = nn.Dropout(dropout)
        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
        self.vis_self_attn.init_weights() # init_weights defined in MultiScaleDeformableAttention

    def forward(self, vis, lang, prompts=None, lang_mask=None, vis_pos=None, vis_padding_mask=None, return_attn=False, **kwargs):
        '''
            - vis: :math:`(N, L, E)` where L is the sequence length, N is the batch size, E is
            the embedding dimension.
            - lang: :math:`(N, S, E)`, where S is the sequence length, N is the batch size, E is
            the embedding dimension.
            - prompts: :math:`(N, P, E)` where P is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - lang_mask :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the zero positions will be ignored while the position
            with the non-zero positions will be unchanged.
            - vis_pos: :math:`(N, L, E)` where L is the sequence length, N is the batch size, E is
            the embedding dimension.
            - vis_padding_mask :math:`(N, L)` where N is the batch size, L is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged.
        '''
        prompted_lang = lang
        pixel_vis = vis

        # V2L cross attn
        _prompted_lang = self.p2l_cross_attn(q=prompted_lang, k=PosEncoding(pixel_vis, vis_pos), v=pixel_vis, attention_mask=(1-vis_padding_mask.byte()))[0]
        if self.with_gamma:
            _prompted_lang = _prompted_lang * self.gamma_v2l
        prompted_lang = prompted_lang + self.l_drop1(_prompted_lang)
        prompted_lang = self.l_norm1(prompted_lang)

        # Linguistic self attn
        _prompted_lang = self.lang_self_attn(q=prompted_lang, k=prompted_lang, v=prompted_lang, attention_mask=lang_mask)[0]
        prompted_lang = prompted_lang + self.l_drop2(_prompted_lang)
        prompted_lang = self.l_norm2(prompted_lang)

        # Linguistic FFN
        _prompted_lang = self.lang_mlp(prompted_lang)
        prompted_lang = prompted_lang + self.l_drop3(_prompted_lang)
        prompted_lang = self.l_norm3(prompted_lang)

        # L2V corss attn
        if return_attn:
            _p_vis, l2v_attn = self.l2p_cross_attn(q=PosEncoding(pixel_vis, vis_pos), k=prompted_lang, v=prompted_lang, attention_mask=lang_mask)
            _p_vis = _p_vis + pixel_vis
        else:
            _p_vis = self.l2p_cross_attn(q=PosEncoding(pixel_vis, vis_pos), k=prompted_lang, v=prompted_lang, attention_mask=lang_mask)[0]
            _p_vis = _p_vis + pixel_vis

        if self.with_gamma:
            _p_vis = _p_vis * self.gamma_l2v
        _p_vis = self.v_norm2(_p_vis)

        # Visual MSDeformable self attn
        with torch.cuda.amp.autocast(enabled=False):
            vis = self.vis_self_attn(_p_vis.float(), value=_p_vis.float(), query_pos=vis_pos.float(), key_padding_mask=vis_padding_mask, **kwargs)
        vis = self.v_norm1(vis)

        # Visual FFN
        _vis = self.vis_mlp(vis)
        vis = vis + self.v_drop3(_vis)
        vis = self.v_norm3(vis)

        lang = prompted_lang
        prompts = None
        if return_attn:
            return vis, lang, prompts, l2v_attn
        return vis, lang, prompts
    
class VLMSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.
    Use learnable prompt, prompt.mean() to predict background
    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`mmcv.ConfigDict` | dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`mmcv.ConfigDict` | dict): Config for activation.
            Defaults to dict(type='ReLU').
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 lang_in_channels=768,
                 strides=[4, 8, 16, 32],
                 feat_channels=256,
                 out_channels=256,
                 num_enc_layers=6,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 im2col_step=16,
                 dropout=0.0,
                 mlp_expand=4,
                 with_prompts=True,
                 num_prompts=10,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 ):
        super().__init__()
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        input_conv_supertoken_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)

        sp_channels = [64, 128, 320, 512]
        # sp_channels = [96, 192, 384, 512]
        # sp_channels = [128, 320, 512, 512]
        for i in range(self.num_input_levels):
            input_conv = ConvModule(
                sp_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_supertoken_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)
        self.input_convs_supertoken = ModuleList(input_conv_supertoken_list)
        self.num_enc_layers = num_enc_layers
        
        self.encoder = nn.ModuleList([VLMSDeformAttnLayer(feat_channels, heads=num_heads, num_levels=num_levels, num_points=num_points,
                                    im2col_step=im2col_step, mlp_expand=mlp_expand, dropout=dropout, with_gamma=True) for i in range(self.num_enc_layers)])
        self.encoder_supertoken = VSAttnLayer(feat_channels, heads=num_heads, mlp_expand=mlp_expand, dropout=dropout)
        self.s2pformer = SAttnLayer(feat_channels, heads=num_heads, mlp_expand=mlp_expand, dropout=dropout)
        self.adaptive_selection = AdaptiveSelection(feat_channels, heads=num_heads, mlp_expand=mlp_expand, dropout=dropout)

        self.postional_encoding = SinePositionalEncoding(num_feats=feat_channels//2, normalize=True)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)
        # language features
        self.lang_in_linear = nn.Linear(lang_in_channels, feat_channels)
        self.lang_in_norm = nn.LayerNorm(feat_channels)
        # visual prompts
        self.with_prompts = with_prompts
        self.num_prompts = num_prompts
        if with_prompts:
            self.vis_prompts = nn.Embedding(num_prompts, feat_channels)

        # fpn-like structure
        # self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        self.lateral_conv = ConvModule(
            in_channels[0],
            feat_channels,
            kernel_size=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg,
            act_cfg=None)
        for i in range(self.num_input_levels):
            if i == 0:
                output_conv = ConvModule(
                    feat_channels,
                    feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            else:
                output_conv = ConvModule(
                    feat_channels + feat_channels//4,
                    feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.output_convs.append(output_conv)

        self.point_generator = MlvlPointGenerator(strides)
        self.mask_feature = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1)
            )
        self.lang_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels)
            )
        
    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')
            
        trunc_normal_(self.lang_in_linear.weight, std=.02)
        nn.init.constant_(self.lang_in_linear.bias, 0)

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        self.mask_feature.apply(caffe2_xavier_init)
        self.lang_embed.apply(caffe2_xavier_init)

    def forward(self, feats, supertoken_feats, lang, lang_mask, return_attn=False):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c_i, h, w).
            lang (Tensor): shape (batch_size, c_l, L).
            lang_mask (Tensor): shape (batch_size, L).

        Returns:
            tuple: A tuple containing the following:

            - mask_feature (Tensor): shape (batch_size, c, h, w).
            - multi_scale_features (list[Tensor]): Multi scale \
                    features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            h, w = feat.shape[-2:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (batch_size, h_i * w_i, c)
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points)

        # shape (batch_size, total_num_query), total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (batch_size, total_num_query, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=1)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        
        extend_l_mask = lang_mask
        prompts = None
        # shape (batch_size, num_total_query, c)
        memory = encoder_inputs
        lang = self.lang_in_norm(self.lang_in_linear(lang)) # [B, N_l, C]
        lang_res = lang
        attns = []
        for i in range(self.num_enc_layers):
            if return_attn:
                memory, lang, prompts, l2v_attn = self.encoder[i](
                    vis=memory,
                    lang=lang,
                    lang_mask=extend_l_mask,
                    prompts=prompts,
                    vis_pos=level_positional_encodings,
                    vis_padding_mask=padding_masks,
                    return_attn=return_attn,
                    spatial_shapes=spatial_shapes,
                    reference_points=reference_points,
                    level_start_index=level_start_index,
                    valid_radios=valid_radios)
                attns.append(l2v_attn)
            else:
                memory, lang, prompts = self.encoder[i](
                    vis=memory,
                    lang=lang,
                    lang_mask=extend_l_mask,
                    prompts=prompts,
                    vis_pos=level_positional_encodings,
                    vis_padding_mask=padding_masks,
                    spatial_shapes=spatial_shapes,
                    reference_points=reference_points,
                    level_start_index=level_start_index,
                    valid_radios=valid_radios)
        # (batch_size, num_total_query, c) -> (batch_size, c, num_total_query)

        # performing on super-pixel token
        encoder_supertoken_list = []
        for idx in range(len(supertoken_feats)):
            supertoken_feat_projected = self.input_convs_supertoken[idx](supertoken_feats[idx])
            encoder_supertoken_list.append(supertoken_feat_projected)

        supertoken_feats = self.encoder_supertoken(encoder_supertoken_list[::-1], lang_res)

        memory = self.s2pformer(memory, supertoken_feats) # b n3+n2+n1 c
        memory = memory.transpose(1,2) # b c n1

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [x.reshape(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for i, x in enumerate(outs)]
        outs = self.adaptive_selection(outs) # b 1 h w

        cur_feat = outs * self.lateral_conv(feats[0])
        y = self.output_convs[0](cur_feat)

        mask_feature = self.mask_feature(y) # [B, C, H, W]
        lang_g = self.lang_embed(lang[:, :2]) # [B, 2, C], [CLS] embeddings
        lang_cls = lang_g
        mask_pred = torch.einsum('bqc,bchw->bqhw', lang_cls, mask_feature) # [B, 2, H, W]

        if return_attn:
            attns = torch.cat(attns, dim=0)
            attns = attns.transpose(-2, -1) # [B*num_layers, num_heads, n_lang, n_vis]
            attns = torch.mean(attns, dim=1) # [B*num_layers, n_lang, n_vis]
            attns = torch.split(attns, num_query_per_level, dim=-1)
            attns = [x.reshape(batch_size*self.num_enc_layers, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for i, x in enumerate(attns)]
            return mask_pred, attns
        return mask_pred
