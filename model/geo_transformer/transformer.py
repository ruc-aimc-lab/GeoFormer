import copy
import torch
import torch.nn as nn

from utils.common_utils import sample_descriptors
from model.geo_transformer.geo_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead, linear=False):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if linear:
            self.attention = LinearAttention()
        else:
            self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.Tanh(),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """


        bs = x.size(0)

        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, x_mask, source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)


        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class GeoTransformer(nn.Module):

    def __init__(self, config, layer_names, d_model, linear=True):
        super(GeoTransformer, self).__init__()

        self.config = config
        self.d_model = d_model
        self.layer_names = layer_names
        self.nhead = config['nhead']
        encoder_layer = LoFTREncoderLayer(self.d_model, self.nhead, linear)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.norm = nn.LayerNorm(self.d_model)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, kp0_cross, kp1_cross, h0, w0, h1, w1, scale,
                mask_self0=None, mask_self1=None, mask_cross0=None, mask_cross1=None):
        """
        inplace operation for feat0 and feat1
        :param feat0:
        :param feat1:
        :param kp0_cross: the corresponding areas in feat0 of each keypoint in feat1
        :param kp1_cross: the corresponding areas in feat1 of each keypoint in feat0
        :param h0: size of feat0 (2D)
        :param w0:
        :param h1: size of feat1 (2D)
        :param w1:
        :param scale: feat(2D) to raw size
        :param mask_self0: used for self-attention on feat0
        :param mask_self1:
        :param mask_cross0: illegal area of kp0_cross
        :param mask_cross1:
        :return:
        """
        assert self.d_model == feat0.size(2) , "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                for step in range(len(feat0)):
                    feat0_at = feat0[step]
                    feat1_at = feat1[step]
                    mask0_at = mask_self0[step]
                    mask1_at = mask_self1[step]
                    if mask0_at.sum() > 0:
                        feat0_at = layer(feat0_at.unsqueeze(0), feat0_at[mask0_at].unsqueeze(0))[0]

                    if mask1_at.sum() > 0:
                        feat1_at = layer(feat1_at.unsqueeze(0), feat1_at[mask1_at].unsqueeze(0))[0]

                    feat0[step] = feat0_at
                    feat1[step] = feat1_at
            elif name == 'cross':
                feat0_map = feat0.view(len(feat0), h0, w0, feat0.shape[-1]).permute(0, 3, 1, 2)
                feat1_map = feat1.view(len(feat1), h1, w1, feat1.shape[-1]).permute(0, 3, 1, 2)
                feat0_cross = sample_descriptors(kp0_cross, feat0_map, scale)
                feat1_cross = sample_descriptors(kp1_cross, feat1_map, scale)
                for step in range(len(feat0_cross)):
                    feat0_at = feat0[step].unsqueeze(1)
                    feat1_at = feat1[step].unsqueeze(1)
                    if feat1_cross[step] is not None:
                        feat0_at_cross = feat0_cross[step]
                        feat1_at_cross = feat1_cross[step]
                        feat0_at = layer(feat0_at, feat1_at_cross, None, mask_cross1[step])
                        feat1_at = layer(feat1_at, feat0_at_cross, None, mask_cross0[step])
                    feat0[step] = feat0_at.squeeze(1)
                    feat1[step] = feat1_at.squeeze(1)

                # feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        # feat0 = self.norm(feat0)
        # feat1 = self.norm(feat1)
        return feat0, feat1


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config, layer_names, d_model, linear=True):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = d_model
        self.layer_names = layer_names
        self.nhead = config['nhead']
        encoder_layer = LoFTREncoderLayer(self.d_model, self.nhead, linear)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.norm = nn.LayerNorm(self.d_model)
        self.final_proj = nn.Conv1d(
            self.d_model, self.d_model,
            kernel_size=1, bias=True)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        # feat0 = self.norm(feat0)
        # feat1 = self.norm(feat1)
        assert self.d_model == feat0.size(2) , "the feature number of src and transformer must be equal"
        # A0, A1 = None, None
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1

        # return feat0, feat1
        # feat0, feat1 = self.final_proj(feat0.permute(0, 2, 1)), self.final_proj(feat1.permute(0, 2, 1))
        # return feat0.permute(0, 2, 1), feat1.permute(0, 2, 1)

class LocalFeatureTransformer_my(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config, layer_names, d_model, linear=True):
        super(LocalFeatureTransformer_my, self).__init__()

        self.config = config
        self.d_model = d_model
        self.layer_names = layer_names
        self.nhead = config['nhead']
        encoder_layer = LoFTREncoderLayer(self.d_model, self.nhead, linear)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.norm = nn.LayerNorm(self.d_model)
        # self.final_proj = nn.Conv1d(
        #     self.d_model, self.d_model,
        #     kernel_size=1, bias=True)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask_self0=None, mask_self1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        # feat0 = self.norm(feat0)
        # feat1 = self.norm(feat1)
        assert self.d_model == feat0.size(2) , "the feature number of src and transformer must be equal"
        # A0, A1 = None, None
        # feat0_map = feat0.view(len(feat0), h0, w0, feat0.shape[-1]).permute(0, 3, 1, 2)
        # feat1_map = feat1.view(len(feat1), h1, w1, feat1.shape[-1]).permute(0, 3, 1, 2)
        # feat0_cross = sample_descriptors(kp0_cross, feat0_map, scale)
        # feat1_cross = sample_descriptors(kp1_cross, feat1_map, scale)
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                for step in range(len(feat0)):
                    feat0_at = feat0[step]
                    feat1_at = feat1[step]
                    mask0_at = mask_self0[step]
                    mask1_at = mask_self1[step]
                    cross_feat = None
                    if mask0_at.sum() > 0:
                        cross_feat = torch.cat((feat0_at[mask0_at], feat1_at[mask1_at]))
                    if cross_feat is not None:
                        feat0_at = layer(feat0_at.unsqueeze(0), cross_feat.unsqueeze(0))[0]
                        feat1_at = layer(feat1_at.unsqueeze(0), cross_feat.unsqueeze(0))[0]

                    feat0[step] = feat0_at
                    feat1[step] = feat1_at
            elif name == 'cross':
                for step in range(len(feat0)):
                    feat0_at = feat0[step]
                    feat1_at = feat1[step]
                    mask0_at = mask_self0[step]
                    mask1_at = mask_self1[step]
                    if mask0_at.sum() > 0:
                        feat0_at = layer(feat0_at.unsqueeze(0), feat1_at[mask1_at].unsqueeze(0))[0]

                    if mask1_at.sum() > 0:
                        feat1_at = layer(feat1_at.unsqueeze(0), feat0_at[mask0_at].unsqueeze(0))[0]

                    feat0[step] = feat0_at
                    feat1[step] = feat1_at

                # feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1
        # feat0, feat1 = self.final_proj(feat0.permute(0, 2, 1)), self.final_proj(feat1.permute(0, 2, 1))
        # return feat0.permute(0, 2, 1), feat1.permute(0, 2, 1)
