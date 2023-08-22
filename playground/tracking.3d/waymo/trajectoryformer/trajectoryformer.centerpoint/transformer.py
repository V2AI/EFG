from torch import nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, token, src, pos=None):
        token_list = []
        output = src
        for layer in self.layers:
            output, token = layer(token, output, pos=pos)
            token_list.append(token)
        if self.norm is not None:
            output = self.norm(output)

        return token_list


class TransformerEncoderGlobalLocal(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        token_list = []
        output = src
        for layer in self.layers:
            output = layer(output)
            token_list.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return token_list


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, token, src, pos=None):
        src_mix = self.point_attn(
            query=src.permute(1, 0, 2),
            key=src.permute(1, 0, 2),
            value=src.permute(1, 0, 2),
        )[0]
        src_mix = src_mix.permute(1, 0, 2)
        src = src + self.dropout1(src_mix)
        src = self.norm1(src)
        src_mix = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src_mix)
        src = self.norm2(src)
        src_summary = self.self_attn(
            token.permute(1, 0, 2), key=src.permute(1, 0, 2), value=src.permute(1, 0, 2)
        )[0]
        src_summary = src_summary.permute(1, 0, 2)
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)

        return src, token

    def forward(self, token, src, pos=None):
        return self.forward_post(token, src, pos)


class TransformerEncoderLayerGlobalLocal(nn.Module):
    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0):
        super().__init__()

        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.local_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src):
        (
            bs,
            num_track,
            candi,
        ) = (
            src.shape[0],
            src.shape[1],
            src.shape[2],
        )
        src_global = src.reshape(bs, -1, src.shape[-1])
        src_mix = self.global_attn(
            query=src_global.permute(1, 0, 2),
            key=src_global.permute(1, 0, 2),
            value=src_global.permute(1, 0, 2),
        )[0]
        src_mix = src_mix.permute(1, 0, 2)
        src_global = self.ffn1(src_global, src_mix)
        src_local = src_global.reshape(bs, num_track, candi, -1).reshape(
            bs * num_track, candi, -1
        )
        src_mix = self.local_attn(
            query=src_local.permute(1, 0, 2),
            key=src_local.permute(1, 0, 2),
            value=src_local.permute(1, 0, 2),
        )[0]
        src_mix = src_mix.permute(1, 0, 2)
        src_local = self.ffn2(src_local, src_mix)

        return src_local.reshape(bs, num_track, candi, -1)

    def forward(self, src):
        return self.forward_post(src)


class FFN(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        dout=None,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
