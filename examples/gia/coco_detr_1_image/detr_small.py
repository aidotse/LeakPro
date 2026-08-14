"""A compact, hackable DETR, for architecture surgery experiments on gradient inversion.

This follows the original DETR (Carion et al. 2020, https://github.com/facebookresearch/detr) in
structure and forward pass, but is written as a single file with every architectural choice exposed
as a constructor argument, so backbone family, backbone depth, transformer depth, width and
normalization can all be varied. Two deliberate departures from the original:

* BatchNorm layers are ordinary trainable ``nn.BatchNorm2d``, not the frozen buffer-only variant.
  Their gradients therefore appear in the shared federated update, which is what makes them
  interesting for gradient inversion.
* The encoder and decoder stacks may have zero layers. With no decoder the class and box heads read
  the backbone feature map cells directly, one prediction per spatial position, which turns DETR
  into a dense single stage detector and lets us isolate what the transformer contributes.

Parameter names mirror ``transformers.DetrForObjectDetection`` (``model.backbone``,
``model.input_projection``, ``model.query_position_embeddings``, ``model.encoder``,
``model.decoder``, ``class_labels_classifier``, ``bbox_predictor``) so the same grouping and
analysis code applies to both this model and the stock pretrained one.
"""
import math
from dataclasses import dataclass

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as f
from transformers import DetrConfig

# Output channels of each of the four residual stages, per torchvision ResNet family.
RESNET_STAGE_CHANNELS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
}


@dataclass
class DetrSmallOutput:
    """Minimal stand-in for transformers' DetrObjectDetectionOutput.

    Only the two fields the set prediction loss consumes are provided, which keeps this model
    interchangeable with the pretrained one as far as ComputeLoss is concerned.
    """

    logits: Tensor
    pred_boxes: Tensor


class PositionEmbeddingSine(nn.Module):
    """Fixed 2D sine position embedding, as in the original DETR.

    Row and column indices are normalized to [0, scale] and expanded into sine/cosine features.
    The original takes a padding mask; here every pixel is valid because the attack optimizes a
    single fixed size image, so the mask reduces to all ones.
    """

    def __init__(self: "PositionEmbeddingSine", num_pos_feats: int, temperature: int = 10000,
                 scale: float = 2 * math.pi) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def forward(self: "PositionEmbeddingSine", x: Tensor) -> Tensor:
        """Build the position embedding for a (batch, channels, height, width) feature map."""
        batch, _, height, width = x.shape
        eps = 1e-6
        y_embed = torch.arange(1, height + 1, dtype=torch.float32, device=x.device)[None, :, None]
        x_embed = torch.arange(1, width + 1, dtype=torch.float32, device=x.device)[None, None, :]
        y_embed = (y_embed / (height + eps) * self.scale).repeat(batch, 1, width)
        x_embed = (x_embed / (width + eps) * self.scale).repeat(batch, height, 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MultiheadAttention(nn.Module):
    """Multi head attention with separate q, k, v projections.

    Written out rather than using nn.MultiheadAttention so that every weight is a plainly named
    Linear, which keeps the per parameter gradient analysis readable and the module easy to modify.
    """

    def __init__(self: "MultiheadAttention", embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _heads(self: "MultiheadAttention", x: Tensor) -> Tensor:
        """Split the channel dimension into heads."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self: "MultiheadAttention", query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Attend from query to key/value, all of shape (batch, length, embed_dim)."""
        batch, length, _ = query.shape
        q = self._heads(self.q_proj(query) * self.scaling)
        k = self._heads(self.k_proj(key))
        v = self._heads(self.v_proj(value))
        weights = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        weights = f.dropout(weights, p=self.dropout, training=self.training)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    """DETR encoder layer: self attention over image tokens, then a feed forward block."""

    def __init__(self: "EncoderLayer", d_model: int, nheads: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nheads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout
        self.pre_norm = pre_norm

    def _drop(self: "EncoderLayer", x: Tensor) -> Tensor:
        """Apply dropout."""
        return f.dropout(x, p=self.dropout, training=self.training)

    def _ff(self: "EncoderLayer", x: Tensor) -> Tensor:
        """Feed forward block."""
        return self.linear2(self._drop(f.relu(self.linear1(x))))

    def forward(self: "EncoderLayer", src: Tensor, pos: Tensor) -> Tensor:
        """Encode image tokens, with the position embedding added to queries and keys only."""
        if self.pre_norm:
            hidden = self.norm1(src)
            src = src + self._drop(self.self_attn(hidden + pos, hidden + pos, hidden))
            return src + self._drop(self._ff(self.norm2(src)))
        src = self.norm1(src + self._drop(self.self_attn(src + pos, src + pos, src)))
        return self.norm2(src + self._drop(self._ff(src)))


class DecoderLayer(nn.Module):
    """DETR decoder layer: query self attention, cross attention to the encoder, feed forward."""

    def __init__(self: "DecoderLayer", d_model: int, nheads: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nheads, dropout)
        self.cross_attn = MultiheadAttention(d_model, nheads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = dropout
        self.pre_norm = pre_norm

    def _drop(self: "DecoderLayer", x: Tensor) -> Tensor:
        """Apply dropout."""
        return f.dropout(x, p=self.dropout, training=self.training)

    def _ff(self: "DecoderLayer", x: Tensor) -> Tensor:
        """Feed forward block."""
        return self.linear2(self._drop(f.relu(self.linear1(x))))

    def forward(self: "DecoderLayer", target: Tensor, memory: Tensor, pos: Tensor, query_pos: Tensor) -> Tensor:
        """Decode object queries against the encoded image."""
        if self.pre_norm:
            hidden = self.norm1(target)
            target = target + self._drop(self.self_attn(hidden + query_pos, hidden + query_pos, hidden))
            hidden = self.norm2(target)
            target = target + self._drop(self.cross_attn(hidden + query_pos, memory + pos, memory))
            return target + self._drop(self._ff(self.norm3(target)))
        target = self.norm1(target + self._drop(self.self_attn(target + query_pos, target + query_pos, target)))
        target = self.norm2(target + self._drop(self.cross_attn(target + query_pos, memory + pos, memory)))
        return self.norm3(target + self._drop(self._ff(target)))


class TransformerEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self: "TransformerEncoder", num_layers: int, **layer_kwargs: object) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(**layer_kwargs) for _ in range(num_layers)])

    def forward(self: "TransformerEncoder", src: Tensor, pos: Tensor) -> Tensor:
        """Run every encoder layer in turn."""
        for layer in self.layers:
            src = layer(src, pos)
        return src


class TransformerDecoder(nn.Module):
    """Stack of decoder layers, with the final LayerNorm the original applies to the output."""

    def __init__(self: "TransformerDecoder", num_layers: int, d_model: int, **layer_kwargs: object) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, **layer_kwargs) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self: "TransformerDecoder", target: Tensor, memory: Tensor, pos: Tensor, query_pos: Tensor) -> Tensor:
        """Run every decoder layer in turn, then normalize."""
        for layer in self.layers:
            target = layer(target, memory, pos, query_pos)
        return self.layernorm(target)


class ResNetBackbone(nn.Module):
    """A torchvision ResNet truncated after a chosen number of residual stages.

    ``stages`` controls both depth and output stride: 4 stages gives stride 32 as in stock DETR,
    3 gives stride 16, and so on, which trades detector capacity for a larger, more directly
    image-tied feature map. ``small_stem`` replaces the 7x7 stride 2 convolution and max pool with a
    3x3 stride 1 convolution, removing the factor 4 downsample that dominates at low resolution;
    the replacement stem is always randomly initialized, since no pretrained weights exist for it.
    """

    def __init__(self: "ResNetBackbone", name: str = "resnet18", pretrained: bool = False,
                 stages: int = 4, small_stem: bool = False) -> None:
        super().__init__()
        if name not in RESNET_STAGE_CHANNELS:
            raise ValueError(f"Unsupported backbone {name!r}, expected one of {list(RESNET_STAGE_CHANNELS)}.")
        if not 1 <= stages <= 4:
            raise ValueError(f"stages must be between 1 and 4, got {stages}.")
        resnet = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        if small_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()
        else:
            self.conv1, self.bn1, self.maxpool = resnet.conv1, resnet.bn1, resnet.maxpool
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList([getattr(resnet, f"layer{i}") for i in range(1, stages + 1)])
        self.num_channels = RESNET_STAGE_CHANNELS[name][stages - 1]

    def forward(self: "ResNetBackbone", x: Tensor) -> Tensor:
        """Extract the final feature map."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    """The original DETR box head: a small multi layer perceptron with ReLU between layers."""

    def __init__(self: "MLP", input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:] + [output_dim])])

    def forward(self: "MLP", x: Tensor) -> Tensor:
        """Apply every layer, with ReLU on all but the last."""
        for i, layer in enumerate(self.layers):
            x = f.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DetrSmallModel(nn.Module):
    """Backbone, projection to the transformer width, and the encoder/decoder, as in DetrModel."""

    def __init__(self: "DetrSmallModel", backbone: ResNetBackbone, d_model: int, nheads: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool, num_queries: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_projection = nn.Conv2d(backbone.num_channels, d_model, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(d_model // 2)
        layer_kwargs = {"d_model": d_model, "nheads": nheads, "dim_feedforward": dim_feedforward,
                        "dropout": dropout, "pre_norm": pre_norm}
        self.encoder = TransformerEncoder(num_encoder_layers, **layer_kwargs) if num_encoder_layers else None
        if num_decoder_layers:
            self.decoder = TransformerDecoder(num_decoder_layers, **layer_kwargs)
            self.query_position_embeddings = nn.Embedding(num_queries, d_model)
        else:
            self.decoder = None
            self.query_position_embeddings = None

    def forward(self: "DetrSmallModel", pixel_values: Tensor) -> Tensor:
        """Return the hidden states the prediction heads read, of shape (batch, queries, d_model)."""
        features = self.backbone(pixel_values)
        pos = self.position_embedding(features).flatten(2).permute(0, 2, 1)
        src = self.input_projection(features).flatten(2).permute(0, 2, 1)
        memory = self.encoder(src, pos) if self.encoder is not None else src
        if self.decoder is None:
            # No decoder: each spatial cell of the feature map is its own prediction slot.
            return memory
        query_pos = self.query_position_embeddings.weight[None].expand(pixel_values.shape[0], -1, -1)
        return self.decoder(torch.zeros_like(query_pos), memory, pos, query_pos)


class DetrSmall(nn.Module):
    """A configurable DETR detector returning logits and normalized cxcywh boxes.

    The ``config`` attribute is a real DetrConfig, because the transformers set prediction loss reads
    its matcher costs, loss coefficients and label count from one. Everything about the architecture
    itself comes from this class's own arguments, not from the config.
    """

    def __init__(self: "DetrSmall", num_labels: int = 91, num_queries: int = 100, d_model: int = 256,
                 nheads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.0, pre_norm: bool = False,
                 backbone: str = "resnet18", backbone_stages: int = 4,
                 pretrained_backbone: bool = False, small_stem: bool = False) -> None:
        super().__init__()
        self.model = DetrSmallModel(
            backbone=ResNetBackbone(backbone, pretrained_backbone, backbone_stages, small_stem),
            d_model=d_model, nheads=nheads, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, pre_norm=pre_norm, num_queries=num_queries)
        self.class_labels_classifier = nn.Linear(d_model, num_labels + 1)
        self.bbox_predictor = MLP(d_model, d_model, 4, 3)
        self.config = DetrConfig(num_labels=num_labels, d_model=d_model, num_queries=num_queries,
                                 encoder_layers=max(num_encoder_layers, 1),
                                 decoder_layers=max(num_decoder_layers, 1), auxiliary_loss=False)
        self._reset_transformer_parameters()

    def _reset_transformer_parameters(self: "DetrSmall") -> None:
        """Xavier initialize the transformer weights, as the original DETR does.

        The backbone keeps torchvision's own initialization (or its pretrained weights); the original
        applies this only to the transformer, and ResNet's Kaiming init suits its convolutions better.
        """
        for module in (self.model.encoder, self.model.decoder):
            if module is None:
                continue
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self: "DetrSmall", pixel_values: Tensor) -> DetrSmallOutput:
        """Run the detector."""
        hidden = self.model(pixel_values)
        return DetrSmallOutput(logits=self.class_labels_classifier(hidden),
                               pred_boxes=self.bbox_predictor(hidden).sigmoid())


# Named points on the ladder from stock-depth DETR down to a shallow dense detector. Each entry
# changes as little as possible from the one above it, so a drop in attack success localizes to the
# component that changed.
VARIANTS = {
    # Stock DETR depth and width, but a ResNet-18 backbone with trainable BatchNorm.
    "r18": {},
    # Same, on a stride 16 backbone: fewer stages, larger and less abstracted feature map.
    "r18-s3": {"backbone_stages": 3},
    # Thin the transformer to one layer each side and halve the width.
    "r18-shallow": {"num_encoder_layers": 1, "num_decoder_layers": 1, "d_model": 128,
                    "dim_feedforward": 512, "nheads": 4},
    # Same as above with the shallower backbone as well: the closest analogue of the YOLO surgery.
    "r18-shallow-s3": {"num_encoder_layers": 1, "num_decoder_layers": 1, "d_model": 128,
                       "dim_feedforward": 512, "nheads": 4, "backbone_stages": 3},
    # Drop the encoder: object queries cross attend directly to projected backbone features.
    "r18-nodec": {"num_encoder_layers": 1, "num_decoder_layers": 0, "d_model": 128,
                  "dim_feedforward": 512, "nheads": 4, "backbone_stages": 3},
    # No transformer at all: heads read backbone cells, i.e. a dense detector with a DETR loss.
    "r18-notrans": {"num_encoder_layers": 0, "num_decoder_layers": 0, "d_model": 128,
                    "backbone_stages": 3},
    # Full resolution stem, for the low resolution experiments.
    "r18-shallow-stem": {"num_encoder_layers": 1, "num_decoder_layers": 1, "d_model": 128,
                         "dim_feedforward": 512, "nheads": 4, "backbone_stages": 3, "small_stem": True},
}


def detr_small(variant: str = "r18", **overrides: object) -> DetrSmall:
    """Build a named DETR variant, with any constructor argument overridable."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {list(VARIANTS)}.")
    return DetrSmall(**{**VARIANTS[variant], **overrides})
