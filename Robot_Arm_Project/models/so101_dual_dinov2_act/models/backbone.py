"""DINOv2 기반 시각적 특징 추출 백본 모듈 (양안 카메라 ACT용)."""

from typing import List, Tuple
import torch
from torch import nn
from .position_encoding import build_position_encoding


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT 패치 토큰을 2D Feature Map으로 변환하는 백본."""

    def __init__(self, name: str = "dinov2_vits14", train_backbone: bool = False, unfreeze_blocks: int = 0):
        super().__init__()
        self.name = name
        self.train_backbone = train_backbone
        self.unfreeze_blocks = int(max(0, unfreeze_blocks))
        self.is_trainable = bool(self.train_backbone or self.unfreeze_blocks > 0)
        self.patch_size = 14

        # PyTorch Hub를 통해 공식 사전학습 DINOv2 로드
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", name)

        # 모델별 채널 수 파악 (vits14: 384, vitb14: 768, vitl14: 1024)
        self.num_channels = self.dinov2.embed_dim

        # 전체 고정 후 전체 해제(train_backbone) 또는 마지막 N개 블록만 부분 해제(unfreeze_blocks)
        for param in self.dinov2.parameters():
            param.requires_grad = False

        if self.train_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = True
        elif self.unfreeze_blocks > 0:
            n_blocks = len(self.dinov2.blocks)
            k = min(self.unfreeze_blocks, n_blocks)
            for block in self.dinov2.blocks[-k:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.dinov2, "norm"):
                for param in self.dinov2.norm.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.is_trainable:
            self.dinov2.eval()
        elif not self.train_backbone and self.unfreeze_blocks > 0:
            self.dinov2.eval()
            k = min(self.unfreeze_blocks, len(self.dinov2.blocks))
            for block in self.dinov2.blocks[-k:]:
                block.train(mode)
            if hasattr(self.dinov2, "norm"):
                self.dinov2.norm.train(mode)
        return self

    def forward(self, tensor: torch.Tensor):
        """
        tensor: (B, 3, H, W) -> H, W는 14의 배수여야 함
        return: dict {"0": (B, num_channels, H // 14, W // 14)}
        """
        B, C, H, W = tensor.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"입력 이미지 크기 ({H}x{W})는 DINOv2 패치 크기 ({self.patch_size})의 배수여야 합니다."
        )
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        if not self.is_trainable:
            with torch.no_grad():
                features = self.dinov2.forward_features(tensor)
                patch_tokens = features["x_norm_patchtokens"]
        else:
            features = self.dinov2.forward_features(tensor)
            patch_tokens = features["x_norm_patchtokens"]

        # (B, patch_h * patch_w, embed_dim) -> (B, embed_dim, patch_h, patch_w)
        feature_map = patch_tokens.reshape(B, patch_h, patch_w, self.num_channels).permute(0, 3, 1, 2)
        return {"0": feature_map}


class Joiner(nn.Sequential):
    """백본 특징 맵과 2D 위치 임베딩을 결합하는 모듈."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        xs = self[0](tensor)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_backbone(
    backbone_name: str = "dinov2_vits14",
    hidden_dim: int = 512,
    train_backbone: bool = False,
    unfreeze_blocks: int = 0,
) -> Joiner:
    """DINOv2 백본 및 2D 위치 임베딩 생성기"""
    position_embedding = build_position_encoding(hidden_dim)
    backbone = DINOv2Backbone(name=backbone_name, train_backbone=train_backbone, unfreeze_blocks=unfreeze_blocks)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
