import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlphaHitConfig:
    horizons_sec: List[int]
    n_features: int
    hidden: int = 128
    depth: int = 3
    dropout: float = 0.05
    # label smoothing helps online stability
    label_smoothing: float = 0.01


class AlphaHitMLP(nn.Module):
    """
    Predicts TP/SL hit probability per horizon, per direction.
      outputs:
        p_tp_long[h], p_sl_long[h], p_tp_short[h], p_sl_short[h]

    We model them as independent Bernoulli heads (TP/SL can both be 0 if timeout/exit-other).
    If your policy guarantees mutually exclusive TP vs SL, you can later couple them.
    """

    def __init__(self, cfg: AlphaHitConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = cfg.horizons_sec
        out_dim = len(self.horizons) * 4  # tpL, slL, tpS, slS for each horizon

        layers = []
        in_dim = cfg.n_features
        for i in range(cfg.depth):
            layers.append(nn.Linear(in_dim, cfg.hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden, out_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, F]
        returns dict of probs: each [B, H]
        """
        z = self.backbone(x)
        logits = self.head(z)  # [B, 4H]
        H = len(self.horizons)
        logits = logits.view(-1, H, 4)  # [B, H, 4]

        # index: 0 tpL, 1 slL, 2 tpS, 3 slS
        p = torch.sigmoid(logits)
        return {
            "p_tp_long": p[:, :, 0],
            "p_sl_long": p[:, :, 1],
            "p_tp_short": p[:, :, 2],
            "p_sl_short": p[:, :, 3],
            "logits": logits,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.forward(x)


def alpha_hit_loss(
    pred: Dict[str, torch.Tensor],
    y_tp_long: torch.Tensor,
    y_sl_long: torch.Tensor,
    y_tp_short: torch.Tensor,
    y_sl_short: torch.Tensor,
    label_smoothing: float = 0.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    y_*: [B, H] float in {0,1}
    sample_weight: [B] or [B,1]
    """
    def bce(p, y):
        if label_smoothing > 0:
            y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy(p, y, reduction="none")

    loss = (
        bce(pred["p_tp_long"], y_tp_long) +
        bce(pred["p_sl_long"], y_sl_long) +
        bce(pred["p_tp_short"], y_tp_short) +
        bce(pred["p_sl_short"], y_sl_short)
    )  # [B, H]
    loss = loss.mean(dim=1)  # [B]
    if sample_weight is not None:
        loss = loss * sample_weight.view(-1)
    return loss.mean()

