import os
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.alpha_hit_mlp import AlphaHitMLP, AlphaHitConfig, alpha_hit_loss


@dataclass
class AlphaTrainerConfig:
    horizons_sec: List[int]
    n_features: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 2e-4
    batch_size: int = 256
    steps_per_tick: int = 2  # per decision loop
    max_buffer: int = 200_000
    # exponential decay for old samples: weight = exp(-age_sec / half_life)
    data_half_life_sec: float = 3600.0  # 1h
    # save/load
    ckpt_path: str = "state/alpha_hit_mlp.pt"
    enable: bool = True


class OnlineAlphaTrainer:
    """
    CPU: add_sample() + buffer + compute decay weights
    GPU: train_step() + predict()
    """
    def __init__(self, cfg: AlphaTrainerConfig):
        self.cfg = cfg
        self.horizons = cfg.horizons_sec
        self.H = len(self.horizons)
        self.device = torch.device(cfg.device)

        mcfg = AlphaHitConfig(horizons_sec=cfg.horizons_sec, n_features=cfg.n_features)
        self.model = AlphaHitMLP(mcfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)

        self.buffer = []  # list of dicts
        self.last_train_ms = 0

        self._try_load()

    def _try_load(self):
        p = self.cfg.ckpt_path
        if os.path.exists(p):
            try:
                obj = torch.load(p, map_location="cpu")
                self.model.load_state_dict(obj["model"])
                if "opt" in obj:
                    self.opt.load_state_dict(obj["opt"])
                print(f"[ALPHA_HIT] loaded ckpt {p}")
            except Exception as e:
                print(f"[ALPHA_HIT] failed to load ckpt {p}: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.cfg.ckpt_path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict()}, self.cfg.ckpt_path)

    def add_sample(
        self,
        x: np.ndarray,              # [F]
        y: Dict[str, np.ndarray],   # each [H] in {0,1}
        ts_ms: int,
        symbol: str,
    ):
        if not self.cfg.enable:
            return
        self.buffer.append({"x": x.astype(np.float32), "y": y, "ts_ms": ts_ms, "sym": symbol})
        if len(self.buffer) > self.cfg.max_buffer:
            # drop oldest
            self.buffer = self.buffer[-self.cfg.max_buffer :]

    def _sample_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        B = self.cfg.batch_size
        now_ms = int(time.time() * 1000)
        idx = np.random.randint(0, len(self.buffer), size=B)
        items = [self.buffer[i] for i in idx]

        X = np.stack([it["x"] for it in items], axis=0)  # [B,F]
        def Ys(key):
            return np.stack([it["y"][key] for it in items], axis=0).astype(np.float32)  # [B,H]

        age_sec = (now_ms - np.array([it["ts_ms"] for it in items])) / 1000.0
        w = np.exp(-age_sec / max(1.0, self.cfg.data_half_life_sec)).astype(np.float32)  # [B]

        batch = {
            "x": torch.from_numpy(X).to(self.device),
            "y_tp_long": torch.from_numpy(Ys("tp_long")).to(self.device),
            "y_sl_long": torch.from_numpy(Ys("sl_long")).to(self.device),
            "y_tp_short": torch.from_numpy(Ys("tp_short")).to(self.device),
            "y_sl_short": torch.from_numpy(Ys("sl_short")).to(self.device),
            "w": torch.from_numpy(w).to(self.device),
        }
        return batch

    def train_tick(self) -> Dict[str, Any]:
        """
        call frequently (each decision loop). Runs few steps on GPU.
        """
        if not self.cfg.enable:
            return {"enabled": False}

        out = {"enabled": True, "buffer_n": len(self.buffer), "loss": None}
        self.model.train()
        last_loss = None
        for _ in range(self.cfg.steps_per_tick):
            batch = self._sample_batch()
            if batch is None:
                break
            pred = self.model(batch["x"])
            loss = alpha_hit_loss(
                pred,
                batch["y_tp_long"], batch["y_sl_long"],
                batch["y_tp_short"], batch["y_sl_short"],
                label_smoothing=self.model.cfg.label_smoothing,
                sample_weight=batch["w"],
            )
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            last_loss = float(loss.detach().cpu().item())

        out["loss"] = last_loss
        return out

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        x: [F]
        return each [H] float prob
        """
        if not self.cfg.enable:
            return {}
        self.model.eval()
        X = torch.from_numpy(x.astype(np.float32)).to(self.device).unsqueeze(0)  # [1,F]
        pred = self.model.predict(X)
        return {
            "p_tp_long": pred["p_tp_long"][0].detach().cpu().numpy(),
            "p_sl_long": pred["p_sl_long"][0].detach().cpu().numpy(),
            "p_tp_short": pred["p_tp_short"][0].detach().cpu().numpy(),
            "p_sl_short": pred["p_sl_short"][0].detach().cpu().numpy(),
        }

