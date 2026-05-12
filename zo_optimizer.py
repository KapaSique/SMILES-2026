from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.002,
        eps: float = 1e-3,
        perturbation_mode: str = "rademacher",
        n_perturbations: int = 256,
        inner_steps: int = 64,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        weight_decay: float = 5e-4,
        grad_clip: float = 1.0,
        total_steps: int | None = 256,
        warmup_steps: int = 8,
        min_lr_ratio: float = 0.05,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "rademacher"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'rademacher', got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode
        self.n_perturbations = int(n_perturbations)
        self.inner_steps = int(inner_steps)

        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.grad_clip = float(grad_clip)
        self.base_lr = float(lr)
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self._outer_t = 0
        self._t = 0
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Layer names not found in model: {missing}.")
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(param)
        return torch.where(
            torch.rand_like(param) < 0.5,
            torch.full_like(param, -1.0),
            torch.full_like(param, 1.0),
        )

    def _spsa_query(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        directions = {n: self._sample_direction(p) for n, p in params.items()}

        with torch.no_grad():
            for n, p in params.items():
                p.data.add_(directions[n], alpha=self.eps)
            f_plus = loss_fn()

            for n, p in params.items():
                p.data.add_(directions[n], alpha=-2.0 * self.eps)
            f_minus = loss_fn()

            for n, p in params.items():
                p.data.add_(directions[n], alpha=self.eps)

        coef = (f_plus - f_minus) / (2.0 * self.eps)
        return {n: directions[n].mul_(coef) for n in directions}

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        agg: dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in params.items()}
        for _ in range(self.n_perturbations):
            grads = self._spsa_query(loss_fn, params)
            for n, g in grads.items():
                agg[n].add_(g)
        inv = 1.0 / float(self.n_perturbations)
        for n in agg:
            agg[n].mul_(inv)

        if self.grad_clip > 0.0:
            total_sq = sum(float(g.pow(2).sum().item()) for g in agg.values())
            total_norm = total_sq ** 0.5
            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-12)
                for n in agg:
                    agg[n].mul_(scale)
        return agg

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        with torch.no_grad():
            for n, p in params.items():
                g = grads[n]

                if n not in self._m:
                    self._m[n] = torch.zeros_like(p)
                    self._v[n] = torch.zeros_like(p)

                self._m[n].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self._v[n].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                m_hat = self._m[n] / bc1
                v_hat = self._v[n] / bc2

                if self.weight_decay > 0.0 and p.dim() > 1:
                    p.data.mul_(1.0 - self.lr * self.weight_decay)

                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.adam_eps), value=-self.lr)

    def _cache_features_only(self) -> bool:
        return all(name.startswith("fc.") for name in self.layer_names)

    def _scheduled_lr(self) -> float:
        import math
        t = self._outer_t
        total = self.total_steps or 1
        warm = max(0, self.warmup_steps)
        if t < warm:
            return self.base_lr * float(t + 1) / float(max(1, warm))
        if total <= warm:
            return self.base_lr
        progress = (t - warm) / float(max(1, total - warm))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()
        model = self.model
        self.lr = self._scheduled_lr()

        if self._cache_features_only() and hasattr(model, "fc"):
            original_forward = model.forward
            cache: dict[str, torch.Tensor] = {}

            def _cached_forward(x: torch.Tensor) -> torch.Tensor:
                if "feat" not in cache:
                    out = model.conv1(x)
                    out = model.bn1(out)
                    out = model.relu(out)
                    out = model.maxpool(out)
                    out = model.layer1(out)
                    out = model.layer2(out)
                    out = model.layer3(out)
                    out = model.layer4(out)
                    out = model.avgpool(out)
                    cache["feat"] = torch.flatten(out, 1)
                return model.fc(cache["feat"])

            model.forward = _cached_forward
            try:
                with torch.no_grad():
                    loss_before = loss_fn()
                for _ in range(self.inner_steps):
                    grads = self._estimate_grad(loss_fn, params)
                    self._update_params(params, grads)
            finally:
                model.forward = original_forward
        else:
            with torch.no_grad():
                loss_before = loss_fn()
            grads = self._estimate_grad(loss_fn, params)
            self._update_params(params, grads)

        self._outer_t += 1
        return float(loss_before)
