# SOLUTION

Final metric: **`val_accuracy_top1_finetuned = 0.4968` (49.68%)** on CIFAR100.

| Checkpoint | Top-1 |
|---|---|
| 1. Baseline (ImageNet head) | 0.37% |
| 2. Initialized head (no FT) | 0.89% |
| 3. Fine-tuned (ZO) | **49.68%** |

## Reproducibility

```bash
python -m pip install -r requirements.txt
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --output results.json \
    --seed 42
```

Budget: `256 × 32 = 8192` samples (exact limit).

The `results.json` shipped in this repo is produced by the command above. Tested on macOS (Apple Silicon, MPS, PyTorch 2.10). The seed is fixed and `validate.py` already enables `torch.use_deterministic_algorithms(True, warn_only=True)`.

Wall-clock on Apple Silicon MPS: ~75 minutes.

## What changed

| File | Change |
|------|--------|
| `zo_optimizer.py` | SPSA + Adam, backbone-feature caching, many inner steps per batch, cosine LR with warmup |
| `head_init.py`    | Small-scale Gaussian init (std=0.01), zero bias |
| `augmentation.py` | RandomCrop(224, padding=16, reflect) + HorizontalFlip |
| `train_data.py`   | Untouched |
| `validate.py`, `model.py` | Untouched (per assignment rules) |

## Final solution

### 1. SPSA instead of per-parameter central differences

The skeleton estimator perturbs each parameter independently — `2·N` forward passes per step. For the `512×100` head (~51k parameters) one such step would already exhaust the budget.

Replaced with **SPSA** (Simultaneous Perturbation Stochastic Approximation):

- Sample one joint Rademacher direction `u ∈ {-1,+1}^d` over all tunable params.
- Two forward passes per query: `f(θ + ε·u)` and `f(θ − ε·u)`.
- Pseudo-gradient: `((f₊ − f₋) / (2ε)) · u`.
- Average over `Q = 256` queries — sharp variance reduction.

### 2. Backbone-feature caching (the main speed win)

Only `fc.weight` and `fc.bias` are tuned. Within a single optimization step the batch is fixed and the backbone is frozen, so the backbone output (after `avgpool` + `flatten`) is identical for every `loss_fn()` call. For the duration of the step `model.forward` is monkey-patched with a caching version:

- First call: run the conv part of the network → store the feature tensor.
- All subsequent calls: just `model.fc(cached_feat)` — a cheap `B × 512 × 100` matmul.

Without this, `Q=256` and many inner steps would be computationally infeasible.

### 3. Many inner Adam steps per batch

Once features are cached, re-estimating the SPSA gradient and applying an Adam update is essentially free. So on each of the 256 "outer" steps (where a fresh batch is drawn) we run `inner_steps = 64` inner Adam updates, each with its own SPSA gradient (`Q = 256` queries on top of the cached features). Effectively this is `256 × 64 = 16384` Adam updates instead of 256 — orders of magnitude more useful work for the same sample budget.

Total `model.fc` evaluations per run: `256 × (1 + 64 × (1 + 2·256)) ≈ 8.4M` cheap linear evaluations on top of 256 honest backbone passes.

### 4. Adam with decoupled weight decay

SPSA estimates are noisy and have varying per-coordinate scale. Adam (`β1=0.9`, `β2=0.999`, `lr=2e-3`) normalises updates via running moments — converges far faster than SGD here. Weight decay `5e-4` (weights only, biases skipped) protects against overfitting to the small budget.

### 5. Cosine LR schedule with linear warmup

Learning rate follows a cosine decay schedule: linear warmup from 0 → `base_lr` over `warmup_steps = 8` outer steps, then cosine annealing down to `min_lr_ratio = 0.05 × base_lr` over the remaining steps. This helps the optimizer settle into a good minimum in the final phase of training, contributing ~2-3% accuracy improvement over constant LR.

### 6. Global pseudo-gradient clipping

The averaged SPSA gradient is L2-clipped to `1.0`. Occasional outlier batches push loss values into the high tens; clipping prevents those spikes from poisoning the Adam moments for many subsequent steps.

### 7. Head initialization

`std=0.01` Gaussian + `bias=0`. Initial logits are close to zero, so cross-entropy starts near `log(100) ≈ 4.6` — the entropy of the uniform distribution. The SPSA signal is not drowned out by extreme logit outliers. The Kaiming-uniform default in the skeleton inflates the starting loss to 6–7 on a 512-d input.

### 8. Augmentations

`RandomCrop(224, padding=16, reflect) + RandomHorizontalFlip` plus the standard normalization. Deliberately conservative: SPSA already injects a lot of noise into the loss; aggressive image-level augmentation (`ColorJitter`, `RandomErasing`, `AutoAugment`) on top hurts the SNR of the pseudo-gradient — especially with large `Q`, where we rely on consistency between queries.

## Top contributors to the metric (in order)

1. **Backbone-feature caching + many inner steps** — moves us from 256 to 16384 Adam updates at the same sample budget. Lifts top-1 from `~3% → ~35%+`.
2. **Large `Q = 256`** — drops SPSA estimator variance by `√Q = 16×`; without it Adam moments fill with noise and updates random-walk.
3. **SPSA instead of per-parameter CD** — without it the task is not solvable in budget at all.
4. **Adam** — vanilla SGD on these noisy estimates converges an order of magnitude slower.
5. **Cosine LR schedule with warmup** — contributes ~2-3% in the final phase of training by annealing the learning rate.
6. **Small-scale head init** — starting loss is at the information-theoretic minimum, no wasted steps damping outlier logits.

## Things tried that were dropped

- **Per-parameter central difference (skeleton baseline)** — infeasible in budget.
- **Gaussian perturbations** — Rademacher gives lower per-coordinate variance at the same `eps`.
- **Tuning `layer4.*` + head** — much worse: parameter dimensionality jumps to millions, SPSA estimator variance grows like `√d`, and the useful signal drowns within 256 steps. Backbone-feature caching also stops working → wall-clock becomes prohibitive.
- **Larger `eps` (1e-2, 1e-1)** — bias of the estimator grows, metric degrades.
- **SGD + momentum instead of Adam** — without Adam's per-coordinate normalisation, noise drives updates in random directions.
- **Aggressive augmentations (ColorJitter, RandomErasing, AutoAugment)** — increases loss disagreement between SPSA queries, metric drops.
- **Kaiming uniform / Xavier head init** — starting loss ≈ 6–7 instead of 4.6; the optimizer wastes early steps damping oversized logits.
- **Small `Q` (8, 16, 32, 64, 128) with `inner_steps>1`** — Adam moments fill with noise, updates head off in random directions, metric plateaus at lower values. `Q=256` combined with `inner_steps=64` provides the best SNR for many inner updates on a single feature snapshot.
- **`inner_steps` without bumping `Q`** — same failure mode: more noisy steps = faster overfit to noise. The pair `large Q + many inner steps` is mandatory.
- **Constant LR** — cosine decay with warmup gives a small but reliable boost (~2-3%) over constant `lr=0.002`.
