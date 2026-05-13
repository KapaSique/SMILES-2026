# Zero-Order Fine-Tuning of ResNet18 on CIFAR100

**49.68% top-1 accuracy** on CIFAR100 with **zero gradient computations** and only **8192 training samples**.

| Checkpoint | Top-1 |
|---|---|
| 1. Baseline (ImageNet head) | 0.37% |
| 2. Initialized head (no FT) | 0.89% |
| 3. Fine-tuned (ZO) | **49.68%** |

## Approach

**SPSA** (Simultaneous Perturbation Stochastic Approximation) + **backbone feature caching** + **many inner Adam steps per batch** + **cosine LR with warmup**.

- SPSA with Rademacher perturbations — 2 forward passes per gradient query regardless of parameter count
- `Q = 256` independent SPSA queries averaged per gradient estimate (variance reduction 16×)
- `inner_steps = 64` Adam updates per outer batch, reusing cached backbone features
- Total: 16384 Adam updates on only 256 honest backbone passes
- Cosine LR schedule with linear warmup (8 steps) down to 5% of base LR
- Adam with decoupled weight decay (`β₁=0.9`, `β₂=0.999`, `lr=2e-3`, `wd=5e-4`)
- Global pseudo-gradient L2 clipping at 1.0
- Small-scale Gaussian head init (`std=0.01`, bias=0)
- Conservative augmentations: `RandomCrop(224, padding=16, reflect)` + `HorizontalFlip`

## Quick Start

```bash
pip install -r requirements.txt
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --output results.json \
    --seed 42
```

CIFAR100 downloads automatically on first run. Wall-clock ~75 min on Apple Silicon MPS. Reproducible to ±0.5% (fixed seed, deterministic algorithms enabled).

## Repository Structure

| File | Role | Editable |
|------|------|----------|
| `zo_optimizer.py` | SPSA estimator, Adam optimizer, backbone feature caching, cosine LR | Yes |
| `head_init.py` | Small-scale Gaussian head initialization | Yes |
| `augmentation.py` | Training augmentations (Crop + Flip + Normalize) | Yes |
| `train_data.py` | CIFAR100 dataset loader | Yes |
| `model.py` | ResNet18 model builder | No |
| `validate.py` | Evaluation harness | No |

## Budget

`n_batches × batch_size ≤ 8192`. This solution uses `256 × 32 = 8192` (exact limit).

## Result

```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.0089,
  "val_accuracy_top1_finetuned": 0.4968,
  "n_batches": 256,
  "batch_size": 32,
  "layers_tuned": ["fc.weight", "fc.bias"]
}
```

See [`SOLUTION.md`](SOLUTION.md) for full methodology, design rationale, and list of failed attempts.
