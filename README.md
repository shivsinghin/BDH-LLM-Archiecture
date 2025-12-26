## Baby Dragon Hatchling (Advanced Implementation)

**A Production-Grade, Multi-GPU Implementation of the BDH Architecture.**
**Baby Dragon Hatchling (BDH)** is a revolutionary biologically-inspired language model architecture that bridges deep learning with neuroscience. Unlike traditional Transformers, BDH models attention as an **emergent property of neuron-level interactions**, providing unprecedented interpretability while matching GPT-2 scale performance.

> 📄 Based on the paper: *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*  
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

## About this Implementation

This repository provides an **alternative BDH implementation** focusing on the recurrent state-space formulation (Definition 4 from the paper) with extensive biological extensions. It differs from the [official implementation](https://github.com/pathwaycom/bdh) which uses a multi-layer transformer-like architecture with RoPE attention.

| Feature | Official Repo | This Implementation |
|---------|:-------------:|:-------------------:|
| Architecture Style | Multi-layer + RoPE Attention | Recurrent State-Space (Def. 4) |
| Multi-GPU DDP Training | ❌ | ✅ |
| Gradient Synchronization | ❌ | ✅ |
| Batched Sequence Processing | ✅ | ✅ |
| Mixed Precision (fp16/bf16) | ✅ | ❌ |
| torch.compile() | ✅ | ❌ |
| Multi-Timescale STDP Decay | ❌ | ✅ |
| Local Synaptic Forgetting | ❌ | ✅ |
| Homeostatic Regulation | ❌ | ✅ |
| k-WTA Lateral Inhibition | ❌ | ✅ |
| Dendritic Subunits | ❌ | ✅ |
| Neuromodulation via Surprisal | ❌ | ✅ |
| Stochastic Spikes | ❌ | ✅ |
| Training Visualization | ❌ | ✅ |
| Text Generation Pipeline | ✅ | ✅ |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BDH-Neuro Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BDHGPURefTorch (Base Model - Definition 4 from paper)                     │
│   ├── Scale-free network topology                                           │
│   ├── Hebbian working memory (ρ matrix)                                     │
│   ├── GPU-friendly state-space formulation                                  │
│   └── Interpretable sparse activations                                      │
│         │                                                                   │
│         ▼ (inherits and extends)                                            │
│                                                                             │
│   BDHNeuroRefTorch (Enhanced with biological features)                      │
│   ├── ✨ Multi-timescale synaptic decay (STDP-like learning)                │
│   ├── 🧹 Local forgetting for inactive synapses                             │
│   ├── ⚖️  Homeostatic regulation of output activity                         │
│   ├── 🔥 k-WTA lateral inhibition (adaptive gain-modulated)                 │
│   ├── 🌳 Dendritic subunits with branch nonlinearities                      │
│   ├── 🧠 Neuromodulation via surprisal gain                                 │
│   ├── ⚡ Stochastic spikes for biological realism                           │
│   └── 📉 x-decay with L1 normalization for realistic sparsity               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Biological Extensions Explained

| Extension | Biological Inspiration | Implementation |
|-----------|----------------------|----------------|
| **Multi-timescale STDP** | Different synaptic plasticity timescales in the brain | `U_kernels=[0.99, 0.97, 0.94]` with weighted mixing |
| **Local Forgetting** | Synaptic pruning of unused connections | `local_forget_eta=0.02` decay on inactive pre-synapses |
| **Homeostasis** | Neural circuits maintain stable activity levels | `homeostasis_tau` scales output to target activity |
| **k-WTA Inhibition** | Winner-take-all competition in cortical circuits | Top-k activation with gain-modulated k |
| **Dendritic Subunits** | Nonlinear integration in dendritic branches | `branches=2` with softplus nonlinearity |
| **Neuromodulation** | Dopamine/norepinephrine modulate learning | Surprisal-based gain scaling of ρ updates |
| **Stochastic Spikes** | Probabilistic neural firing | `spike_rate=0.01` random activation injection |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bdh-advanced.git
cd bdh-advanced

# Install dependencies
pip install -r requirements.txt
```

### Training

#### Option 1: Multi-GPU Training (Recommended for serious work)

```bash
# Use all 8 A100 GPUs with DDP
torchrun --nproc_per_node=8 train.py \
    --file input.txt \
    --epochs 5 \
    --n 128 \
    --d 32 \
    --lr 1e-3 \
    --batch_size 8 \
    --seq_len 256
```

**What happens under the hood:**
- Each GPU processes different batches (DistributedSampler)
- Gradients are automatically synchronized across all GPUs
- Effective batch size = `batch_size × num_gpus` (e.g., 8 × 8 = 64)
- ~8x speedup with linear scaling

#### Option 2: Single GPU Training

```bash
python train.py --file input.txt --epochs 5 --device cuda
```

### Inference (Text Generation)

```bash
python inference.py \
    --file input.txt \
    --seed_text "Once upon a time" \
    --length 500 \
    --temperature 0.8 \
    --device cuda
```

### Visualize Training Dynamics

```bash
# First, save training output to a log file
python train.py --file input.txt --epochs 5 --device cuda > train.log

# Then visualize the metrics
python plot_sparsity.py --file train.log
```

This generates plots for:
- 📉 **Loss** — Training convergence
- 📊 **Sparsity x** — Input activation sparsity
- 📊 **Sparsity y** — Output activation sparsity  
- 📈 **ρF** — Frobenius norm of working memory
- 🎚️ **Gain** — Neuromodulation gain over time

## Project Structure

```
BDH/
├── model.py          # Core architecture (BDHGPURefTorch + BDHNeuroRefTorch)
├── train.py          # Multi-GPU training pipeline with DDP
├── inference.py      # Text generation with trained models
├── plot_sparsity.py  # Training dynamics visualization
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Configuration Reference

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | `input.txt` | Path to training text file |
| `--epochs` | `3` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--n` | `128` | Neuron dimension (network width) |
| `--d` | `32` | Latent embedding dimension |
| `--batch_size` | `8` | Sequences per GPU |
| `--seq_len` | `256` | Length of each training sequence |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | `input.txt` | Training text file (for vocab reconstruction) |
| `--seed_text` | `"Hello "` | Starting text for generation |
| `--length` | `500` | Number of characters to generate |
| `--temperature` | `0.8` | Sampling temperature (lower = more deterministic) |
| `--n` | `128` | Must match training |
| `--d` | `32` | Must match training |

### Model Hyperparameters (in `train.py`)

```python
BDHNeuroRefTorch(
    n=128,                          # Neuron dimension
    d=32,                           # Embedding dimension
    V=vocab_size,                   # Vocabulary size
    U_kernels=[0.99, 0.97, 0.94],   # Multi-timescale decay rates
    U_weights=[0.5, 0.3, 0.2],      # Decay kernel weights
    local_forget_eta=0.02,          # Synaptic forgetting rate
    homeostasis_tau=0.15 * n,       # Homeostatic target activity
    k_wta=n // 8,                   # k-WTA winners (16 of 128)
    branches=2,                     # Dendritic branch count
    branch_nl="softplus",           # Branch nonlinearity
    mod_gamma_max=0.8,              # Max neuromodulation gain
    spike_rate=0.01,                # Stochastic spike probability
    x_decay=0.97,                   # Input state decay
    ln_before_Dy=True,              # LayerNorm placement
    use_relu_lowrank=True           # ReLU in low-rank projection
)
```

## Scaling Properties

BDH follows **Transformer-like scaling laws** while maintaining full interpretability:

| Model Size | Parameters | Comparable To |
|------------|------------|---------------|
| Small | ~10M | GPT-2 Small |
| Medium | ~100M | GPT-2 Medium |
| Large | ~1B | GPT-2 Large |

The architecture's **sparse, positive activations** enable:
- Direct interpretation of neuron contributions
- Monosemantic features (each neuron = one concept)
- Biological plausibility for neuroscience research

## Research Applications

This implementation is suitable for:

- **Neuroscience Research** — Study emergent attention from local neuron dynamics
- **Interpretability Research** — Analyze sparse, monosemantic representations
- **Scaling Studies** — Investigate BDH scaling laws on large corpora
- **Biological Modeling** — Test hypotheses about synaptic plasticity and homeostasis
- **Production Deployment** — Multi-GPU training enables practical model sizes

## Learn More

📄 *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.* [**The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain**](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

## Acknowledgements

This implementation builds upon the foundational BDH architecture from [Pathway](https://pathway.com). Official repository: **[github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)**

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{kosowski2025dragon,
  title={The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain},
  author={Kosowski, Adrian and Uznański, Przemysław and Chorowski, Jan and Stamirowska, Zuzanna and Bartoszkiewicz, Michał},
  journal={arXiv preprint arXiv:2509.26507},
  year={2025}
}
```
