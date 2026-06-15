# SBI (Simulation-Based Inference) for Cosmic Web Eigenvalues

## Overview

We train a conditional normalizing flow to learn the posterior distribution of cosmic web eigenvalues (λ₁, λ₂, λ₃) given galaxy observables.

---

## SBI Paradigm (Cranmer et al. 2019)

We use **Neural Posterior Estimation (NPE)**, one of three main SBI approaches:

| Method | What It Learns | Our Use |
|--------|----------------|---------|
| **NPE** (Neural Posterior Estimation) | p(θ \| x) directly | ✅ This work |
| NLE (Neural Likelihood Estimation) | p(x \| θ) | Not used |
| NRE (Neural Ratio Estimation) | p(θ \| x) / p(θ) | Not used |

### NPE in Our Context:
- **θ** = eigenvalues (λ₁, λ₂, λ₃) - what we want to infer
- **x** = galaxy observables (mass, velocity, position, neighbors)
- **Simulator** = IllustrisTNG simulation (provides paired x, θ data)
- **Amortization** = Train once, get posterior for ANY x instantly

The key insight from Cranmer et al.: Instead of running MCMC for each new observation, NPE learns to map observations → posteriors directly.

---

## Fixed vs Sequential Simulations

### Our Approach: Fixed Pre-Computed Simulations

```
Standard Sequential NPE:        Our Fixed NPE:
────────────────────────        ──────────────────────
Round 1: θ ~ p(θ)              θ = eigenvalues (fixed from TNG)
         x ~ simulator(θ)       x = galaxy features (fixed from TNG)
         Train q₁(θ|x)              ↓
Round 2: θ ~ q₁(θ|x₀)          Train q(θ|x) once on full dataset
         x ~ simulator(θ)       
         Train q₂(θ|x)          No iteration, no new simulations
         ...
```

### Is Fixed-Simulation SBI Typical?

**Yes, very common in cosmology/astrophysics:**

| Context | Simulation Cost | Approach |
|---------|-----------------|----------|
| Particle physics (LHC) | Fast | Sequential SBI possible |
| Cosmological N-body | Expensive (~hours/sim) | Fixed suite (e.g., Quijote) |
| Hydrodynamical (TNG, EAGLE) | Very expensive (~months) | **Fixed, no iteration** |
| Galaxy formation | Extremely expensive | Fixed snapshots only |

### Examples in Literature:

1. **Cosmological parameter inference** - Alsing+ 2019: [arXiv:1903.00007](https://arxiv.org/abs/1903.00007)
   - DELFI with active learning, fixed simulation suites
2. **Galaxy morphology** - Huertas-Company+ 2019: [arXiv:1901.07047](https://arxiv.org/abs/1901.07047)
   - CNNs on TNG/EAGLE images for morphological classification
3. **Weak lensing** - Jeffrey+ 2024: [arXiv:2403.02314](https://arxiv.org/abs/2403.02314)
   - SBI with DES Y3 weak-lensing maps, neural compression
4. **SBI Review** - Cranmer+ 2019: [arXiv:1911.01429](https://arxiv.org/abs/1911.01429)
   - The foundational SBI taxonomy paper
5. **Our work**: Fixed TNG eigenvalue-galaxy pairs

### Could We Use Adaptive/Sequential Approaches?

**In principle, yes!** Possible adaptations with fixed underlying simulation:

| Adaptation | How It Would Work | Benefit |
|------------|-------------------|---------|
| **Adaptive masking** | Focus on uncertain cosmic web regions | Better calibration in transition zones |
| **Luminosity function weighting** | Reweight samples to match observations | Transfer to real data |
| **Active subsampling** | Prioritize high-error nodes | Faster convergence |
| **Proposal refinement** | Importance sampling on existing data | Focus on tail regions |

**Key constraint**: We can't generate NEW (θ, x) pairs, but we can resample/reweight existing data.

### Why We Use Fixed Approach:

1. **TNG is expensive** - ~100M CPU hours for one box
2. **Good coverage** - TNG spans diverse environments (voids → clusters)
3. **Large dataset** - 100k+ nodes provides sufficient training signal
4. **Amortization goal** - We want instant inference for any new observation

## GNN Embeddings - Information Flow

```
Node i's embedding h_i = f(node_i features, neighbor features, graph structure)
```

**Each node gets a UNIQUE embedding** that incorporates:
1. Its own features (mass, velocity, etc.)
2. Neighbors' features (via message passing)
3. Local graph structure (connectivity pattern)

### Important Clarification:

| Concept | What Happens |
|---------|--------------|
| Full graph | GNN sees all nodes during forward pass |
| But embeddings are local | h_i depends on i's k-hop neighborhood |
| Message passing | Information flows: neighbors → target |
| Result | h_i ≠ h_j even if they share neighbors |

The posterior for node i is:
```
p(λ | h_i) = p(λ | local_cosmic_web_context_of_galaxy_i)
```

**This is the power of GNN + SBI**: Eigenvalue predictions depend on cosmic web context (neighbors), not just intrinsic galaxy properties.

---

## What the Model Does

**Each node gets its own posterior** - the model learns p(λ₁, λ₂, λ₃ | features) where features are node-specific.

### Posteriors Are Per-Node

| Concept | Meaning |
|---------|---------|
| **One posterior per node** | Each galaxy has its own p(θ \| x) |
| **Sampling N times** | Draw N samples from that node's posterior |
| **Test mask** | Subset of nodes held out for evaluation |

---

## Training

- **Loss**: Negative log probability: -log p(θ_true | x)
- **Goal**: Maximize probability of true eigenvalues given features
- **Models**: Distrax (5000 epochs) vs Flowjax (4000→7000 epochs)

| Model | Test LogP | Notes |
|-------|-----------|-------|
| Distrax | 4.18 | Better score, no sampling support |
| Flowjax | 3.85 | Sampling works, spline-based |

---

## Evaluation Plots

### 1. Training Curves
- NLL should decrease
- LogP should increase
- Watch for overfitting (gap between train/val)

### 2. SBC Calibration
- Histogram of **ranks** (fraction of samples < true value)
- **Uniform** = well-calibrated
- **Left-tilted** = posteriors biased HIGH (overpredicting)
- **Right-tilted** = posteriors biased LOW
- **U-shaped** = overconfident
- **Inverted-U** = underconfident

### 3. Predictions Scatter
- Posterior mean vs true value
- Points on diagonal = good
- R² close to 1 = accurate

### 4. Corner Plots
- Shows joint posterior p(λ₁, λ₂, λ₃ | x)
- Individual nodes: noisy (few samples from tight posterior)
- Aggregate: smoother but different interpretation

---

## Interpretation of λ₂/λ₃ Calibration

From flowjax evaluation:
- λ₁: KS=0.017 (most uniform) ✅
- λ₂: KS=0.08, tilted left ⚠️
- λ₃: KS=0.062, tilted left ⚠️

**Left tilt means**: True values often < posterior mean → Model overpredicts λ₂, λ₃

---

## File Locations

| File | Purpose |
|------|---------|
| `jraph_sbi_pipeline.py` | Distrax training |
| `jraph_sbi_flowjax.py` | Flowjax training |
| `plot_sbi_evaluation.py` | Evaluation and plotting |
| `sbi_plots_*/` | Output plots |
