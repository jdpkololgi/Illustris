"""Target representations for T-Web tidal-tensor eigenvalues.

CANONICAL TARGET POLICY — read before changing any target/head logic.

Two independent choices; do not conflate them:

1. Target *quantity*: models are trained on the tidal-tensor EIGENVALUES
   (λ₁ ≤ λ₂ ≤ λ₃), not on shape-parameter (I₁, e, p) or invariant
   (I₁, I₂, I₃) representations. Those have pathological distributions for ML
   (degenerate I₂/I₃, heavy-tailed I₁, ellipticity piling up near 1) and are
   DEPRECATED as targets — kept here for legacy caches / reference only.

2. Target *parameterisation*: the eigenvalues are trained as ORDERED SOFTPLUS
   INCREMENTS, not as three free λ's:

       v₁ = λ₁                        (anchor, predicted directly)
       v₂ = inverse_softplus(λ₂ - λ₁) (non-negative increment)
       v₃ = inverse_softplus(λ₃ - λ₂) (non-negative increment)

   Reconstruction (`increments_to_eigenvalues`) returns λ₁ ≤ λ₂ ≤ λ₃ BY
   CONSTRUCTION. This is the inductive bias that removes ordering violations.
   Do NOT replace the increment head with a direct 3-output (λ₁, λ₂, λ₃)
   regressor/flow — that silently reintroduces ordering violations.

The network trains and emits in increment space. Inversion to physical
(λ₁, λ₂, λ₃) is applied ONLY at evaluation / plotting time
(`increments_to_eigenvalues`, `samples_to_raw_eigenvalues`).
"""

import numpy as np
import jax.numpy as jnp
#########################################################################
# Shape Parameter Conversion Functions
# DEPRECATED as ML targets (pathological distributions). Legacy/reference only.
# Canonical target = ordered softplus increments (see "Softplus ordering" below).
#########################################################################

def eigenvalues_to_shape_params(eigenvalues):
    """
    Convert ordered eigenvalues (λ₁, λ₂, λ₃) to shape parameters (I₁, e, p)
    
    Args:
        eigenvalues: [N, 3] array of ordered eigenvalues [λ₁, λ₂, λ₃]
    
    Returns:
        [N, 3] array of [I₁, e, p]
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    # I₁: trace (sum of eigenvalues)
    I1 = lambda1 + lambda2 + lambda3
    
    # Trace absolute for normalization
    abs_I1 = jnp.abs(I1)
    
    # e: ellipticity (deviation from sphericity)
    # p: prolateness (prolate vs oblate)
    # If the trace is near zero, shape parameters are ill-defined and physically uninteresting.
    # Set them to zero for stability in empty regions/voids.
    # Using a threshold of 0.01 (units are likely density or potential-based)
    threshold = 1e-2
    mask = abs_I1 > threshold
    
    e = jnp.zeros_like(I1)
    p = jnp.zeros_like(I1)
    
    # Only compute for non-zero tracers
    # Avoid division by zero by using a safe denominator
    # For masked values, denom doesn't matter as result gets masked out anyway, 
    # but we must avoid Inf/NaN in the intermediate calculation.
    safe_denom = jnp.where(mask, 2 * abs_I1, 1.0)
    
    e_val = (lambda3 - lambda1) / safe_denom
    p_val = (lambda1 + lambda3 - 2*lambda2) / safe_denom
    
    # Apply mask
    e = jnp.where(mask, e_val, 0.0)
    p = jnp.where(mask, p_val, 0.0)
    
    # Final safety: Cap e and p to [0, 1] and [-1, 1]
    e = jnp.clip(e, 0.0, 1.0)
    p = jnp.clip(p, -1.0, 1.0)
    
    return jnp.stack([I1, e, p], axis=-1)

def shape_params_to_eigenvalues(params):
    """
    Convert shape parameters (I₁, e, p) to ordered eigenvalues (λ₁, λ₂, λ₃)
    
    Args:
        params: [N, 3] array of [I₁, e, p]
    
    Returns:
        [N, 3] array of ordered eigenvalues [λ₁, λ₂, λ₃]
    """
    I1 = params[:, 0]
    e = params[:, 1]
    p = params[:, 2]
    
    # Linear recovery formulas
    # definitions:
    # I1 = l1 + l2 + l3
    # e = (l3 - l1) / 2|I1|
    # p = (l1 + l3 - 2l2) / 2|I1|
    
    abs_I1 = jnp.abs(I1)
    
    # solving for l1, l2, l3:
    # 3*l2 = I1 - 2*|I1|*p
    lambda2 = I1/3.0 - (2.0/3.0)*abs_I1*p
    
    # solving for l1, l3:
    # l1 = I1/3 + |I1|*(p/3 - e)
    # l3 = I1/3 + |I1|*(p/3 + e)
    
    lambda1 = I1/3.0 + abs_I1 * (p/3.0 - e)
    lambda3 = I1/3.0 + abs_I1 * (p/3.0 + e)
    
    return jnp.stack([lambda1, lambda2, lambda3], axis=-1)

def compute_shape_param_statistics(eigenvalues, train_idx):
    """
    Compute statistics for bounded activations from training eigenvalues
    
    Args:
        eigenvalues: [N, 3] raw eigenvalues
        train_idx: indices of training set
    
    Returns:
        Dictionary with statistics for bounded activations
    """
    # Convert to shape parameters
    shape_params = eigenvalues_to_shape_params(eigenvalues)
    
    # Get training set only
    I1_train = shape_params[train_idx, 0]
    e_train = shape_params[train_idx, 1]
    p_train = shape_params[train_idx, 2]
    
    # Compute statistics (use percentiles to avoid outliers)
    stats = {
        'I1_min': float(jnp.percentile(I1_train, 1)),
        'I1_max': float(jnp.percentile(I1_train, 99)),
        'I1_mean': float(jnp.mean(I1_train)),
        'I1_median': float(jnp.median(I1_train)),
        'e_min': float(jnp.min(e_train)),
        'e_max': float(jnp.percentile(e_train, 99)),
        'e_mean': float(jnp.mean(e_train)),
        'p_min': float(jnp.percentile(p_train, 1)),
        'p_max': float(jnp.percentile(p_train, 99)),
        'p_mean': float(jnp.mean(p_train)),
    }
    

#########################################################################
# Hessian Invariants (I1, I2, I3)
# DEPRECATED as ML targets (degenerate I2/I3, heavy-tailed I1). Reference only.
#########################################################################

def eigenvalues_to_invariants(eigenvalues):
    """
    Convert ordered eigenvalues (λ₁, λ₂, λ₃) to standard invariants (I₁, I₂, I₃).
    
    I₁ = Tr(H) = λ₁ + λ₂ + λ₃
    I₂ = λ₁λ₂ + λ₂λ₃ + λ₃λ₁
    I₃ = Det(H) = λ₁λ₂λ₃
    
    Args:
        eigenvalues: [N, 3] array of ordered eigenvalues
    
    Returns:
        [N, 3] array of [I₁, I₂, I₃]
    """
    l1 = eigenvalues[:, 0]
    l2 = eigenvalues[:, 1]
    l3 = eigenvalues[:, 2]
    
    I1 = l1 + l2 + l3
    I2 = l1*l2 + l2*l3 + l3*l1
    I3 = l1 * l2 * l3
    
    return jnp.stack([I1, I2, I3], axis=-1)

def invariants_to_eigenvalues(invariants):
    """
    Convert invariants (I₁, I₂, I₃) back to sorted eigenvalues.
    Solves the cubic characteristic equation: λ³ - I₁λ² + I₂λ - I₃ = 0
    
    Args:
        invariants: [N, 3] array of [I₁, I₂, I₃]
        
    Returns:
        [N, 3] array of [λ₁, λ₂, λ₃] sorted
    """
    I1 = invariants[:, 0]
    I2 = invariants[:, 1]
    I3 = invariants[:, 2]
    
    # Analytic solution for cubic equation x^3 + a*x^2 + b*x + c = 0
    # Here: x^3 - I1*x^2 + I2*x - I3 = 0
    # So: a = -I1, b = I2, c = -I3
    
    a = -invariants[:, 0]
    b = invariants[:, 1]
    c = -invariants[:, 2]
    
    # Reduction to depressed cubic: t^3 + p*t + q = 0
    # x = t - a/3
    
    p = b - (a**2)/3.0
    q = (2*(a**3))/27.0 - (a*b)/3.0 + c
    
    # Trigonometric solution for 3 real roots (Vieta's substitution)
    # We expect 3 real roots because these characteristic polynomials come from symmetric matrices (Hessians)
    # t_k = 2 * sqrt(-p/3) * cos( (acos(3q/2p * sqrt(-3/p)) - 2pi*k) / 3 )
    
    # Avoid division by zero and sqrt of positive p (should be negative for 3 real roots)
    # For numerical stability with p ~ 0, we can mask, but let's assume valid inputs first.
    
    # Check for p >= 0 case (rare, means 1 real root or multiple roots, usually numerical noise if from real symmetric)
    # Force p to be negative slightly if 0 to avoid NaNs, or handle complex logic.
    # Given the physics, p should be negative.
    
    safe_p = jnp.where(p >= 0, -1e-20, p) # Force negative p for sqrt(-p)
    
    # term inside acos: 3q / (2p) * sqrt(-3/p)
    # let's simplify: 
    # sqrt(-p/3)
    r = jnp.sqrt(-safe_p / 3.0)
    
    # argument for acos: -q / (2 * r^3)
    # But let's follow the formula: cos(phi) = ...
    # 4t^3 - 3t = ... Chebyshev?
    
    # Standard formula:
    # t_k = 2 * sqrt(-p/3) * cos( (1/3) * arccos( (3q)/(2p) * sqrt(-3/p) ) - k*2pi/3 )
    
    # Let's use the explicit form:
    # term = -q / 2 / sqrt(-(p/3)^3)
    term = -q / (2.0 * r**3 + 1e-30) # Add epsilon to avoid div by zero
    
    # Clamp for acos stability [-1, 1]
    term = jnp.clip(term, -1.0, 1.0)
    
    phi = jnp.arccos(term)
    
    t1 = 2.0 * r * jnp.cos(phi / 3.0)
    t2 = 2.0 * r * jnp.cos((phi + 2.0*jnp.pi) / 3.0)
    t3 = 2.0 * r * jnp.cos((phi + 4.0*jnp.pi) / 3.0)
    
    # Recover x = t - a/3
    offset = -a / 3.0
    x1 = t1 + offset
    x2 = t2 + offset
    x3 = t3 + offset
    
    roots = jnp.stack([x1, x2, x3], axis=-1)
    
    # Sort roots per sample to match (lambda1, lambda2, lambda3)
    roots = jnp.sort(roots, axis=-1)
        
    return roots


####################################
# Softplus ordering for eigenvalues
# CANONICAL TARGET PARAMETERISATION: train/predict in increment space; invert
# to (lambda1, lambda2, lambda3) only for evaluation/plotting.
####################################

def eigenvalues_to_increments(eigenvalues):
    """
    Convert sorted eigenvalues (λ₁ <= λ₂ <= λ₃) to softplus increments.
    
    Representation:
    v₁ = λ₁
    v₂ = inverse_softplus(λ₂ - λ₁)
    v₃ = inverse_softplus(λ₃ - λ₂)
    
    This mappings the non-negative differences to the entire real line.
    """
    l1 = eigenvalues[:, 0]
    l2 = eigenvalues[:, 1]
    l3 = eigenvalues[:, 2]
    
    # Differences (must be >= 0)
    d1 = l2 - l1
    d2 = l3 - l2
    
    # Numerical stability
    epsilon = 1e-7
    d1 = jnp.maximum(d1, epsilon)
    d2 = jnp.maximum(d2, epsilon)
    
    # Inverse Softplus: log(exp(d) - 1)
    # Stable implementation: jnp.where(d > 20, d, jnp.log(jnp.expm1(d)))
    v1 = l1
    v2 = jnp.where(d1 > 20.0, d1, jnp.log(jnp.expm1(d1)))
    v3 = jnp.where(d2 > 20.0, d2, jnp.log(jnp.expm1(d2)))
    
    return jnp.stack([v1, v2, v3], axis=-1)


def increments_to_eigenvalues(increments):
    """
    Convert softplus increments back to sorted eigenvalues.
    
    λ₁ = v₁
    λ₂ = λ₁ + softplus(v₂)
    λ₃ = λ₂ + softplus(v₃)
    """
    v1 = increments[:, 0]
    v2 = increments[:, 1]
    v3 = increments[:, 2]
    
    l1 = v1
    # softplus = log(1 + exp(x))
    l2 = l1 + jnp.logaddexp(0., v2)
    l3 = l2 + jnp.logaddexp(0., v3)

    return jnp.stack([l1, l2, l3], axis=-1)


def eigenvalues_to_linear_increments(eigenvalues):
    """Convert sorted eigenvalues (λ₁≤λ₂≤λ₃) to PLAIN linear increments.

    v₁ = λ₁, v₂ = λ₂ - λ₁, v₃ = λ₃ - λ₂ (no inverse-softplus). This is the
    parameterisation used by the 15-d Abacus wedge regression cache. Unlike the
    softplus increments it is well-conditioned (no heavy left tail for small
    gaps), but it does NOT enforce ordering: a model is free to predict/sample a
    negative increment, so λ₂<λ₁ is possible. Only the softplus form guarantees
    λ₁≤λ₂≤λ₃ by construction.
    """
    l1 = eigenvalues[:, 0]
    d2 = eigenvalues[:, 1] - eigenvalues[:, 0]
    d3 = eigenvalues[:, 2] - eigenvalues[:, 1]
    return jnp.stack([l1, d2, d3], axis=-1)


def linear_increments_to_eigenvalues(increments):
    """Invert eigenvalues_to_linear_increments via cumulative sum.

    λ₁ = v₁, λ₂ = λ₁ + v₂, λ₃ = λ₂ + v₃. No clamping — if the increments are
    negative the recovered eigenvalues may violate ordering (by design; see
    eigenvalues_to_linear_increments).
    """
    l1 = increments[:, 0]
    l2 = l1 + increments[:, 1]
    l3 = l2 + increments[:, 2]
    return jnp.stack([l1, l2, l3], axis=-1)


def resolve_increment_mode(use_transformed_eig):
    """Normalise a parameterisation spec to one of 'softplus' | 'linear' | 'raw'.

    Accepts the legacy boolean (True→'softplus', False→'raw') or an explicit
    string, so older call sites that pass a bool keep working unchanged.
    """
    if isinstance(use_transformed_eig, str):
        mode = use_transformed_eig.lower()
        if mode not in ('softplus', 'linear', 'raw'):
            raise ValueError(f"unknown increment mode {use_transformed_eig!r}")
        return mode
    return 'softplus' if use_transformed_eig else 'raw'


####################################
# Utility for SBI/Flow outputs
####################################

def samples_to_raw_eigenvalues(samples, target_scaler, use_transformed_eig):
    """
    Convert samples from flow distribution to raw eigenvalues.
    
    This function handles the inverse transformation pipeline:
    1. Inverse scale (StandardScaler.inverse_transform)
    2. If transformed, convert increments to eigenvalues
    
    Args:
        samples: [N, 3] or [N, K, 3] array of flow samples (scaled)
        target_scaler: sklearn StandardScaler used during training
        use_transformed_eig: parameterisation of the targets — legacy bool
            (True→softplus increments, False→raw eigenvalues) or an explicit
            string 'softplus' | 'linear' | 'raw'.

    Returns:
        Raw eigenvalues [N, 3] or [N, K, 3]
    """
    import numpy as np

    mode = resolve_increment_mode(use_transformed_eig)
    original_shape = samples.shape
    if len(original_shape) == 3:
        # Reshape for scaler: [N*K, 3]
        samples = samples.reshape(-1, 3)

    # Step 1: Inverse scale
    samples_unscaled = target_scaler.inverse_transform(samples)

    # Step 2: invert the increment parameterisation back to eigenvalues
    if mode == 'softplus':
        raw_eig = np.array(increments_to_eigenvalues(jnp.array(samples_unscaled)))
    elif mode == 'linear':
        raw_eig = np.array(linear_increments_to_eigenvalues(jnp.array(samples_unscaled)))
    else:  # 'raw'
        raw_eig = samples_unscaled
    
    # Reshape back if needed
    if len(original_shape) == 3:
        raw_eig = raw_eig.reshape(original_shape)

    return raw_eig


def posterior_to_classprobs(eig_samples, lambda_th=0.2, warn_tol=1e-6):
    """T-Web class probabilities from posterior eigenvalue samples.

    Implements SCIENCE_LOG validation step (1): per galaxy, threshold posterior
    eigenvalue samples at ``lambda_th`` and count how many of (λ₁,λ₂,λ₃) exceed
    it. With ordering λ₁≤λ₂≤λ₃ the crossing count n∈{0,1,2,3} maps directly to
    Void / Wall / Filament / Cluster.

    Two mathematically equivalent estimates are returned and cross-checked:

    * **count-based** — n = Σ_k 1[λ_k > λ_th] per sample, then the fraction of
      samples with n = 0/1/2/3 (order-independent: valid even if a sample
      violates ordering).
    * **marginal-based** — from per-eigenvalue exceedance p_k = P(λ_k > λ_th).
      This module's canonical ordering is **ascending** (λ₁≤λ₂≤λ₃, see
      ``increments_to_eigenvalues``), so p₁≤p₂≤p₃ and the decomposition is
      P(void)=1-p₃, P(wall)=p₃-p₂, P(filament)=p₂-p₁, P(cluster)=p₁. (Note this
      is the mirror of the descending λ₁≥λ₂≥λ₃ convention some external notes
      use — the count-based path is order-independent and is the ground truth.)

    They agree to MC error iff every sample is ordered. A divergence above
    ``warn_tol`` therefore flags an inversion/sampling/ordering bug (e.g. a flow
    trained on a non-order-enforcing parameterisation emitting λ₁>λ₂ samples) and
    is surfaced as ``consistency_max_abs_diff`` (plus a printed warning).

    Args:
        eig_samples: raw eigenvalues, shape [K, 3] (one galaxy) or [N, K, 3].
        lambda_th: T-Web eigenvalue threshold. Default 0.2 — the CACTUS default
            (``threshold=0.2``) used to label the Abacus CWEB column; at 0.2 this
            decomposition reproduces the catalog CWEB exactly. MUST match the
            threshold used to build the ground-truth labels you compare against
            (λ_th=0.0 gives physically wrong fractions for this data).
        warn_tol: tolerance above which the count/marginal mismatch warns.

    Returns:
        dict with keys 'void','wall','filament','cluster' (scalar for [K,3]
        input, shape [N] for [N,K,3]), the per-eigenvalue 'p_exceed', and
        'consistency_max_abs_diff'.
    """
    import numpy as np

    arr = np.asarray(eig_samples)
    single = (arr.ndim == 2)
    if single:
        arr = arr[None, ...]          # [1, K, 3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected [K,3] or [N,K,3] eigenvalue samples, got {arr.shape}")

    exceed = arr > lambda_th          # [N, K, 3] bool

    # count-based: n exceeding per sample -> class fractions over K
    n_cross = exceed.sum(axis=-1)     # [N, K] in {0,1,2,3}
    K = arr.shape[1]
    count = {
        'void':     np.mean(n_cross == 0, axis=1),
        'wall':     np.mean(n_cross == 1, axis=1),
        'filament': np.mean(n_cross == 2, axis=1),
        'cluster':  np.mean(n_cross == 3, axis=1),
    }

    # marginal-based: from per-column exceedance probabilities
    # Ascending convention λ₁≤λ₂≤λ₃  ->  p1≤p2≤p3.
    p = exceed.mean(axis=1)           # [N, 3] = (p1, p2, p3)
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
    marg = {
        'void':     1.0 - p3,
        'wall':     p3 - p2,
        'filament': p2 - p1,
        'cluster':  p1,
    }

    max_diff = max(float(np.max(np.abs(count[c] - marg[c]))) for c in count)
    if max_diff > warn_tol:
        print(f"[posterior_to_classprobs] WARNING: count vs marginal class-prob "
              f"mismatch {max_diff:.3e} > tol {warn_tol:.1e} — likely ordering "
              f"violations in the samples (non-order-enforcing parameterisation?).")

    out = dict(count)
    out['p_exceed'] = p[0] if single else p
    out['consistency_max_abs_diff'] = max_diff
    if single:
        for c in ('void', 'wall', 'filament', 'cluster'):
            out[c] = float(out[c][0])
    return out