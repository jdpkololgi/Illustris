import numpy as np
import jax.numpy as jnp
#########################################################################
# Shape Parameter Conversion Functions
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
        use_transformed_eig: whether targets were transformed
    
    Returns:
        Raw eigenvalues [N, 3] or [N, K, 3]
    """
    import numpy as np
    
    original_shape = samples.shape
    if len(original_shape) == 3:
        # Reshape for scaler: [N*K, 3]
        samples = samples.reshape(-1, 3)
    
    # Step 1: Inverse scale
    samples_unscaled = target_scaler.inverse_transform(samples)
    
    # Step 2: If transformed, convert increments to eigenvalues
    if use_transformed_eig:
        raw_eig = np.array(increments_to_eigenvalues(jnp.array(samples_unscaled)))
    else:
        raw_eig = samples_unscaled
    
    # Reshape back if needed
    if len(original_shape) == 3:
        raw_eig = raw_eig.reshape(original_shape)
    
    return raw_eig