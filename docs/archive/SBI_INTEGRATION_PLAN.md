# SBI Pipeline Integration Plan for Shape Parameters

## Quick Reference: What Needs to Change

### 1. `jraph_sbi_flowjax.py` Updates

#### Location 1: Data Loading (Line ~63-75)
**Current:**
```python
targets = data['regression_targets']  # Scaled eigenvalues
```

**New:**
```python
from eigenvalue_transformations import eigenvalues_to_shape_params

use_shape_params = getattr(args, 'use_shape_params', False)

if use_shape_params:
    # Load raw eigenvalues and transform to shape params
    eigenvalues_raw = data['eigenvalues_raw']
    shape_params_raw = eigenvalues_to_shape_params(eigenvalues_raw)
    targets = jnp.array(shape_params_raw, dtype=jnp.float32)

    # Compute statistics for reporting
    shape_param_stats = compute_shape_param_statistics(eigenvalues_raw, train_mask)
else:
    targets = data['regression_targets']  # Scaled eigenvalues
    shape_param_stats = None
```

#### Location 2: Flow Setup (Line ~107-115)
**Current:**
```python
flow = masked_autoregressive_flow(
    flow_key,
    base_dist=base_dist,
    cond_dim=args.latent_size,
    flow_layers=args.num_flow_layers,
    nn_width=args.flow_hidden_size,
    nn_depth=2,
    transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12),
)
```

**New:**
```python
# Adjust spline range based on representation
if use_shape_params:
    # Shape params: I₁ ∈ [I1_min, I1_max], e ∈ [0, e_max], p ∈ [-p_max, p_max]
    # Use wider range to accommodate all three
    spline_interval = max(shape_param_stats['I1_max'] - shape_param_stats['I1_min'], 1.0)
else:
    # Eigenvalues: scaled to roughly [-12, 12]
    spline_interval = 12

flow = masked_autoregressive_flow(
    flow_key,
    base_dist=base_dist,
    cond_dim=args.latent_size,
    flow_layers=args.num_flow_layers,
    nn_width=args.flow_hidden_size,
    nn_depth=2,
    transformer=RationalQuadraticSpline(knots=args.num_bins, interval=spline_interval),
)
```

#### Location 3: Test Evaluation (Line ~393-423)
**Add after line 423:**
```python
if use_shape_params:
    from eigenvalue_transformations import shape_params_to_eigenvalues

    # TODO: Sample from flow and convert to eigenvalues for physical interpretation
    print("\nGenerating samples for physical interpretation...")

    # Sample for a few test nodes
    test_indices = jnp.where(test_mask)[0][:10]  # First 10 test nodes

    for idx in test_indices:
        embedding = graph.nodes[idx:idx+1]

        # Sample shape parameters from posterior
        # (This requires implementing sampling in the evaluation function)
        # samples_shape = flow.sample(num_samples=100, condition=embedding)

        # Convert to eigenvalues
        # samples_eig = shape_params_to_eigenvalues(samples_shape)

        # Report statistics
        pass  # Implement based on Flowjax API
```

#### Location 4: Add Argument (Line ~441-461)
**Add:**
```python
parser.add_argument('--use_shape_params', action='store_true', default=False,
                    help='Use shape parameters (I₁, e, p) instead of eigenvalues')
```

---

### 2. `jraph_sbi_two_stage.py` Updates

#### Location 1: Stage 1 GNN Training (Line ~51-293)
**The `train_gnn_encoder_stage1()` function needs minimal changes:**

Add at the beginning:
```python
use_shape_params = getattr(args, 'use_shape_params', False)
```

Update target preparation (after line 51):
```python
if use_shape_params:
    from eigenvalue_transformations import eigenvalues_to_shape_params

    # Transform targets to shape parameters
    eigenvalues_raw = eigenvalue_scaler.inverse_transform(targets)
    shape_params = eigenvalues_to_shape_params(eigenvalues_raw)
    targets = jnp.array(shape_params, dtype=jnp.float32)
```

#### Location 2: Stage 2 Flow Training (Line ~337-463)

**Update function signature:**
```python
def train_flow_stage2_ili(args, embeddings_data, targets, train_mask, val_mask,
                         eigenvalue_scaler, use_shape_params=False):
```

**Add transformation (after line 351):**
```python
if use_shape_params:
    from eigenvalue_transformations import eigenvalues_to_shape_params

    # Convert scaled eigenvalues back to raw
    train_eig_raw = eigenvalue_scaler.inverse_transform(targets[train_mask])
    val_eig_raw = eigenvalue_scaler.inverse_transform(targets[val_mask])

    # Transform to shape parameters
    train_targets = eigenvalues_to_shape_params(train_eig_raw)
    val_targets = eigenvalues_to_shape_params(val_eig_raw)

    print(f"Transformed to shape parameters:")
    print(f"  I₁ range: [{train_targets[:, 0].min():.3f}, {train_targets[:, 0].max():.3f}]")
    print(f"  e range: [{train_targets[:, 1].min():.3f}, {train_targets[:, 1].max():.3f}]")
    print(f"  p range: [{train_targets[:, 2].min():.3f}, {train_targets[:, 2].max():.3f}]")
else:
    train_targets = np.array(targets[train_mask])
    val_targets = np.array(targets[val_mask])
```

**Update prior (line ~377-386):**
```python
if use_shape_params:
    # Prior bounds for shape parameters
    I1_min = np.min(train_targets[:, 0])
    I1_max = np.max(train_targets[:, 0])
    e_max = np.max(train_targets[:, 1])
    p_range = np.max(np.abs(train_targets[:, 2]))

    # Add margin for safety
    theta_min = np.array([I1_min - 0.5, 0.0, -p_range - 0.1])
    theta_max = np.array([I1_max + 0.5, e_max + 0.1, p_range + 0.1])
else:
    # Original eigenvalue bounds
    theta_min = np.min(train_targets, axis=0) - 2.0
    theta_max = np.max(train_targets, axis=0) + 2.0
```

**Update validation sampling (line ~439-446):**
```python
if use_shape_params:
    from eigenvalue_transformations import shape_params_to_eigenvalues

    for i in range(num_val_samples):
        x_val_point = torch.tensor(val_emb[i:i+1], dtype=torch.float32)
        samples_shape = posterior.sample((100,), x=x_val_point)

        # Convert to eigenvalues
        samples_eig = shape_params_to_eigenvalues(samples_shape.numpy())

        true_theta_shape = val_targets[i]
        true_theta_eig = shape_params_to_eigenvalues(true_theta_shape.reshape(1, -1))[0]

        sample_mean_shape = samples_shape.mean(dim=0).numpy()
        sample_mean_eig = samples_eig.mean(axis=0)

        print(f"  Val {i}:")
        print(f"    Shape Params - True: {true_theta_shape}, Pred: {sample_mean_shape}")
        print(f"    Eigenvalues  - True: {true_theta_eig}, Pred: {sample_mean_eig}")
else:
    # Original validation code (lines 440-446)
    ...
```

#### Location 3: Main Function (Line ~466-576)
**Update data loading (line ~488-493):**
```python
use_shape_params = getattr(args, 'use_shape_params', False)

# ... existing data loading ...

eigenvalue_scaler = data.get('eigenvalue_scaler', None)
```

**Update function calls (line ~498-500):**
```python
gnn_params, stage1_train_losses, stage1_val_losses, embeddings = train_gnn_encoder_stage1(
    args, graph, targets, train_mask, val_mask, eigenvalue_scaler
)
```

**Update Stage 2 call (line ~529-531):**
```python
posterior, stage2_train_losses, stage2_val_losses, stage2_train_log_probs, stage2_val_log_probs = train_flow_stage2_ili(
    args, embeddings_data, targets, train_mask, val_mask, eigenvalue_scaler, use_shape_params=use_shape_params
)
```

#### Location 4: Add Argument (Line ~579-604)
**Add:**
```python
parser.add_argument('--use_shape_params', action='store_true', default=False,
                    help='Use shape parameters (I₁, e, p) instead of eigenvalues')
```

---

## Testing Commands

### Test `jraph_sbi_flowjax.py` with Shape Parameters:
```bash
python jraph_sbi_flowjax.py \
    --use_shape_params \
    --epochs 5000 \
    --num_flow_layers 5 \
    --num_bins 8 \
    --seed 42
```

### Test `jraph_sbi_two_stage.py` with Shape Parameters:
```bash
python jraph_sbi_two_stage.py \
    --use_shape_params \
    --stage1_epochs 5000 \
    --stage2_epochs 5000 \
    --flow_backend ili \
    --seed 42
```

---

## Validation Checklist

After implementing these changes, verify:

- [ ] Data loads correctly with shape parameters
- [ ] Training runs without errors
- [ ] Flow samples are in correct range:
  - I₁: positive values
  - e: [0, ~0.5]
  - p: [-0.5, 0.5] roughly
- [ ] Inverse transformation to eigenvalues works correctly
- [ ] Posterior samples convert to valid eigenvalue triplets
- [ ] Metrics reported in both shape parameter and eigenvalue spaces

---

## Key Differences from Regression Pipeline

| Aspect | Regression Pipeline | SBI Pipeline |
|--------|-------------------|--------------|
| **Loss Function** | MSE in output space | Negative log-likelihood |
| **Output** | Point predictions | Probability distributions |
| **Scaling** | StandardScaler | Raw shape params (bounded by flow) |
| **Prior** | Not applicable | Must define bounds carefully |
| **Evaluation** | R² scores | Coverage, calibration, CRPS |

---

## Common Pitfalls to Avoid

1. **Don't scale shape parameters twice**: Flow handles bounded ranges, no StandardScaler needed
2. **Check prior bounds**: Shape parameters have different ranges than eigenvalues
3. **Validate transformations**: Always check round-trip eigenvalues → shape params → eigenvalues
4. **Monitor training**: Shape parameter ranges may affect flow convergence
5. **Test edge cases**: Ensure valid eigenvalues for extreme shape parameter values

---

## Expected Benefits for SBI

1. **Tighter posteriors**: Bounded ranges lead to more constrained distributions
2. **Better calibration**: Physical constraints automatically enforced
3. **Faster training**: Better conditioned optimization landscape
4. **More interpretable**: Posteriors in (I₁, e, p) space have clear physical meaning
5. **Improved coverage**: Rotation invariance means more robust to coordinate frame

---

## Next Steps After Integration

1. Run comparison experiments (shape params vs eigenvalues)
2. Validate posterior calibration with rank statistics
3. Create corner plots comparing both representations
4. Measure computational cost differences
5. Document posterior interpretation guidelines
