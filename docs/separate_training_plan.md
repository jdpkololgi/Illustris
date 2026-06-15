# Plan: Separate Training of GNN Encoder and Normalizing Flow

## Overview
The current unified SBI pipeline (`jraph_sbi_flowjax.py`) trains the GNN encoder and normalizing flow together, leading to numerical instability. This plan outlines how to train them separately for improved stability.

## Current Architecture Analysis

### Existing Components:
1. **`make_gnn_encoder`** (`graph_net_models.py`): Returns node embeddings `[N, latent_size]`
2. **`jraph_pipeline.py`**: Full regression pipeline using `make_graph_network` (includes decoder)
3. **`jraph_sbi_flowjax.py`**: Unified SBI pipeline with GNN + Flowjax flow

### Key Compatibility Points:
- **JAX/Haiku**: Fully compatible - Haiku modules are JAX-compatible
- **Flowjax/Equinox**: Flowjax uses Equinox, which is JAX-compatible
- **Multi-GPU**: Both pipelines use `jax.pmap` for parallelization

## Implementation Plan

### Phase 1: Train GNN Encoder Separately (Regression Task)

**Objective**: Train the GNN encoder to produce meaningful embeddings that predict eigenvalues.

**Approach**: Modify `jraph_pipeline.py` to use `make_gnn_encoder` + a simple linear decoder.

**Steps**:
1. Create new function `train_gnn_encoder_regression.py`:
   - Use `make_gnn_encoder` from `graph_net_models.py`
   - Add a simple linear decoder: `hk.Linear(3)` to map embeddings → eigenvalues
   - Use MSE loss (same as current regression pipeline)
   - Train on eigenvalue regression task
   - Save trained GNN encoder parameters

2. **Key Implementation Details**:
   ```python
   # Encoder-only GNN
   gnn_fn = make_gnn_encoder(
       num_passes=args.num_passes,
       latent_size=args.latent_size,
       num_heads=args.num_heads,
       dropout_rate=args.dropout,
   )
   gnn = hk.transform(gnn_fn)
   
   # Simple decoder for regression
   def decoder_fn(embeddings):
       return hk.Linear(3)(embeddings)
   decoder = hk.transform(decoder_fn)
   
   # Combined forward pass
   embeddings = gnn.apply(gnn_params, rng, graph, is_training=True)
   predictions = decoder.apply(decoder_params, rng, embeddings)
   ```

3. **Training Configuration**:
   - Use same hyperparameters as current regression pipeline
   - Multi-GPU support via `jax.pmap`
   - Save: `gnn_encoder_params.pkl` + config

4. **Output**: Trained GNN encoder that produces stable embeddings

---

### Phase 2: Extract Embeddings from Trained GNN

**Objective**: Generate embeddings for all training/validation/test nodes using the trained encoder.

**Steps**:
1. Create `extract_embeddings.py`:
   - Load trained GNN encoder parameters
   - Run forward pass: `embeddings = gnn.apply(params, rng, graph, is_training=False)`
   - Extract embeddings for all nodes: `[N, latent_size]`
   - Save embeddings + corresponding eigenvalues: `(embeddings, targets)`

2. **Data Structure**:
   ```python
   {
       'train_embeddings': [N_train, latent_size],
       'train_targets': [N_train, 3],
       'val_embeddings': [N_val, latent_size],
       'val_targets': [N_val, 3],
       'test_embeddings': [N_test, latent_size],
       'test_targets': [N_test, 3],
       'masks': (train_mask, val_mask, test_mask),
   }
   ```

3. **Output**: `node_embeddings_and_targets.pkl`

---

### Phase 3: Train Normalizing Flow Separately

**Objective**: Train normalizing flow on (embedding, eigenvalue) pairs without GNN gradients.

**Approach**: Create standalone flow training script. Two options available:

#### Option A: Using LtU-ILI (Recommended for Astrophysics)

**Steps**:
1. Create `train_flow_separately_ili.py`:
   - Load embeddings and targets from Phase 2
   - Use LtU-ILI's NPE (Neural Posterior Estimator):
     ```python
     from ili.inference import NPE
     
     trainer = NPE(
         x=embeddings_train,  # GNN embeddings [N, latent_size]
         theta=targets_train,  # Eigenvalues [N, 3]
         net='maf',  # Masked autoregressive flow
         hidden_features=args.flow_hidden_size,
         num_transforms=args.num_flow_layers,
     )
     trainer.train(epochs=args.epochs)
     ```

2. **Key Advantages**:
   - High-level API with best practices built-in
   - Built-in validation and diagnostics
   - Astrophysics-specific optimizations
   - Automatic hyperparameter tuning support

3. **Output**: Trained LtU-ILI model (saved via `trainer.save()`)

#### Option B: Using Flowjax Directly (Fine-grained Control)

**Steps**:
1. Create `train_flow_separately_flowjax.py`:
   - Load embeddings and targets from Phase 2
   - Initialize Flowjax conditional flow:
     ```python
     from flowjax.flows import masked_autoregressive_flow
     from flowjax.distributions import Normal
     
     base_dist = Normal(jnp.zeros(3), jnp.ones(3))
     flow = masked_autoregressive_flow(
         flow_key,
         base_dist=base_dist,
         cond_dim=latent_size,  # Conditioning on embeddings
         flow_layers=args.num_flow_layers,
         nn_width=args.flow_hidden_size,
         transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12),
     )
     ```

2. **Training Loop**:
   - Loss: NLL = `-flow.log_prob(targets, condition=embeddings)`
   - No GNN gradients - embeddings are fixed
   - Multi-GPU support via `jax.pmap` (shard embeddings/targets)

3. **Key Advantages**:
   - Full control over flow architecture
   - Direct Equinox integration
   - Matches current implementation

4. **Output**: Trained flow model saved as `.eqx` file

**Recommendation**: Start with LtU-ILI (Option A) for easier implementation, fall back to Flowjax (Option B) if you need more control.

---

### Phase 4: Combined Inference Pipeline

**Objective**: Combine trained GNN encoder + trained flow for SBI inference.

**Steps**:
1. Create `sbi_inference.py`:
   - Load trained GNN encoder parameters
   - Load trained flow model
   - Forward pass:
     ```python
     # Step 1: Get embeddings
     embeddings = gnn.apply(gnn_params, rng, graph, is_training=False)
     
     # Step 2: Sample from flow
     samples = flow.sample(rng, condition=embeddings)  # [N, 3]
     # Or compute log_prob
     log_probs = flow.log_prob(targets, condition=embeddings)  # [N]
     ```

2. **Compatibility**:
   - Both models are JAX-compatible
   - Can use `jax.jit` for fast inference
   - Multi-GPU support if needed

---

## File Structure

```
TNG/Illustris/
├── graph_net_models.py          # Existing: make_gnn_encoder
├── jraph_pipeline.py             # Existing: Full regression pipeline
├── jraph_sbi_flowjax.py          # Existing: Unified SBI (to be replaced)
│
├── train_gnn_encoder_regression.py  # NEW: Phase 1 - Train encoder
├── extract_embeddings.py            # NEW: Phase 2 - Extract embeddings
├── train_flow_separately_ili.py     # NEW: Phase 3 - Train flow (LtU-ILI)
├── train_flow_separately_flowjax.py # NEW: Phase 3 - Train flow (Flowjax)
├── sbi_inference_ili.py              # NEW: Phase 4 - Inference (LtU-ILI)
├── sbi_inference_flowjax.py          # NEW: Phase 4 - Inference (Flowjax)
│
└── docs/
    └── separate_training_plan.md     # This document
```

---

## Compatibility Analysis: JAX/Haiku + Flowjax + LtU-ILI

### ✅ Full Compatibility Confirmed:

1. **Haiku → JAX Arrays**:
   - Haiku parameters are standard JAX pytrees
   - Can be passed to any JAX function
   - Compatible with `jax.jit`, `jax.pmap`, `jax.grad`

2. **Flowjax → JAX Arrays**:
   - Flowjax models are Equinox modules
   - Equinox is JAX-compatible
   - Can use `eqx.filter` to extract trainable arrays

3. **LtU-ILI → JAX Arrays**:
   - LtU-ILI is built on JAX
   - Fully compatible with JAX ecosystem
   - Can accept JAX arrays (embeddings) as input
   - Returns JAX arrays (samples/log_probs)

4. **Data Flow**:
   ```
   Graph → GNN (Haiku) → Embeddings [N, latent_size] (JAX array)
   Embeddings → Flow (LtU-ILI or Flowjax) → log_prob/samples (JAX array)
   ```

5. **Multi-GPU**:
   - All components support `jax.pmap` for parallel training/evaluation
   - Haiku params, Equinox models, and LtU-ILI models can be replicated

---

## Implementation Details

### Phase 1: GNN Encoder Training

**File**: `train_gnn_encoder_regression.py`

**Key Functions**:
```python
def make_encoder_decoder(num_passes, latent_size, num_heads, dropout_rate):
    """Combines encoder + simple decoder."""
    encoder_fn = make_gnn_encoder(...)
    def combined_fn(graph, is_training=True):
        embeddings = encoder_fn(graph, is_training)
        decoder = hk.Linear(3)  # 3 eigenvalues
        return decoder(embeddings)
    return combined_fn

# Training uses MSE loss (same as jraph_pipeline.py regression mode)
```

**Arguments**: Reuse from `jraph_pipeline.py` (same hyperparameters)

---

### Phase 2: Embedding Extraction

**File**: `extract_embeddings.py`

**Key Functions**:
```python
def extract_embeddings(gnn_params, graph, masks, rng):
    """Extract embeddings for all nodes."""
    gnn_fn = make_gnn_encoder(...)
    gnn = hk.transform(gnn_fn)
    
    embeddings = gnn.apply(gnn_params, rng, graph, is_training=False)
    
    # Split by masks
    train_emb = embeddings[train_mask]
    val_emb = embeddings[val_mask]
    test_emb = embeddings[test_mask]
    
    return {
        'train_embeddings': train_emb,
        'val_embeddings': val_emb,
        'test_embeddings': test_emb,
        # ... targets, masks
    }
```

---

### Phase 3: Flow Training

**Option A: Using LtU-ILI** (`train_flow_separately_ili.py`)

**Key Functions**:
```python
from ili.inference import NPE
from ili.validation import Validation

def train_flow_with_ili(embeddings_train, targets_train, 
                        embeddings_val, targets_val, args):
    """Train flow using LtU-ILI framework."""
    
    # Initialize NPE (Neural Posterior Estimator)
    trainer = NPE(
        x=embeddings_train,  # Context: GNN embeddings [N, latent_size]
        theta=targets_train,  # Parameters: eigenvalues [N, 3]
        net='maf',  # Masked autoregressive flow
        hidden_features=args.flow_hidden_size,
        num_transforms=args.num_flow_layers,
    )
    
    # Train (LtU-ILI handles optimization internally)
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # Validate
    validator = Validation(trainer, x_val=embeddings_val, theta_val=targets_val)
    metrics = validator.validate()
    
    # Save
    trainer.save(args.output_dir / 'trained_ili_model')
    
    return trainer, metrics
```

**Advantages**:
- High-level API with best practices
- Built-in validation and diagnostics
- No manual optimization loop needed
- Astrophysics-specific optimizations

**Option B: Using Flowjax** (`train_flow_separately_flowjax.py`)

**Key Functions**:
```python
def train_flow_flowjax(embeddings, targets, masks, args):
    """Train flow on fixed embeddings using Flowjax."""
    # Initialize flow
    flow = masked_autoregressive_flow(...)
    
    # Training loop (no GNN gradients)
    def loss_fn(flow_arrays, embeddings_batch, targets_batch):
        flow_model = eqx.combine(flow_arrays, flow_static)
        log_probs = jax.vmap(flow_model.log_prob)(
            targets_batch, condition=embeddings_batch
        )
        return -jnp.mean(log_probs)
    
    # Optimize only flow parameters
    for epoch in range(args.epochs):
        grads = jax.grad(loss_fn)(flow_arrays, train_emb, train_targets)
        flow_arrays = optax.apply_updates(flow_arrays, updates)
```

**Advantages**:
- Full control over flow architecture
- Direct Equinox integration
- No numerical instability from GNN gradients
- Faster per-iteration (smaller graph)

---

### Phase 4: Inference

**File**: `sbi_inference.py`

**Key Functions**:
```python
def load_models(gnn_checkpoint, flow_checkpoint):
    """Load both trained models."""
    gnn_params = load_gnn_params(gnn_checkpoint)
    flow = eqx.tree_deserialise_leaves(flow_checkpoint, flow_template)
    return gnn_params, flow

@jax.jit
def predict_posterior(gnn_params, flow, graph, rng, num_samples=100):
    """Sample from posterior."""
    # Get embeddings
    embeddings = gnn.apply(gnn_params, rng, graph, is_training=False)
    
    # Sample from flow
    samples = jax.vmap(
        lambda emb, key: flow.sample(key, condition=emb)
    )(embeddings, jax.random.split(rng, len(embeddings)))
    
    return samples  # [N, num_samples, 3]
```

---

## Migration Strategy

### Option A: Gradual Migration (Recommended)
1. Implement Phase 1-4 as new scripts
2. Keep `jraph_sbi_flowjax.py` for comparison
3. Compare results: unified vs. separate training
4. Once validated, deprecate unified training

### Option B: Direct Replacement
1. Implement all phases
2. Replace `jraph_sbi_flowjax.py` with new pipeline
3. Update documentation

---

## Testing & Validation

1. **Phase 1 Validation**:
   - Compare encoder+decoder MSE with full GNN regression
   - Should achieve similar or better performance (focused training)

2. **Phase 2 Validation**:
   - Check embedding statistics (mean, std, no NaNs)
   - Visualize embeddings (PCA/t-SNE) to ensure meaningful structure

3. **Phase 3 Validation**:
   - Monitor flow training loss (should decrease smoothly)
   - Check for numerical stability (no NaNs, no infs)
   - Compare with unified training loss curves

4. **Phase 4 Validation**:
   - Sample from posterior and check coverage
   - Compare log-probabilities with ground truth
   - Test on held-out test set

---

## Expected Benefits

1. **Numerical Stability**:
   - No gradient flow through both GNN and flow
   - Isolated training reduces numerical issues

2. **Training Speed**:
   - Phase 3 (flow training) is faster (smaller graph)
   - Can use different batch sizes/learning rates

3. **Flexibility**:
   - Can retrain flow without retraining GNN
   - Can experiment with different flow architectures

4. **Debugging**:
   - Easier to identify issues (GNN vs. flow)
   - Can validate each component independently

---

## Next Steps

1. ✅ Review this plan
2. Implement Phase 1: `train_gnn_encoder_regression.py`
3. Implement Phase 2: `extract_embeddings.py`
4. Implement Phase 3: `train_flow_separately.py`
5. Implement Phase 4: `sbi_inference.py`
6. Test end-to-end pipeline
7. Compare with unified training results

---

## LtU-ILI Integration

### About LtU-ILI
[LtU-ILI (Learning the Universe - Implicit Likelihood Inference)](https://github.com/maho3/ltu-ili) is a JAX-based SBI framework specifically designed for astrophysics and cosmology. It provides:
- Neural posterior estimation (NPE) with normalizing flows
- Built-in training pipelines for SBI
- JAX/Haiku compatibility
- Multi-GPU support

### Compatibility with Our Stack
✅ **Full Compatibility Confirmed**:
- **JAX-based**: LtU-ILI is built on JAX, ensuring seamless integration
- **Haiku support**: Can work with Haiku modules (our GNN encoder)
- **Multi-GPU**: Supports JAX's `pmap` for parallel training
- **Normalizing flows**: Has built-in support for neural posterior estimation

### Integration Options

#### Option A: Use LtU-ILI for Flow Training (Recommended for Astrophysics)
LtU-ILI provides a high-level API for training neural posterior estimators:
```python
from ili.inference import NPE
from ili.validation import Validation

# Initialize NPE with embeddings as context
trainer = NPE(
    x=embeddings_train,  # [N_train, latent_size] - GNN embeddings
    theta=targets_train,  # [N_train, 3] - eigenvalues
    net='maf',  # Masked autoregressive flow
    flow_architecture='maf',
)

# Train
trainer.train(epochs=args.epochs)

# Validate
validator = Validation(trainer, x_val=embeddings_val, theta_val=targets_val)
```

**Advantages**:
- High-level API with built-in best practices
- Automatic hyperparameter tuning support
- Built-in validation and diagnostics
- Astrophysics-specific optimizations

#### Option B: Use Flowjax Directly (Current Approach)
Continue using Flowjax for fine-grained control:
- More control over flow architecture
- Direct integration with Equinox
- Matches current `jraph_sbi_flowjax.py` implementation

### Updated Phase 3: Flow Training with LtU-ILI

**File**: `train_flow_separately_ili.py` (LtU-ILI version)

**Key Implementation**:
```python
import ili
from ili.inference import NPE
from ili.validation import Validation

def train_flow_with_ili(embeddings_train, targets_train, 
                        embeddings_val, targets_val, args):
    """Train flow using LtU-ILI framework."""
    
    # Initialize NPE
    trainer = NPE(
        x=embeddings_train,  # Context: GNN embeddings [N, latent_size]
        theta=targets_train,   # Parameters: eigenvalues [N, 3]
        net='maf',            # Masked autoregressive flow
        flow_architecture='maf',
        hidden_features=args.flow_hidden_size,
        num_transforms=args.num_flow_layers,
        # Additional LtU-ILI specific args
    )
    
    # Train
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # Validate
    validator = Validation(
        trainer, 
        x_val=embeddings_val, 
        theta_val=targets_val
    )
    metrics = validator.validate()
    
    return trainer, metrics
```

**Alternative: Flowjax Version**
Keep `train_flow_separately.py` using Flowjax for direct control (as in original plan).

### Phase 4: Inference with LtU-ILI

**File**: `sbi_inference_ili.py` (LtU-ILI version)

**Key Implementation**:
```python
from ili.inference import NPE

def load_and_infer(gnn_params, ili_model_path, graph, rng, num_samples=100):
    """Load models and perform inference."""
    
    # Step 1: Get embeddings from GNN
    embeddings = gnn.apply(gnn_params, rng, graph, is_training=False)
    
    # Step 2: Load trained LtU-ILI model
    trainer = NPE.load(ili_model_path)
    
    # Step 3: Sample from posterior
    samples = trainer.sample(
        x=embeddings,  # Condition on embeddings
        num_samples=num_samples,
    )
    
    return samples  # [N, num_samples, 3]
```

### Recommendation

**Hybrid Approach**:
1. **Phase 1-2**: Keep as-is (GNN training + embedding extraction)
2. **Phase 3**: Provide **both options**:
   - `train_flow_separately_ili.py` - Using LtU-ILI (easier, best practices)
   - `train_flow_separately_flowjax.py` - Using Flowjax (more control)
3. **Phase 4**: Provide **both inference scripts**:
   - `sbi_inference_ili.py` - Using LtU-ILI
   - `sbi_inference_flowjax.py` - Using Flowjax

This allows:
- Quick start with LtU-ILI's high-level API
- Fine-grained control with Flowjax when needed
- Comparison between approaches

