import os
import sys
from pathlib import Path
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print("usage: hyperparameter_tuning.py [--help]\n\nLegacy Optuna hyperparameter tuning script for Jraph pipeline.")
    raise SystemExit(0)

import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import optax
try:
    import optuna
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None
    MedianPruner = None
import pickle
import time

import numpy as np
# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import from existing pipeline
from jraph_pipeline import load_data, calculate_class_weights
from graph_net_models import make_graph_network

# Configuration
N_EPOCHS = 500 # Sufficient to judge potential
N_TRIALS = 50
STORAGE = "sqlite:///jraph_optuna.db"
STUDY_NAME = "jraph_optimization"

def objective(trial):
    # 1. Sample Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    
    latent_size = trial.suggest_categorical("latent_size", [64, 80, 128])
    num_heads = trial.suggest_categorical("num_heads", [4, 8]) 
    num_passes = trial.suggest_int("num_passes", 3, 5)
    
    # Pruning check (fail fast if memory risk with 128 latent on 4 passes?)
    # A100 40GB handled 80 latent, 4 passes. 128 might OOM.
    # We'll try, and catch OOM.

    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    # 2. Load Data (Cached)
    # Re-loading every trial is inefficient if large, but safer for memory isolation.
    # ideally we load once globally.
    global graph, train_mask, val_mask, test_mask, labels, class_weights
    
    # 3. Initialize Model
    net_fn = make_graph_network(
        num_passes=num_passes,
        latent_size=latent_size,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        num_classes=4
    )
    net = hk.transform(net_fn)
    
    # Init Params
    rng = jax.random.PRNGKey(trial.number) # Seed with trial number
    try:
        params = net.init(rng, graph, is_training=True)
    except Exception as e:
        print(f"Pruning trial due to Init Error (likely OOM): {e}")
        raise optuna.TrialPruned()

    # 4. Optimizer
    # Minimal scheduler for tuning
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate / 10,
        peak_value=learning_rate,
        warmup_steps=50,
        decay_steps=N_EPOCHS,
        end_value=learning_rate / 100
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    # 5. Compilation (JIT instead of PMAP for simpler single-process tuning)
    # Using pmap inside Optuna on single GPU can be tricky with processes.
    # We will use simple JIT for the tuning script to run on 1 GPU.
    # Note: mask logic needs adjustment if not using pmap sharding.
    
    # Adjust Masks for JIT (No sharding)
    # We use the full masks directly.
    
    @jax.jit
    def loss_fn(params, graph, labels, mask, rng):
        # Cross Entropy Loss
        logits = net.apply(params, rng, graph, is_training=True).nodes
        labels_one_hot = jax.nn.one_hot(labels, num_classes=4)
        per_node_loss = optax.softmax_cross_entropy(logits, labels_one_hot)
        weights = jnp.take(class_weights, labels)
        loss = jnp.sum(per_node_loss * weights * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        return loss

    @jax.jit
    def update(params, opt_state, graph, labels, mask, rng):
        grads = jax.grad(loss_fn)(params, graph, labels, mask, rng)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    @jax.jit
    def evaluate(params, graph, labels, mask):
        logits = net.apply(params, None, graph, is_training=False).nodes
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.sum((preds == labels) * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        return accuracy

    # 6. Training Loop
    step_rng = rng
    best_val_acc = 0.0
    
    for epoch in range(N_EPOCHS):
        step_rng, train_rng = jax.random.split(step_rng)
        try:
            params, opt_state = update(params, opt_state, graph, labels, train_mask, train_rng)
        except Exception as e:
             print(f"Pruning trial due to OOM during Update: {e}")
             raise optuna.TrialPruned()

        if epoch % 10 == 0:
            val_acc = evaluate(params, graph, labels, val_mask)
            val_acc = float(val_acc)
            
            # Report to Optuna
            trial.report(val_acc, epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
                
    return best_val_acc

if __name__ == "__main__":
    if optuna is None:
        raise ImportError("optuna is required to run this script. Install with `pip install optuna`.")
    # Load Data Once
    print("Loading data...")
    # Fix Unpacking
    graph, labels, masks = load_data()
    train_mask, val_mask, test_mask = masks
    
    # Calculate Weights
    # labels is jax array, convert to numpy for sklearn logic in calculate_class_weights
    class_weights = calculate_class_weights(np.array(labels))
    # Move to device once
    # For JIT, we leave them as numpy or jax arrays, JIT handles transition or we verify device.
    # jraph definitions are usually on host until placed.
    # Let's put them on device explicitly if needed, but JIT handles it.
    
    # Create Study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )
    
    print("Starting Optimization...")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("Best Params:", study.best_params)
    print("Best Acc:", study.best_value)
    
    # Save best params
    with open("best_hyperparameters.pkl", "wb") as f:
        pickle.dump(study.best_params, f)
