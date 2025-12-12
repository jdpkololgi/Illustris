import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import numpy as np
import os
import argparse
from jraph_pipeline import load_data
from graph_net_models import make_graph_network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_size", type=int, default=80)
    parser.add_argument("--num_heads", type=int, default=8)
    args = parser.parse_args()

    # Seeds used in ensemble
    seeds = [42, 43, 44, 45, 46]
    
    print("Loading Data...")
    graph, labels, masks = load_data()
    train_mask, val_mask, test_mask = masks
    
    # Model Definition (Must match training!)
    # Note: Dropout rate irrelevant for inference as is_training=False
    net_fn = make_graph_network(
        num_passes=4, 
        latent_size=args.latent_size, 
        num_heads=args.num_heads, 
        dropout_rate=0.0
    )
    net = hk.transform(net_fn)
    
    preds_list = []
    
    print(f"Evaluating Ensemble of {len(seeds)} models...")
    
    # We can JIT the prediction step
    # We pass a dummy RNG because we removed without_apply_rng, 
    # but is_training=False prevents its usage.
    @jax.jit
    def predict(params, graph, rng):
        return net.apply(params, rng, graph, is_training=False).nodes

    rng = jax.random.PRNGKey(0)

    for seed in seeds:
        fname = f'jraph_model_seed_{seed}.pkl'
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found. Skipping.")
            continue
            
        print(f"Loading {fname}...")
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        
        logits = predict(params, graph, rng)
        preds_list.append(logits)
        
    if not preds_list:
        print("No models found!")
        return

    # Stack [Models, Nodes, Classes]
    all_logits = jnp.stack(preds_list)
    
    # Soft Voting (Average Logits/Probabilities)
    # Averaging logits is standard.
    mean_logits = jnp.mean(all_logits, axis=0)
    
    # Test Accuracy
    preds = jnp.argmax(mean_logits, axis=-1)
    correct = (preds == labels) & test_mask
    test_acc = jnp.sum(correct) / jnp.sum(test_mask)
    
    print("-" * 30)
    print(f"Ensemble Test Accuracy: {test_acc*100:.3f}%")
    print("-" * 30)
    
    # Individual Accuracies
    print("Individual Model Performance:")
    for i, seed in enumerate(seeds):
        fname = f'jraph_model_seed_{seed}.pkl'
        if not os.path.exists(fname): continue
        
        l = preds_list[i]
        p = jnp.argmax(l, axis=-1)
        acc = jnp.sum((p == labels) * test_mask) / jnp.sum(test_mask)
        print(f"Seed {seed}: {acc*100:.3f}%")

if __name__ == "__main__":
    main()
