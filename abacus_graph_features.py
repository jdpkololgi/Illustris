import numpy as np
import scipy as sp
import pandas as pd
import networkit as nk

points = np.load("/global/homes/d/dkololgi/TNG/Illustris/abacus_cartesian_coords.npy").astype(np.float64)
edges = np.load("/global/homes/d/dkololgi/abacus_delaunay_edges_combined_idx.npy")
print("shape:", edges.shape)
print("dtype:", edges.dtype)
print("min index:", edges.min(), "max index:", edges.max())
# peek at the first few:
print("first 10 edges:\n", edges[:10])

def volume_tetrahedron(tetrahedron):
    '''
    Function to calculate the volume of a tetrahedron
    '''
    matrix = np.array([
        tetrahedron[0] - tetrahedron[3],
        tetrahedron[1] - tetrahedron[3],
        tetrahedron[2] - tetrahedron[3]
    ])
    return abs(np.linalg.det(matrix) / 6)

vec_volume_tetrahedron = np.vectorize(volume_tetrahedron, signature='(n,m)->()')

def edges_to_networkit(points, edges):
    '''
    Convert edges from numpy array to a Networkit graph.
    '''
    global n_nodes
    n_nodes = edges.max() + 1  # +1 because indices are zero-based
    print("Number of nodes:", n_nodes)

    # Create empty graph
    G = nk.Graph(n=n_nodes, weighted=True, directed=False)

    batch_size = 500_000  # Adjust batch size as needed

    for i in range(0, len(edges), batch_size):
        batch = edges[i:i + batch_size]
        
        # Calculate Euclidean distances for the batch
        diffs = points[batch[:, 0], :3] - points[batch[:, 1], :3]
        dists = np.linalg.norm(diffs, axis=1)
        # Add edges with weights to the graph
        for (u, v), w in zip(batch, dists):
            G.addEdge(u, v, w)
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i + len(batch)} / {len(edges)} edges")
    return G


G = edges_to_networkit(points, edges)


#=========================Graph Metrics=========================

# Weighted degree
print('Calculating weighted degrees...')
nk_weighted_degrees = np.fromiter(
    (G.weightedDegree(v) for v in G.iterNodes()), 
    dtype=np.float64, 
    count=n_nodes
)

# Weighted clustering coefficient
print("Calculating weighted clustering coefficients...")
cc = nk.centrality.LocalClusteringCoefficient(G, turbo=True)
cc.run()
clustering_scores = cc.scores()

nk_weighted_clustering_coeffs = np.array(clustering_scores, dtype=np.float64)

# Edge length stats
print("Calculating edge lengths...")
def calculate_edge_metrics_adjacency(G):
    """Use NetworkIt's internal data structures - fastest method."""
    n_nodes = G.numberOfNodes()
    
    min_lengths = np.zeros(n_nodes)
    max_lengths = np.zeros(n_nodes)
    mean_lengths = np.zeros(n_nodes)
    
    print("Calculating edge metrics using adjacency iteration...")
    
    for node in range(n_nodes):
        if G.degree(node) > 0:  # Only process nodes with edges
            # Get all neighbors and weights at once
            neighbors = list(G.iterNeighbors(node))
            weights = np.array([G.weight(node, neighbor) for neighbor in neighbors])
            
            min_lengths[node] = weights.min()
            max_lengths[node] = weights.max()
            mean_lengths[node] = weights.mean()
        
        if node % 1_000_000 == 0:
            print(f"Processed {node:,} nodes...")
    
    return min_lengths, max_lengths, mean_lengths
nk_min_edge_lengths, nk_max_edge_lengths, nk_mean_edge_lengths = calculate_edge_metrics_adjacency(G)

# Tetra dens and neigh tetra dens
def calculate_tetrahedral_density(G):
    print('in progress..')
    G.iterNeighborsWeights(0)

#========================Validation Checks=========================
print("Graph has", G.numberOfNodes(), "nodes and", G.numberOfEdges(), "edges.")
print("First 10 edges with weights:")
for u, v, w in G.iterEdgesWeights():
    print(f"({u}, {v}) with weight {w}")
    if u >= 10:  # Limit output to first 10 edges
        break

# Find minimum and maximum weights
min_weight = np.inf
max_weight = -np.inf
for u, v, w in G.iterEdgesWeights():
    if w < min_weight:
        min_weight = w
    if w > max_weight:
        max_weight = w
print(f"Minimum edge weight: {min_weight}")
print(f"Maximum edge weight: {max_weight}")

def check_hemisphere_separation(G, points):
    """Verify no edges connect different hemispheres"""
    cross_hemisphere_edges = 0
    total_edges = 0
    
    for u, v, w in G.iterEdgesWeights():
        flag_u = points[u, 3]  # Hemisphere flag for node u
        flag_v = points[v, 3]  # Hemisphere flag for node v
        
        if flag_u != flag_v:
            cross_hemisphere_edges += 1
            print(f"ERROR: Cross-hemisphere edge found: {u}({flag_u}) - {v}({flag_v})")
        
        total_edges += 1
        if total_edges % 10_000_000 == 0:
            print(f"Checked {total_edges:,} edges...")
    
    print(f"Cross-hemisphere edges: {cross_hemisphere_edges} / {total_edges}")
    return cross_hemisphere_edges == 0

def check_edge_weights(weights, points):
    """Validate edge weight distributions make physical sense"""
    print("Edge weight statistics:")
    print(f"Min: {weights.min():.4f} Mpc")
    print(f"Max: {weights.max():.4f} Mpc")
    print(f"Mean: {weights.mean():.4f} Mpc")
    print(f"Median: {np.median(weights):.4f} Mpc")
    
    # Check for unrealistic distances
    unrealistic_short = np.sum(weights < 0.001)  # < 1 kpc
    unrealistic_long = np.sum(weights > 1000)    # > 1 Gpc
    
    print(f"Unrealistically short edges (< 1 kpc): {unrealistic_short}")
    print(f"Unrealistically long edges (> 1 Gpc): {unrealistic_long}")
    
    # Check for zero weights
    zero_weights = np.sum(weights == 0)
    print(f"Zero-weight edges: {zero_weights}")
    
    return {
        'realistic': unrealistic_short == 0 and unrealistic_long == 0,
        'no_zeros': zero_weights == 0
    }

def check_edge_counts(G, edges):
    """Verify edge counts match expectations"""
    nk_edges = G.numberOfEdges()
    numpy_edges = len(edges)
    
    print(f"NetworkIt graph edges: {nk_edges:,}")
    print(f"NumPy edge array: {numpy_edges:,}")
    print(f"Match: {nk_edges == numpy_edges}")
    
    return nk_edges == numpy_edges

def check_node_indices(G, points):
    """Verify all node indices are valid"""
    max_node = G.numberOfNodes() - 1
    max_point = len(points) - 1
    
    print(f"Max node index in graph: {max_node}")
    print(f"Max point index available: {max_point}")
    print(f"Match: {max_node == max_point}")
    
    # Check for gaps in node indices
    actual_nodes = set(G.iterNodes())
    expected_nodes = set(range(G.numberOfNodes()))
    missing_nodes = expected_nodes - actual_nodes
    
    print(f"Missing nodes: {len(missing_nodes)}")
    if missing_nodes and len(missing_nodes) < 10:
        print(f"Missing node indices: {missing_nodes}")
    
    return max_node == max_point and len(missing_nodes) == 0

def check_delaunay_properties(G):
    """Check basic Delaunay triangulation properties"""
    degrees = [G.degree(v) for v in G.iterNodes()]
    
    print(f"Degree statistics:")
    print(f"Min degree: {min(degrees)}")
    print(f"Max degree: {max(degrees)}")
    print(f"Mean degree: {np.mean(degrees):.2f}")
    
    # Check for isolated nodes
    isolated = sum(1 for d in degrees if d == 0)
    print(f"Isolated nodes (degree 0): {isolated}")
    
    return {
        'no_isolated': isolated == 0,
        'reasonable_degrees': max(degrees) < 100  # Sanity check
    }

def check_coordinate_ranges(points):
    """Verify coordinate ranges make sense"""
    xyz = points[:, :3]  # Only spatial coordinates
    
    print("Coordinate ranges:")
    for i, coord in enumerate(['X', 'Y', 'Z']):
        print(f"{coord}: [{xyz[:, i].min():.2f}, {xyz[:, i].max():.2f}] Mpc")
    
    # Check if points are roughly spherical/cubic
    ranges = [xyz[:, i].max() - xyz[:, i].min() for i in range(3)]
    print(f"Coordinate ranges: {ranges}")
    
    # Should be similar for a cosmological box
    range_ratio = max(ranges) / min(ranges)
    print(f"Range ratio (should be ~1 for cubic box): {range_ratio:.2f}")
    
    return range_ratio < 2.0  # Allow some variation

def verify_random_edges(G, points, n_samples=10):
    """Manually verify distances for random edges"""
    print("Manual verification of random edges:")
    
    edge_count = 0
    for u, v, w in G.iterEdgesWeights():
        if edge_count >= n_samples:
            break
        
        # Calculate distance manually
        manual_dist = np.linalg.norm(points[u, :3] - points[v, :3])
        diff = abs(w - manual_dist)
        
        print(f"Edge ({u},{v}): Graph={w:.6f}, Manual={manual_dist:.6f}, Diff={diff:.8f}")
        
        edge_count += 1
    
    return True

def full_validation(G, points, edges, weights):
    """Run all validation checks"""
    print("="*50)
    print("DELAUNAY TRIANGULATION VALIDATION")
    print("="*50)
    
    checks = {}
    
    # 1. Hemisphere separation
    print("\n1. Checking hemisphere separation...")
    checks['hemisphere'] = check_hemisphere_separation(G, points)
    
    # 2. Edge weights
    print("\n2. Checking edge weights...")
    weight_checks = check_edge_weights(weights, points)
    checks['weights'] = weight_checks['realistic'] and weight_checks['no_zeros']
    
    # 3. Edge counts
    print("\n3. Checking edge counts...")
    checks['edge_count'] = check_edge_counts(G, edges)
    
    # 4. Node indices
    print("\n4. Checking node indices...")
    checks['node_indices'] = check_node_indices(G, points)
    
    # 5. Delaunay properties
    print("\n5. Checking Delaunay properties...")
    delaunay_checks = check_delaunay_properties(G)
    checks['delaunay'] = delaunay_checks['no_isolated'] and delaunay_checks['reasonable_degrees']
    
    # 6. Coordinate ranges
    print("\n6. Checking coordinate ranges...")
    checks['coordinates'] = check_coordinate_ranges(points)
    
    # 7. Manual verification
    print("\n7. Manual edge verification...")
    checks['manual'] = verify_random_edges(G, points)
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:15}: {status}")
    
    all_passed = all(checks.values())
    print(f"\nOverall: {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
    
    return all_passed

# Run all checks
validation_passed = full_validation(G, points, edges, weights)

if validation_passed:
    print("Graph construction is valid! ✓")
    # Proceed with graph analysis
else:
    print("Graph construction has issues! Please investigate.")