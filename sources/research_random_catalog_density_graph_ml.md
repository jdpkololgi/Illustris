# Random-catalog and density-robust graph/point-cloud methods

Research lookup: 2026-07-14. The preferred `parallel-cli` backend was not
installed, so the references below were verified through primary journal,
conference, arXiv, and collaboration sources.

## Closest astrophysical match: ASTRA

Forero-Romero et al., *Cosmic web classification through stochastic
topological ranking*, RASTI 4 (2025), arXiv:2404.01124,
doi:10.1093/rasti/rzaf032.

- Paper: https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf032/8221862
- Preprint: https://arxiv.org/abs/2404.01124
- Merge an object catalogue O with a random catalogue R that has the same
  selection function and geometry, then Delaunay-triangulate O union R.
- For each point use the bounded local contrast
  `r = (N_O - N_R)/(N_O + N_R)`. Unlike `N_O/N_R - 1`, it remains defined when
  a dense region has no random neighbours.
- Repeat over random-catalogue realizations to get class probabilities and
  entropy.
- Important limitation: its three tested catalogues differ in global number
  density by only about a factor 6.5, not 10^3. It is a strong methodological
  precedent, not validation of three-decade sampling-density transfer.

Zapata-Zuluaga et al., *The Cosmic Web in the DESI Early Data Release: A
Probabilistic Environment Catalog* (2026), arXiv:2604.01456.

- https://arxiv.org/abs/2604.01456
- Applies ASTRA to DESI BGS, LRG, ELG, and QSO catalogues using matched randoms
  and 100 realizations per tracer-zone pair.

## Why matched random catalogues are the correct reference measure

Ross et al., *The Construction of Large-scale Structure Catalogs for the Dark
Energy Spectroscopic Instrument*, JCAP 2025(01)125, arXiv:2405.16593.

- https://arxiv.org/abs/2405.16593
- DESI randoms are an unclustered sampling of the probability that DESI could
  observe a tracer at each location. They encode footprint, completeness, and
  redshift-dependent selection.

Landy & Szalay, *Bias and variance of angular correlation functions*, ApJ 412,
64 (1993), doi:10.1086/172900.

- https://inspirehep.net/literature/369928
- Establishes the data/random reference construction through the nearly
  Poisson-variance estimator `(DD - 2 DR + RR)/RR`.

Cole, *Maximum Likelihood Random Galaxy Catalogues and Luminosity Function
Estimation*, MNRAS 416, 739 (2011), arXiv:1104.0009.

- https://arxiv.org/abs/1104.0009
- Constructs random catalogues for flux-limited samples without fitting away
  real radial large-scale structure.

## Density-adaptive astronomical geometry

Schaap & van de Weygaert, *Continuous Fields and Discrete Samples:
Reconstruction through Delaunay Tessellations*, A&A 363, L29 (2000),
astro-ph/0011007.

- https://arxiv.org/abs/astro-ph/0011007
- DTFE is fully adaptive: high-density regions are resolved finely, while
  low-density regions are reconstructed more smoothly; it also preserves
  anisotropic structures such as walls and filaments.

## ML operators for non-uniform point sampling

Hermosilla et al., *Monte Carlo Convolution for Learning on Non-Uniformly
Sampled Point Clouds*, SIGGRAPH Asia 2018, arXiv:1806.01759.

- https://arxiv.org/abs/1806.01759
- Treats convolution as a Monte Carlo integral and divides each neighbour's
  contribution by its estimated sampling PDF. This is the cleanest theoretical
  basis for density-corrected graph messages.

Wu, Qi & Fuxin, *PointConv: Deep Convolutional Networks on 3D Point Clouds*,
CVPR 2019, arXiv:1811.07246.

- https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.html
- Learns a continuous spatial kernel and applies an inverse KDE density scale
  to compensate non-uniform sampling.

Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a
Metric Space*, NeurIPS 2017, arXiv:1706.02413.

- https://arxiv.org/abs/1706.02413
- Uses multi-scale/multi-resolution grouping plus random input dropout so the
  model learns to change receptive scale as sampling density changes.

Anagnostidis et al., *Cosmology from Galaxy Redshift Surveys with PointNet*,
MNRAS 2023, arXiv:2211.12346.

- https://arxiv.org/abs/2211.12346
- Direct astrophysical use of PointNeXt/PointNet-style models; explicitly
  motivates point methods because voxelization struggles with galaxy-density
  dynamic range and notes adaptive multi-scale processing of non-uniform sets.

Makinen et al., *The Cosmic Graph: Optimal Information Extraction from
Large-Scale Structure using Catalogues*, Open Journal of Astrophysics 2024,
arXiv:2207.05202.

- https://arxiv.org/abs/2207.05202
- Graph-ML precedent for catalogue-level cosmology, including noisy survey
  cuts, but it does not itself solve a 10^3 selection-density transfer problem.

## Recommended combined construction for a GNN

1. Generate a matched random catalogue with intensity proportional to the
   expected selection function, not the observed clustered density.
2. Build a Delaunay or multi-scale radius graph on data plus random nodes, with
   a node-type bit.
3. Aggregate data and random messages separately and expose a stabilized
   contrast such as `(S_D - alpha S_R)/(S_D + alpha S_R + epsilon)`.
4. Add `log nbar` or a random-derived local spacing, and normalize lengths by
   `ell(x) = nbar(x)^(-1/3)`. A 10^3 density range is then a 10x range in mean
   interpoint spacing.
5. Use density-compensated messages (Monte Carlo convolution/PointConv) or
   multi-scale branches, and train with Poisson thinning sampled log-uniformly
   over the intended density range.
6. Validate by held-out density shells and spatial blocks. Randoms remove the
   selection baseline; they cannot restore information lost to shot noise, so
   uncertainty must widen and very sparse cases may remain prior dominated.
