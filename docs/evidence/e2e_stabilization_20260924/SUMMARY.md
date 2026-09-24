# Stabilization: exploratory development readout

Four development observations, two seeds. No confirmation or promotion claim.
Covariance is16-probe error; power ratios should be1. All gates include every shell.

| Target | LR | Weights | Update | Mean | Covariance | DC power | Top power | Full gates |
|---|---|---|---:|---:|---:|---:|---:|---:|
| stochastic | constant | raw | 81920 | 0.10413 | 0.09350 | 1.09373 | 1.12733 | 0/8 |
| stochastic | constant | raw | 98304 | 0.10167 | 0.09842 | 1.09176 | 1.12987 | 1/8 |
| stochastic | constant | ema | 81920 | 0.08521 | 0.09544 | 1.10900 | 1.13675 | 0/8 |
| stochastic | constant | ema | 98304 | 0.08232 | 0.09232 | 1.09094 | 1.13886 | 0/8 |
| stochastic | decay | raw | 81920 | 0.09591 | 0.08905 | 1.06409 | 1.13226 | 0/8 |
| stochastic | decay | raw | 98304 | 0.08346 | 0.08583 | 1.02987 | 1.13275 | 0/8 |
| stochastic | decay | ema | 81920 | 0.08423 | 0.09148 | 1.08029 | 1.13474 | 0/8 |
| stochastic | decay | ema | 98304 | 0.08189 | 0.08542 | 1.03008 | 1.13220 | 0/8 |
| exact | constant | raw | 81920 | 0.08508 | 0.08162 | 1.08153 | 1.12305 | 1/8 |
| exact | constant | raw | 98304 | 0.06339 | 0.07537 | 1.03576 | 1.11873 | 2/8 |
| exact | constant | ema | 81920 | 0.05802 | 0.07677 | 1.05615 | 1.11776 | 2/8 |
| exact | constant | ema | 98304 | 0.05492 | 0.07657 | 1.05293 | 1.11317 | 2/8 |
| exact | decay | raw | 81920 | 0.06511 | 0.07833 | 1.04764 | 1.11919 | 1/8 |
| exact | decay | raw | 98304 | 0.05756 | 0.07469 | 1.02811 | 1.11282 | 2/8 |
| exact | decay | ema | 81920 | 0.05843 | 0.07521 | 1.03907 | 1.11645 | 2/8 |
| exact | decay | ema | 98304 | 0.05667 | 0.07459 | 1.02775 | 1.11306 | 2/8 |

## Stability across both checkpoints

- target_exact=False, LR=constant, weights=raw: False
- target_exact=False, LR=constant, weights=ema: False
- target_exact=False, LR=decay, weights=raw: False
- target_exact=False, LR=decay, weights=ema: False
- target_exact=True, LR=constant, weights=raw: False
- target_exact=True, LR=constant, weights=ema: False
- target_exact=True, LR=decay, weights=raw: False
- target_exact=True, LR=decay, weights=ema: False

## DC decomposition (posterior-sd units)

| Fit | Weights | Update | Template | Offset | Scatter | Offset share |
|---|---|---:|---:|---:|---:|---:|
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | raw | 81920 | 0 | -0.12565 | 0.12185 | 0.5153573408144952 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | raw | 81920 | 1 | -0.25090 | 0.04351 | 0.9708038771400241 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | raw | 98304 | 0 | 0.28493 | 0.05256 | 0.9670905852571599 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | raw | 98304 | 1 | 0.10116 | 0.03463 | 0.895108550031583 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | ema | 81920 | 0 | -0.01967 | 0.11541 | 0.02823465871572578 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | ema | 81920 | 1 | -0.08632 | 0.01265 | 0.9789616209939804 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | ema | 98304 | 0 | -0.05280 | 0.09219 | 0.24699171182636126 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_constant | ema | 98304 | 1 | -0.04360 | 0.01830 | 0.8501389971688648 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | raw | 81920 | 0 | -0.05966 | 0.10849 | 0.2321650733587735 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | raw | 81920 | 1 | -0.20308 | 0.03267 | 0.9747718640773079 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | raw | 98304 | 0 | 0.05808 | 0.07703 | 0.3624714591833015 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | raw | 98304 | 1 | 0.03817 | 0.00246 | 0.995868586204841 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | ema | 81920 | 0 | -0.01786 | 0.09599 | 0.03345655949793219 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | ema | 81920 | 1 | -0.06693 | 0.01505 | 0.9518672777331935 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | ema | 98304 | 0 | -0.04359 | 0.07866 | 0.23490509301007728 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | ema | 98304 | 1 | -0.04344 | 0.00561 | 0.9836154084066219 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | raw | 81920 | 0 | -0.63954 | 0.00297 | 0.9999784981992769 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | raw | 81920 | 1 | -0.29604 | 0.02301 | 0.9939962225875599 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | raw | 98304 | 0 | -0.18221 | 0.04684 | 0.9380098962806426 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | raw | 98304 | 1 | -0.31628 | 0.00088 | 0.9999922080065748 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | ema | 81920 | 0 | -0.06492 | 0.02819 | 0.8413859764011272 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | ema | 81920 | 1 | -0.03735 | 0.03601 | 0.518308084168938 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | ema | 98304 | 0 | -0.06726 | 0.02324 | 0.8933178555054688 |
| alpha0.25_white_bridge_exact_seed17_amortised_constant | ema | 98304 | 1 | -0.05183 | 0.00530 | 0.9896525524941698 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | raw | 81920 | 0 | -0.02814 | 0.04740 | 0.2605276399501034 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | raw | 81920 | 1 | -0.05242 | 0.03381 | 0.7062622429704506 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | raw | 98304 | 0 | -0.09786 | 0.02523 | 0.9376778119439301 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | raw | 98304 | 1 | -0.07658 | 0.02417 | 0.9093994043614698 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | ema | 81920 | 0 | -0.09217 | 0.03617 | 0.8665412435828861 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | ema | 81920 | 1 | -0.07603 | 0.02841 | 0.8774794132926642 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | ema | 98304 | 0 | -0.08820 | 0.02547 | 0.9230487666962885 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | ema | 98304 | 1 | -0.05846 | 0.02480 | 0.8474244884301354 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | raw | 81920 | 0 | 0.18309 | 0.02517 | 0.981456996359985 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | raw | 81920 | 1 | 0.13696 | 0.16338 | 0.4126871795020992 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | raw | 98304 | 0 | -0.10270 | 0.13501 | 0.36655133739723533 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | raw | 98304 | 1 | -0.21655 | 0.15469 | 0.6621379797268209 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | ema | 81920 | 0 | -0.10213 | 0.04022 | 0.8657421876413895 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | ema | 81920 | 1 | -0.06750 | 0.14730 | 0.17355031557067174 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | ema | 98304 | 0 | -0.10206 | 0.12647 | 0.394392861927054 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_constant | ema | 98304 | 1 | -0.02934 | 0.09548 | 0.08627835535355304 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | raw | 81920 | 0 | -0.17938 | 0.01039 | 0.996659040190984 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | raw | 81920 | 1 | -0.25254 | 0.13414 | 0.7799425740761843 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | raw | 98304 | 0 | -0.19537 | 0.00763 | 0.9984790568575973 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | raw | 98304 | 1 | -0.10634 | 0.09668 | 0.547480022475997 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | ema | 81920 | 0 | -0.10582 | 0.01907 | 0.9685422067079379 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | ema | 81920 | 1 | -0.05651 | 0.12551 | 0.16853610553662832 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | ema | 98304 | 0 | -0.10747 | 0.00915 | 0.9928108016416455 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | ema | 98304 | 1 | -0.02458 | 0.09175 | 0.06696266029759734 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | raw | 81920 | 0 | 0.47351 | 0.03073 | 0.995804555433535 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | raw | 81920 | 1 | 0.36634 | 0.08778 | 0.9457004515140075 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | raw | 98304 | 0 | -0.06024 | 0.01611 | 0.9332673204159138 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | raw | 98304 | 1 | -0.07373 | 0.04928 | 0.6912126382852051 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | ema | 81920 | 0 | 0.00208 | 0.00102 | 0.806207974684487 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | ema | 81920 | 1 | -0.05918 | 0.07733 | 0.36937790387511926 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | ema | 98304 | 0 | -0.00087 | 0.00100 | 0.43222075400937954 |
| alpha0.25_white_bridge_exact_seed29_amortised_constant | ema | 98304 | 1 | -0.04493 | 0.07698 | 0.254050617693553 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | raw | 81920 | 0 | 0.34096 | 0.01514 | 0.9980309451176772 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | raw | 81920 | 1 | 0.17693 | 0.09383 | 0.7804946435408718 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | raw | 98304 | 0 | -0.01825 | 0.03210 | 0.24426401684190074 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | raw | 98304 | 1 | -0.11105 | 0.08123 | 0.6514426275424626 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | ema | 81920 | 0 | 0.00015 | 0.02053 | 5.309575101302913e-05 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | ema | 81920 | 1 | -0.07016 | 0.08506 | 0.40486280286965903 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | ema | 98304 | 0 | -0.00618 | 0.02950 | 0.042029118778822365 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | ema | 98304 | 1 | -0.07168 | 0.08272 | 0.42889606682514475 |

Only two observations/template; scatter estimates are weakly replicated.
Maximum paired128/256NFE shell change: 0.000347769.
COMPLETE.json also contains MC terms, observed/predicted absorption, DC nulls and exact-target velocity risks.
