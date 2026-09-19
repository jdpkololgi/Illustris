# Pinned cosmological reference data

`desi_distance_701f498.dat` is the unmodified numerical DESI distance table from
cosmodesi/cosmoprimo commit701f498eb936a172b91e0ccbd6f52f1770f7154c,
`cosmoprimo/data/desi.dat`. Source URL and SHA256 are recorded in
`../e2e_coupled_coordinates_v2.json`. Columns are redshift, E(z), distance in
Mpc/h. No project data are sent upstream; runtime requires no network fetch.
Do not multiply the table distance by h again. The pinned upstream source
defines DESI as AbacusSummitBase. This data file is an upstream reference,
not an independently computed or phase-fitted cosmology.
The upstream BSD3-Clause copyright, conditions and disclaimer are preserved in
`cosmoprimo-LICENSE` alongside the redistributed table.
