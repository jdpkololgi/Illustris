# Unchanged-training Gaussian reference continuation

|Model|Updates|Mean RMS|Covariance error|Variance ratio|Octant coverage|Passed cells|
|---|---:|---:|---:|---:|---:|---:|
|vdm_fixed|4096|0.1532|0.3955|1.0970|0.8958|0/4|
|vdm_fixed|8192|0.1386|0.2878|1.0527|0.8972|0/4|
|vdm_fixed|16384|0.1108|0.2917|0.9915|0.8925|0/4|
|vdm_fixed|32768|0.1030|0.2283|0.9881|0.8928|0/4|
|vdm_fixed|65536|0.0805|0.1883|0.9744|0.8957|2/4|
|vdm_fixed (2048 draws)|65536|0.0751|0.1453|0.9758|0.8957|2/4|
|vdm_amortised|4096|0.3686|0.3869|1.1132|0.8888|0/4|
|vdm_amortised|8192|0.2936|0.2973|1.0559|0.8951|0/4|
|vdm_amortised|16384|0.2526|0.2588|1.0865|0.8972|0/4|
|vdm_amortised|32768|0.2279|0.1893|1.0482|0.8844|0/4|
|vdm_amortised|65536|0.1833|0.2041|0.9691|0.8915|0/4|
|vdm_amortised (2048 draws)|65536|0.1814|0.1597|0.9705|0.8925|0/4|
|cfm_fixed|4096|0.0938|0.2483|1.0258|0.9183|0/4|
|cfm_fixed|8192|0.0856|0.1717|1.0073|0.9006|3/4|
|cfm_fixed|16384|0.0707|0.1392|0.9935|0.9013|4/4|
|cfm_fixed|32768|0.0979|0.1424|1.0083|0.8991|1/4|
|cfm_fixed|65536|0.0640|0.1299|1.0039|0.8991|4/4|
|cfm_fixed (2048 draws)|65536|0.0519|0.0674|1.0021|0.9018|4/4|
|cfm_amortised|4096|0.2784|0.2535|1.0853|0.9207|0/4|
|cfm_amortised|8192|0.2221|0.1693|1.0640|0.9126|0/4|
|cfm_amortised|16384|0.1846|0.1477|1.0188|0.9081|0/4|
|cfm_amortised|32768|0.1719|0.1963|1.0527|0.9086|0/4|
|cfm_amortised|65536|0.1438|0.1577|1.0132|0.8857|0/4|
|cfm_amortised (2048 draws)|65536|0.1402|0.0973|1.0115|0.8893|0/4|

Matched cases0/1, both seeds; no best-checkpoint selection. Coverage is exact posterior mass in sample octant intervals, not population TARP. Covariance is a16-probe diagnostic, not full-field certification.

Full draws/checkpoints remain in Scratch; these receipts contain every registered cell.
