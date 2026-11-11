# Non-language primary and uncertain-routing subsets
## mosei

| group | n | share | label_mean | label_abs_mean | mae | ib_conf_t | ib_conf_a | ib_conf_v |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language | 4634 | 0.9981 | 0.1410 | 0.8385 | 0.4995 | 0.6897 | 0.5052 | 0.5206 |
| acoustic | 9 | 0.0019 | 0.3333 | 0.4074 | 0.4151 | 0.6396 | 0.5051 | 0.5004 |
| visual | 0 | 0.0000 | nan | nan | nan | nan | nan | nan |
Uncertain routing (lowest 20% w_max):

| quantile | w_max_cutoff | n_uncertain | mae_uncertain | mae_rest | nonlanguage_share_uncertain |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 0.4102 | 929 | 0.3629 | 0.5335 | 0.0000 |

## simsv2

| group | n | share | label_mean | label_abs_mean | mae | ib_conf_t | ib_conf_a | ib_conf_v |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language | 1031 | 0.9971 | -0.0485 | 0.5055 | 0.2914 | 0.5814 | 0.5116 | 0.5249 |
| acoustic | 0 | 0.0000 | nan | nan | nan | nan | nan | nan |
| visual | 3 | 0.0029 | 0.7333 | 0.7333 | 0.0643 | 0.6041 | 0.5134 | 0.7292 |
Uncertain routing (lowest 20% w_max):

| quantile | w_max_cutoff | n_uncertain | mae_uncertain | mae_rest | nonlanguage_share_uncertain |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 0.4317 | 207 | 0.3074 | 0.2865 | 0.0097 |

