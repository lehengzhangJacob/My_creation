# Audit-grade forward-only model complexity

FLOPs are forward-only. For each completed training run, the audit sums every train/dev/test forward call and divides by completed epochs. Batch signatures are cached only when tensor shapes and stage/mode match exactly.

| case | params | trainable | avg_epoch_forward_FLOPs | total_forward_FLOPs | peak_forward_mem_GiB | peak_training_mem_GiB | avg_complete_epoch_sec | epochs | forward calls | call check | B×accum | seq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| mosi | 191,793,425 | 191,793,425 | 11.967T | 442.782T | 2.526 | 8.958 | 14.68 | 37 | 3,293 | OK | 16×2 | 50 |
| mosei | 185,237,940 | 185,237,940 | 140.512T | 7025.622T | 1.565 | 5.715 | 518.72 | 50 | 205,950 | OK | 4×8 | 50 |

## Per-case audit details

### mosi

- FLOP backend: `fvcore+audit_handles`
- Protocol: `forward_only_sum_all_train_dev_test_forwards_by_exact_batch_signature`
- Unsupported ops after audit handles: `{}`
- Stage epoch counts: `{'1': 5, '2': 32}`
- Batches per epoch: `{'train': 81, 'dev': 2, 'test': 6}`
- Time source: `/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation/Model_Complexity_Analysis/nohup_fulltrain_serial_20260525_gpu0_1105.outer.log`
- Training memory source: `/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation/Model_Complexity_Analysis/results_mosi_fulltrain_paper_20260525_gpu0_1105.md`

| split | total forward calls | total forward FLOPs | avg epoch forward FLOPs |
|---|---:|---:|---:|
| train | 2,997 | 266.238T | 7.196T |
| dev | 74 | 44.231T | 1.195T |
| test | 222 | 132.312T | 3.576T |

| split | stage | mode | input shape | calls | FLOPs/forward | peak forward GiB |
|---|---:|---|---|---:|---:|---:|
| dev | 1 | eval | `[101, 50]` | 5 | 528.102G | 1.414 |
| dev | 1 | eval | `[128, 50]` | 5 | 667.340G | 1.593 |
| dev | 2 | eval | `[101, 50]` | 32 | 528.102G | 1.414 |
| dev | 2 | eval | `[128, 50]` | 32 | 667.340G | 1.593 |
| test | 1 | eval | `[128, 50]` | 25 | 667.340G | 1.593 |
| test | 1 | eval | `[45, 50]` | 5 | 239.313G | 1.041 |
| test | 2 | eval | `[128, 50]` | 160 | 667.340G | 1.593 |
| test | 2 | eval | `[45, 50]` | 32 | 239.313G | 1.041 |
| train | 1 | train | `[1, 50]` | 5 | 12.413G | 0.878 |
| train | 1 | train | `[16, 50]` | 400 | 89.790G | 2.526 |
| train | 2 | train | `[1, 50]` | 32 | 12.413G | 0.878 |
| train | 2 | train | `[16, 50]` | 2,560 | 89.790G | 2.526 |

### mosei

- FLOP backend: `fvcore+audit_handles`
- Protocol: `forward_only_sum_all_train_dev_test_forwards_by_exact_batch_signature`
- Unsupported ops after audit handles: `{}`
- Stage epoch counts: `{'1': 10, '2': 40}`
- Batches per epoch: `{'train': 4067, 'dev': 15, 'test': 37}`
- Time source: `/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation/Model_Complexity_Analysis/nohup_fulltrain_serial_20260525_gpu0_1105.outer.log`
- Training memory source: `/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation/Model_Complexity_Analysis/results_mosei_fulltrain_paper_20260525_gpu0_1105.md`

| split | total forward calls | total forward FLOPs | avg epoch forward FLOPs |
|---|---:|---:|---:|
| train | 203,350 | 5425.610T | 108.512T |
| dev | 750 | 459.245T | 9.185T |
| test | 1,850 | 1140.767T | 22.815T |

| split | stage | mode | input shape | calls | FLOPs/forward | peak forward GiB |
|---|---:|---|---|---:|---:|---:|
| dev | 1 | eval | `[128, 50]` | 140 | 628.837G | 1.565 |
| dev | 1 | eval | `[77, 50]` | 10 | 381.173G | 1.229 |
| dev | 2 | eval | `[128, 50]` | 560 | 628.837G | 1.565 |
| dev | 2 | eval | `[77, 50]` | 40 | 381.173G | 1.229 |
| test | 1 | eval | `[128, 50]` | 360 | 628.837G | 1.565 |
| test | 1 | eval | `[35, 50]` | 10 | 177.215G | 0.947 |
| test | 2 | eval | `[128, 50]` | 1,440 | 628.837G | 1.565 |
| test | 2 | eval | `[35, 50]` | 40 | 177.215G | 0.947 |
| train | 1 | train | `[1, 50]` | 10 | 12.112G | 0.840 |
| train | 1 | train | `[4, 50]` | 40,660 | 26.685G | 1.143 |
| train | 2 | train | `[1, 50]` | 40 | 12.112G | 0.840 |
| train | 2 | train | `[4, 50]` | 162,640 | 26.685G | 1.143 |
