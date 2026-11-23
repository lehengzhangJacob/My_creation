# MOSEI sensitivity (paper Fig. `fig:mosei_sensitivity`) — trial 36

Base config: Nov-2026 search **`mosei_deberta` trial 36** (paper PRISM DeBERTa MOSEI row / ablation full model).
Sweeps one of $\beta_{\mathrm{IB}}$, $\alpha_{\mathrm{IB}}$, $\rho_{\mathrm{mse}}$, $\tau_{\mathrm{sel}}$, $\lambda_{\mathrm{rib}}$ inside Table `tab:hyperparam_search_space`; other knobs frozen. Center point is **injected** from trial 36 (not re-trained).

```bash
cd My_creation
# background queue (fills free GPUs, does not kill search)
NEED_MIB=9800 MAX_JOBS=2 bash sensitivity_analysis/run_mosei_trial36_sensitivity_queue.sh
```

Outputs:
- runs: `runs/sensitivity_mosei/trial36/<axis>/v*/`
- summary + PDF: `sensitivity_analysis/results/mosei_trial36/`
- Overleaf: copies over `graph/mosei_trial70_sensitivity_emoe_style_fontlarge_wspace018.pdf` when the queue finishes
