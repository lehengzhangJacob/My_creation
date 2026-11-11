# MOSI/MOSEI Loss Curve Analysis

This folder parses PRISM training logs into the paper-style loss decomposition
and plots MOSI/MOSEI curves.

## Canonical Runs

| Dataset | Run | Hparams | Log |
|---|---|---|---|
| MOSI | more/mosi trial 39 | `../ablation_study/mosi_more_trial39_hparams.py` | `../logs/optuna/4090D_restart/more/mosi/train_logs/mosi_more_mosi_trial_39.log` |
| MOSEI | phase1 trial 70 | `../fixed_experiment/mosei_phase1_trial70_hparams.py` | `../fixed_experiment/runs/mosei_phase1_trial70/train.log` |

MOSI uses the paper-row trial 39. The phase4 trial 234 run under
`fixed_experiment/` is not used for these curves.

## Loss Terms

The plotted terms are:

- `L_total = L_task + alpha_IB * L_tib + alpha_IB * L_lib + lambda_rib * L_rib`
- `L_task`
- `alpha_IB * L_tib`
- `alpha_IB * L_lib`
- `lambda_rib * L_rib`

The log `Detail` values are raw, unweighted values. The parser reads
`alpha_ib`, `selector_rib_weight`, `stage1_epochs`, and
`GRADIENT_ACCUMULATION_STEP` from the frozen hparams files.

The printed `Loss total=` value in `train.log` is scaled by gradient
accumulation, so the paper total is checked against `task + ib`, not directly
against the printed total.

## Reproduce

Run from `My_creation`:

```bash
python3 loss_curve_analysis/parse_train_log_losses.py
python3 loss_curve_analysis/plot_loss_curves.py \
  --output loss_curve_analysis/results/mosi_mosei_loss_curves.pdf
```

Optional PNG:

```bash
python3 loss_curve_analysis/plot_loss_curves.py \
  --output loss_curve_analysis/results/mosi_mosei_loss_curves.pdf \
  --also-png
```

Expected outputs:

- `results/mosi_trial39_losses.csv`
- `results/mosei_trial70_losses.csv`
- `results/mosi_mosei_loss_curves.pdf`

Validation printed by the parser checks that `L_total` is close to `task + ib`
and that stage-1 weighted `L_rib` is zero. The stage-2 starts are epoch 6 for
MOSI and epoch 11 for MOSEI.

## Paper-Ready Figure

The paper-style figure separates the main loss scale from the weighted
regularizer scale:

```bash
python3 loss_curve_analysis/plot_loss_curves.py \
  --paper \
  --output loss_curve_analysis/results/mosi_mosei_loss_curves_paper.pdf \
  --also-png
```

Expected outputs:

- `results/mosi_mosei_loss_curves_paper.pdf`
- `results/mosi_mosei_loss_curves_paper.png`

Suggested caption:

```latex
\caption{Epoch-averaged training-loss decomposition on CMU-MOSI and
CMU-MOSEI. Each point is averaged over all mini-batches in one epoch, and
regularization terms are shown after applying their training weights. Dashed
vertical lines mark the transition from stage 1 to stage 2.}
```

If the PDF is copied to `overleaf_69e83a58/graph/`, the figure snippet can be:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{./graph/mosi_mosei_loss_curves_paper.pdf}
  \caption{Epoch-averaged training-loss decomposition on CMU-MOSI and
  CMU-MOSEI. Each point is averaged over all mini-batches in one epoch, and
  regularization terms are shown after applying their training weights. Dashed
  vertical lines mark the transition from stage 1 to stage 2.}
  \label{fig:loss_curves}
\end{figure*}
```

## Individual Component Figures

To show the real epoch-to-epoch variation more clearly, write one figure per
dataset with each loss component on its own y-axis:

```bash
python3 loss_curve_analysis/plot_loss_curves.py \
  --individual \
  --output loss_curve_analysis/results/loss_components_individual.pdf \
  --also-png
```

Expected outputs:

- `results/loss_components_individual_mosi.pdf`
- `results/loss_components_individual_mosi.png`
- `results/loss_components_individual_mosei.pdf`
- `results/loss_components_individual_mosei.png`

These plots still use real epoch-averaged training losses; they do not add
synthetic noise or smoothing.

## MOAC-Style Iteration-Averaged Curves

MOAC reports learning curves by averaging loss values every ten iterations.
To reproduce that style for MOSI trial 39, rerun training with per-step loss
logging, aggregate every ten iterations, and plot each component separately:

```bash
bash loss_curve_analysis/run_mosi_trial39_step_loss.sh
python3 loss_curve_analysis/aggregate_step_losses.py \
  --input loss_curve_analysis/runs/mosi_trial39_step_loss/step_losses.csv \
  --output loss_curve_analysis/results/mosi_trial39_stepavg10_losses.csv \
  --window 10
python3 loss_curve_analysis/plot_loss_curves.py \
  --moac-step \
  --inputs loss_curve_analysis/results/mosi_trial39_stepavg10_losses.csv \
  --output loss_curve_analysis/results/mosi_trial39_moac_stepavg10_losses.pdf \
  --also-png
```

Expected outputs:

- `runs/mosi_trial39_step_loss/step_losses.csv`
- `results/mosi_trial39_stepavg10_losses.csv`
- `results/mosi_trial39_moac_stepavg10_losses.pdf`
- `results/mosi_trial39_moac_stepavg10_losses.png`

These curves are based on real mini-batch losses averaged over fixed
10-iteration windows, following MOAC's smoothing convention.

### MOSEI trial 70 (same MOAC pipeline)

Rerun with per-step logging (outputs under `loss_curve_analysis/runs/`, not
`fixed_experiment/runs/`):

```bash
# foreground
bash loss_curve_analysis/run_mosei_trial70_step_loss.sh

# or detached
cd My_creation
nohup bash loss_curve_analysis/run_mosei_trial70_step_loss.sh \
  > loss_curve_analysis/runs/mosei_trial70_step_loss/nohup.out 2>&1 &
echo $! > loss_curve_analysis/runs/mosei_trial70_step_loss/train.pid
```

After training finishes:

```bash
bash loss_curve_analysis/finish_mosei_trial70_moac_plot.sh
```

Expected outputs:

- `runs/mosei_trial70_step_loss/step_losses.csv`
- `results/mosei_trial70_stepavg10_losses.csv`
- `results/mosei_trial70_moac_stepavg10_losses.pdf`
- `results/mosei_trial70_moac_stepavg10_losses.png`
