import scipy.io
import numpy as np
import torch
from kan import KAN
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import euclidean_distances
from collections import Counter
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import contextlib
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc as sklearn_auc
warnings.filterwarnings("ignore")

# 1. Load and balance 
mat_data = scipy.io.loadmat('dynamic_features_for_kan_.mat')
X_all    = mat_data['final_features']
y_all    = mat_data['y_labels'].flatten()

feature_names = [
    'trans_prob', 'occupancy_1',
    'dwell_time_s1', 'dwell_time_s2',
    'GE_s1', 'CC_s1', 'CPL_s1',
    'GE_s2', 'CC_s2', 'CPL_s2',
]

scaler_tmp  = StandardScaler()
X_sc_tmp    = scaler_tmp.fit_transform(X_all)
healthy_idx = np.where(y_all == 0)[0]
mtbi_idx    = np.where(y_all == 1)[0]

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss()
print(f"Device: {device}")

# 2. Configuration 
N_RUNS  = 100
K_FOLDS = 5
THRES = 0.49
STEPS = 30
LAMB = 0.05
LAMB_ENTROPY = 1e-2

FEAT_THRESHOLDS = [0, 0.1, 0.15, 0.2, 0.25, 0.35]

# 3. Core helpers 
def select_features(X_train, y_train, threshold):
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_train)
    X_t  = torch.from_numpy(X_sc).float().to(device)
    y_t  = torch.from_numpy(y_train).long().to(device)
    n    = X_train.shape[1]
    ds   = {'train_input': X_t, 'train_label': y_t,
            'test_input':  X_t, 'test_label':  y_t}
    m    = KAN(width=[n, 2], grid=3, k=3, device=device)
    def _a(): return torch.mean((torch.argmax(m(ds['train_input']),dim=1)==ds['train_label']).float())
    def _b(): return torch.mean((torch.argmax(m(ds['test_input']), dim=1)==ds['test_label']).float())
    with contextlib.redirect_stdout(open(os.devnull, 'w')), \
     contextlib.redirect_stderr(open(os.devnull, 'w')):
        m.fit(ds, opt="Adam", steps=STEPS, loss_fn=loss_fn,
            metrics=(_a, _b), lamb=LAMB, lamb_entropy=LAMB_ENTROPY)
    m(X_t)
    scores   = m.feature_score.detach().cpu().numpy()
    selected = [i for i, s in enumerate(scores) if s >= threshold]
    if len(selected) < 2:
        selected = list(np.argsort(scores)[-2:])
    return selected, scores

def train_predict_proba(X_tr, y_tr, X_te, feat_cols):
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr[:, feat_cols])
    X_te = sc.transform(X_te[:, feat_cols])
    n    = len(feat_cols)
    w    = [n, 2] 
    Xtr  = torch.from_numpy(X_tr).float().to(device)
    Xte  = torch.from_numpy(X_te).float().to(device)
    ytr  = torch.from_numpy(y_tr).long().to(device)
    yte  = torch.zeros(len(X_te), dtype=torch.long).to(device)
    ds   = {'train_input': Xtr, 'train_label': ytr,
            'test_input':  Xte, 'test_label':  yte}
    m    = KAN(width=w, grid=3, k=3, device=device)
    def _a(): return torch.mean((torch.argmax(m(ds['train_input']),dim=1)==ds['train_label']).float())
    def _b(): return torch.mean((torch.argmax(m(ds['test_input']), dim=1)==ds['test_label']).float())
    with contextlib.redirect_stdout(open(os.devnull, 'w')), \
     contextlib.redirect_stderr(open(os.devnull, 'w')):
        results = m.fit(ds, opt="Adam", steps=STEPS, loss_fn=loss_fn,
                    metrics=(_a, _b), lamb=LAMB, lamb_entropy=LAMB_ENTROPY)
    with torch.no_grad():
        prob = torch.softmax(m(Xte), dim=1)[0, 1].item()
    return prob, results['train_loss'], results['test_loss']
    
def plot_kan_loss(results_dict):
    """
    Plots the Training vs Testing loss over epochs.
    results_dict: the dictionary returned by model.fit()
    """
    train_loss = results_dict['train_loss']
    test_loss = results_dict['test_loss']
    
    plt.figure(figsize=(10, 5))
    
    # Plotting both lines
    plt.plot(train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(test_loss, label='Test Loss', color='#ff7f0e', linestyle='--', linewidth=2)
    
    # Identifying the "Overfitting Point"
    min_test_idx = np.argmin(test_loss)
    plt.axvline(x=min_test_idx, color='red', alpha=0.3, label='Potential Overfit Start')
    
    plt.title('KAN Learning Curve: Loss vs. Epochs')
    plt.xlabel('Steps (Epochs)')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()    

def compute_metrics(preds, labels, probs):
    acc  = np.mean(preds == labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc  = roc_auc_score(labels, probs)
    return acc, sens, spec, prec, auc

# 4. Storage for results across all runs 
# Phase 1 (k-fold CV) metrics per run
p1_acc,  p1_sens, p1_spec  = [], [], []
p1_prec, p1_auc             = [], []

# Phase 2 (LOO test) metrics per run
p2_acc,  p2_sens, p2_spec  = [], [], []
p2_prec, p2_auc             = [], []

# Track which features and params were chosen across runs
all_feat_counters  = Counter()
all_param_choices  = []
all_run_train_losses = []
all_run_test_losses = []
all_phase1_probs  = []
all_phase1_labels = []
all_phase2_probs  = []
all_phase2_labels = []
# Track raw scores for every threshold across all runs
all_threshold_scores = {t: [] for t in FEAT_THRESHOLDS}

print("=" * 58)
print(f"Running {N_RUNS} independent repetitions")
print(f"Each run: new 80/20 split → Phase1 ({K_FOLDS}-fold) → Phase2 (LOO)")
print("=" * 58 + "\n")

# 5. Main loop 
for run in range(N_RUNS):

    # Different random seed per run → different 80/20 split each time
    rng = np.random.RandomState(run)

    # Choose randomly 30 out of the 49 healthy subjects 
    sel_healthy_idx = rng.choice(healthy_idx, size=30, replace=False)
    # Combine with the 30 mTBI subjects
    final_idx = np.concatenate([sel_healthy_idx, mtbi_idx])
    # Balanced classes(30-30)
    X_bal = X_all[final_idx]
    y_bal = y_all[final_idx]

    h_idx    = np.where(y_bal == 0)[0]
    m_idx    = np.where(y_bal == 1)[0]
    test_h   = rng.choice(h_idx, size=6,  replace=False)
    test_m   = rng.choice(m_idx, size=6,  replace=False)
    test_idx = np.concatenate([test_h, test_m])
    dev_idx  = np.setdiff1d(np.arange(len(y_bal)), test_idx)

    X_dev  = X_bal[dev_idx]
    y_dev  = y_bal[dev_idx]
    X_test = X_bal[test_idx]
    y_test = y_bal[test_idx]

    # Phase 1: k-fold CV on 48 dev subjects 
    skf            = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                     random_state=run)
    phase1_results = []

    for feat_threshold in FEAT_THRESHOLDS:
        fold_probs, fold_labels = [], []
        fold_feature_lists      = []
        run_fold_scores = []

        for tr_idx, val_idx in skf.split(X_dev, y_dev):
            X_fold_tr,  y_fold_tr  = X_dev[tr_idx],  y_dev[tr_idx]
            X_fold_val, y_fold_val = X_dev[val_idx],  y_dev[val_idx]

            feat_cols, scores = select_features(
                X_fold_tr, y_fold_tr,
                feat_threshold,
            )

            run_fold_scores.append(scores)
            fold_feature_lists.append(
                [feature_names[i] for i in feat_cols]
            )

            for i in range(len(X_fold_val)):
                prob, tr_loss, te_loss = train_predict_proba(
                    X_fold_tr, y_fold_tr,
                    X_fold_val[[i]],
                    feat_cols,
                )
                fold_probs.append(prob)
                fold_labels.append(y_fold_val[i])

        fold_probs  = np.array(fold_probs)
        fold_labels = np.array(fold_labels)
        avg_scores = np.mean(run_fold_scores, axis=0)
        fold_preds  = (fold_probs > THRES).astype(int)
        acc, sens, spec, prec, auc = compute_metrics(
            fold_preds, fold_labels, fold_probs
        )
        phase1_results.append({
            'feat_threshold': feat_threshold,
            'fold_feature_lists': fold_feature_lists,
            'avg_scores': avg_scores,
            'fold_probs': fold_probs.tolist(),    # add this
            'fold_labels': fold_labels.tolist(),  # add this
            'acc': acc, 'sens': sens,
            'spec': spec, 'prec': prec, 'auc': auc
        })

        # Average the scores of the 5 folds for this threshold and store it
        all_threshold_scores[feat_threshold].append(avg_scores)

    # Best config this run
    best = sorted(phase1_results,
                  key=lambda x: (x['auc'], x['acc']), reverse=True)[0]

    best_thresh = best['feat_threshold']
    for res in phase1_results:
        if res['feat_threshold'] == best_thresh:
            all_phase1_probs.extend(res['fold_probs'])
            all_phase1_labels.extend(res['fold_labels'])

    p1_acc.append(best['acc']);   p1_sens.append(best['sens'])
    p1_spec.append(best['spec']); p1_prec.append(best['prec'])
    p1_auc.append(best['auc'])

    # Feature and param tracking
    for fl in best['fold_feature_lists']:
        all_feat_counters.update(fl)
    all_param_choices.append(
        f"thresh={best['feat_threshold']}"
    )

    # Final feature selection on full dev set with best params
    final_feat_cols, scores = select_features(
        X_fold_tr, y_fold_tr,
        best['feat_threshold'],
    )

    # Phase 2: True LOO on 12 test subjects 
    test_preds, test_labels, test_probs = [], [], []

    for i in range(len(X_test)):
        # Train on 48 dev + 11 other test subjects = 59 total
        X_others    = np.delete(X_test, i, axis=0)
        y_others    = np.delete(y_test, i)
        X_train_loo = np.vstack([X_dev, X_others])
        y_train_loo = np.concatenate([y_dev, y_others])

        prob, tr_loss, te_loss = train_predict_proba(
            X_train_loo, y_train_loo,
            X_test[[i]],
            final_feat_cols,
        )
        test_preds.append(int(prob > THRES))
        test_labels.append(y_test[i])
        test_probs.append(prob)
        all_run_train_losses.append(tr_loss)    # store for plotting
        all_run_test_losses.append(te_loss)
        all_phase2_probs.append(prob)
        all_phase2_labels.append(y_test[i])

    test_preds  = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs  = np.array(test_probs)

    acc2, sens2, spec2, prec2, auc2 = compute_metrics(
        test_preds, test_labels, test_probs
    )
    p2_acc.append(acc2);   p2_sens.append(sens2)
    p2_spec.append(spec2); p2_prec.append(prec2)
    p2_auc.append(auc2)

    # Progress every 10 runs
    if (run + 1) % 10 == 0:
        print(f"Run {run+1:3d}/{N_RUNS} | "
              f"P1 AUC={np.mean(p1_auc):.4f} | "
              f"P2 AUC={np.mean(p2_auc):.4f} | "
              f"P2 Acc={np.mean(p2_acc):.4f} ± {np.std(p2_acc):.4f}")
        
all_phase1_probs  = np.array(all_phase1_probs)
all_phase1_labels = np.array(all_phase1_labels)

fpr, tpr, thresholds = roc_curve(all_phase1_labels, all_phase1_probs)
youden        = tpr - fpr
optimal_idx   = np.argmax(youden)
optimal_threshold = thresholds[optimal_idx]

print(f"\nYouden's Index Analysis (across all Phase 1 predictions):")
print(f"  Optimal threshold: {optimal_threshold:.4f}")
print(f"  Sensitivity at optimal: {tpr[optimal_idx]:.4f}")
print(f"  Specificity at optimal: {1 - fpr[optimal_idx]:.4f}")
print(f"  Youden's J: {youden[optimal_idx]:.4f}")
print(f"  Currently used threshold (τ_pred): {THRES}")
print(f"  Difference: {abs(optimal_threshold - THRES):.4f}")

# Mean ROC Curve across all Phase 2 predictions 
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

n_test = 12
for run in range(N_RUNS):
    run_probs  = all_phase2_probs[run * n_test : (run + 1) * n_test]
    run_labels = all_phase2_labels[run * n_test : (run + 1) * n_test]
    fpr, tpr, _ = roc_curve(run_labels, run_probs)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(sklearn_auc(fpr, tpr))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr  = np.std(tprs, axis=0)
mean_auc = np.mean(aucs)
std_auc  = np.std(aucs)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(mean_fpr, mean_tpr, color='#1f77b4', linewidth=2,
        label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
ax.fill_between(mean_fpr,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                alpha=0.15, color='#1f77b4', label='± 1 std')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
        linewidth=1, label='Random chance')

ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
ax.set_title(f'Mean ROC Curve — Phase 2 LOO across {N_RUNS} runs', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('phase2_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("ROC curve saved: phase2_roc_curve.png")

# KAN Architecture Plot (one representative model) 
sc_plot = StandardScaler()
X_plot  = sc_plot.fit_transform(X_bal)
X_plot_t = torch.from_numpy(X_plot).float().to(device)
y_plot_t  = torch.from_numpy(y_bal).long().to(device)

ds_plot = {
    'train_input': X_plot_t, 'train_label': y_plot_t,
    'test_input':  X_plot_t, 'test_label':  y_plot_t
}

m_plot = KAN(width=[10, 2], grid=3, k=3, device=device, seed=22)

with contextlib.redirect_stdout(open(os.devnull, 'w')), \
     contextlib.redirect_stderr(open(os.devnull, 'w')):
    m_plot.fit(ds_plot, opt="Adam", steps=25, loss_fn=loss_fn, lamb=LAMB)

m_plot(X_plot_t)  # forward pass needed before plot

m_plot.plot(
    in_vars=['TP', 'OT₁', 'DT₁', 'DT₂', 'GE₁', 'CC₁', 'CPL₁', 'GE₂', 'CC₂', 'CPL₂'],
    out_vars=['Healthy', 'mTBI'],
    title = '10 feature KAN',
    beta=10,
    scale=0.5,
    sample=True
)

plt.savefig('kan_architecture.png', dpi=300, bbox_inches='tight')
plt.show()        

# 6. Final aggregated results 
print(f"\n{'='*58}")
print(f"  FINAL RESULTS — mean ± std over {N_RUNS} runs")
print(f"{'='*58}")
print(f"\n  Phase 1 — {K_FOLDS}-fold CV (development, n=48)")
print(f"    Accuracy:    {np.mean(p1_acc):.4f} ± {np.std(p1_acc):.4f}")
print(f"    Sensitivity: {np.mean(p1_sens):.4f} ± {np.std(p1_sens):.4f}")
print(f"    Specificity: {np.mean(p1_spec):.4f} ± {np.std(p1_spec):.4f}")
print(f"    AUC-ROC:     {np.mean(p1_auc):.4f} ± {np.std(p1_auc):.4f}")

print(f"\n  Phase 2 — LOO (test set, n=12, true LOO on 59 subjects)")
print(f"    Accuracy:    {np.mean(p2_acc):.4f} ± {np.std(p2_acc):.4f}")
print(f"    Sensitivity: {np.mean(p2_sens):.4f} ± {np.std(p2_sens):.4f}")
print(f"    Specificity: {np.mean(p2_spec):.4f} ± {np.std(p2_spec):.4f}")
print(f"    AUC-ROC:     {np.mean(p2_auc):.4f} ± {np.std(p2_auc):.4f}")

# Feature selection frequency
total_fold_selections = sum(all_feat_counters.values())
print(f"\n  Feature selection frequency across all runs × folds:")
for feat, cnt in sorted(all_feat_counters.items(),
                        key=lambda x: x[1], reverse=True):
    pct = 100 * cnt / total_fold_selections
    bar = "█" * int(pct / 2)
    print(f"    {feat:15} | {cnt:4d} times | {pct:5.1f}% | {bar}")

# Hyperparameter selection frequency
param_counts = Counter(all_param_choices)
print(f"\n  Best config chosen per run:")
for cfg, cnt in param_counts.most_common():
    print(f"    {cfg}: {cnt}/{N_RUNS} runs ({100*cnt/N_RUNS:.0f}%)")

# 8. Consolidated Feature Stability Analysis 
importance_list, selection_list, combined_rows = [], [], []

for feat_idx, feat_name in enumerate(feature_names):
    row_imp, row_sel = [], []
    combined_entry = {'Feature': feat_name}
    
    for t in FEAT_THRESHOLDS:
        scores_for_t = np.array(all_threshold_scores[t])[:, feat_idx]
        avg_score = np.mean(scores_for_t)
        sel_rate  = np.mean(scores_for_t >= t) * 100
        
        row_imp.append(avg_score)
        row_sel.append(sel_rate)
        combined_entry[f'T={t}'] = f"{avg_score:.2f} ({sel_rate:.0f}%)"
    
    importance_list.append(row_imp)
    selection_list.append(row_sel)
    combined_rows.append(combined_entry)

feature_labels = {
    'trans_prob':    'Transition Probability',
    'occupancy_1':   'Occupancy (S1)',
    'dwell_time_s1': 'Dwell Time (S1)',
    'dwell_time_s2': 'Dwell Time (S2)',
    'GE_s1':         'Global Efficiency (S1)',
    'CC_s1':         'Clustering Coeff. (S1)',
    'CPL_s1':        'Char. Path Length (S1)',
    'GE_s2':         'Global Efficiency (S2)',
    'CC_s2':         'Clustering Coeff. (S2)',
    'CPL_s2':        'Char. Path Length (S2)',
}
display_names = [feature_labels[f] for f in feature_names]

# Sort rows by mean importance across thresholds (most important on top) 
df_imp = pd.DataFrame(
    importance_list,
    index=display_names,
    columns=[f'T={t}' for t in FEAT_THRESHOLDS]
)
df_sel = pd.DataFrame(
    selection_list,
    index=display_names,
    columns=[f'T={t}' for t in FEAT_THRESHOLDS]
)

# Sort by mean importance descending so strongest features sit at top
sort_order = df_imp.mean(axis=1).sort_values(ascending=False).index
df_imp = df_imp.loc[sort_order]
df_sel = df_sel.loc[sort_order]

GE_s1 = X_all[:, 4]  # column index for GE_s1
GE_s2 = X_all[:, 7]  # column index for GE_s2
t, p = scipy.stats.ttest_rel(GE_s1, GE_s2, alternative='greater')
print(f"Global Efficiency State 1 vs State 2:")
print(f"t = {t:.4f}, p = {p:.6f}")

# Heatmap
fig, ax = plt.subplots(figsize=(11, 6))

sns.heatmap(
    df_imp,
    annot=df_sel.applymap(lambda x: f'{x:.0f}%'),
    fmt="",
    cmap="YlGnBu",
    linewidths=0.4,
    linecolor='white',
    cbar_kws={'label': 'Mean Intrinsic Importance Score', 'shrink': 0.8},
    ax=ax
)

ax.set_title(
    'KAN Feature Stability: Importance Score & Selection Rate per Threshold',
    fontsize=13, pad=14
)
ax.set_ylabel('MEG Dynamic Feature', fontsize=11)
ax.set_xlabel('Feature Selection Threshold', fontsize=11)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=10)

# Add a note explaining the annotation
fig.text(
    0.5, -0.02,
    'Cell color = mean importance score across 100 runs  |  '
    'Cell label = % of runs in which feature exceeded threshold',
    ha='center', fontsize=9, style='italic', color='gray'
)

plt.tight_layout()
plt.savefig('kan_feature_stability_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ── Appendix table ────────────────────────────────────────────────────────────
appendix_df = pd.DataFrame(combined_rows)
print(f"\n{'='*85}")
print(f"  APPENDIX: FEATURE IMPORTANCE & SELECTION RATES")
print(f"{'='*85}")
print(appendix_df.to_string(index=False))

# ── 7. Loss curve plot across all Phase 2 LOO predictions ────────────────────
# Each Phase 2 prediction produces one loss curve of length STEPS.
# We plot the mean ± std band across all N_RUNS × 12 predictions.
all_train = np.array(all_run_train_losses)  # shape: (N_RUNS*12, STEPS)
all_test  = np.array(all_run_test_losses)   # shape: (N_RUNS*12, STEPS)

mean_train = np.mean(all_train, axis=0)
mean_test  = np.mean(all_test,  axis=0)

# ── Print key values ──────────────────────────────────────────────────────────
print(f"\nLoss Curve Summary:")
print(f"  Train loss — initial (step 1): {mean_train[0]:.4f}")
print(f"  Train loss — final  (step {STEPS}): {mean_train[-1]:.4f}")
print(f"  Test loss  — initial (step 1): {mean_test[0]:.4f}")
print(f"  Test loss  — final  (step {STEPS}): {mean_test[-1]:.4f}")
print(f"  Test loss  — minimum: {np.min(mean_test):.4f} at step {np.argmin(mean_test)+1}")
print(f"  Train/test gap at final step: {mean_train[-1] - mean_test[-1]:.4f}")
print(f"  Random chance baseline: 0.6931")
print(f"  Both curves above random chance: {mean_train[-1] > 0.693 and mean_test[-1] > 0.693}")

steps_range = np.arange(1, STEPS + 1)

plt.figure(figsize=(10, 5))
plt.plot(steps_range, mean_train, color='#1f77b4', linewidth=2,
         label='Train Loss (mean)')

plt.plot(steps_range, mean_test, color='#ff7f0e', linewidth=2,
         linestyle='--', label='Test Loss (mean)')

# Mark the step where mean test loss is minimum
min_test_step = np.argmin(mean_test)
plt.axvline(x=min_test_step + 1, color='red', alpha=0.5,
            linestyle=':', label=f'Min test loss (step {min_test_step+1})')
plt.axhline(y=0.693, color='gray', linewidth=1,
            linestyle=':', label='Random chance (ln 2 ≈ 0.693)')

plt.title(f'KAN Learning Curves — Mean over {N_RUNS} runs × 12 LOO predictions')
plt.xlabel('Training Steps')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('kan_loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("Loss plot saved: kan_loss_curves.png")

print(f"\n{'='*58}")

# 2x2 Mean Connectivity Matrix Heatmap
mat = scipy.io.loadmat('mean_connectivity_matrices.mat')

mean_s1_healthy = mat['mean_net_s1_healthy']  # 90x90
mean_s2_healthy = mat['mean_net_s2_healthy']
mean_s1_mtbi    = mat['mean_net_s1_mtbi']
mean_s2_mtbi    = mat['mean_net_s2_mtbi']

vmin = 0
vmax = np.max([mean_s1_healthy.max(), mean_s2_healthy.max(),
               mean_s1_mtbi.max(),    mean_s2_mtbi.max()])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

titles = [
    ['Healthy — State 1 (High Connectivity)', 'Healthy — State 2 (Low Connectivity)'],
    ['mTBI — State 1 (High Connectivity)',    'mTBI — State 2 (Low Connectivity)']
]
matrices = [
    [mean_s1_healthy, mean_s2_healthy],
    [mean_s1_mtbi,    mean_s2_mtbi]
]

for i in range(2):
    for j in range(2):
        im = axes[i, j].imshow(matrices[i][j], cmap='hot', vmin=vmin, vmax=vmax,
                                aspect='auto')
        axes[i, j].set_title(titles[i][j], fontsize=12)
        axes[i, j].set_xlabel('AAL Region', fontsize=9)
        axes[i, j].set_ylabel('AAL Region', fontsize=9)
        plt.colorbar(im, ax=axes[i, j], shrink=0.8, label='iPLV')

plt.suptitle('Mean Connectivity Matrices per State and Group', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('connectivity_heatmaps_2x2.png', dpi=300, bbox_inches='tight')
plt.show()

# Difference matrices (Healthy minus mTBI) 
diff_s1 = mean_s1_healthy - mean_s1_mtbi
diff_s2 = mean_s2_healthy - mean_s2_mtbi

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
vlim = np.max([np.abs(diff_s1).max(), np.abs(diff_s2).max()])

for ax, diff, title in zip(axes2, 
                            [diff_s1, diff_s2],
                            ['Healthy minus mTBI — State 1 (High Connectivity)',
                             'Healthy minus mTBI — State 2 (Low Connectivity)']):
    im = ax.imshow(diff, cmap='RdBu', vmin=-vlim, vmax=vlim, aspect='auto')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('AAL Region', fontsize=9)
    ax.set_ylabel('AAL Region', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔiPLV (Healthy − mTBI)')

plt.suptitle('Connectivity Differences between Groups per Microstate',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('connectivity_diff_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

mean_abs_diff_s1 = np.mean(np.abs(diff_s1))
mean_abs_diff_s2 = np.mean(np.abs(diff_s2))
print(f"Mean absolute connectivity difference — State 1: {mean_abs_diff_s1:.6f}")
print(f"Mean absolute connectivity difference — State 2: {mean_abs_diff_s2:.6f}")

# Hyperparameter grids to search in Phase 1
svm_params  = [
    {'C': 0.1, 'kernel': 'rbf'},
    {'C': 1,   'kernel': 'rbf'},
    {'C': 10,  'kernel': 'rbf'},
    {'C': 1,   'kernel': 'linear'},
]
mlp_params  = [
    {'hidden_layer_sizes': (5,),  'max_iter': 1000},
    {'hidden_layer_sizes': (10,), 'max_iter': 1000},
    {'hidden_layer_sizes': (20,), 'max_iter': 1000},
]

N_RUNS  = 100
K_FOLDS = 5

results = {
    'SVM': {'acc': [], 'sens': [], 'spec': [], 'auc': []},
    'MLP': {'acc': [], 'sens': [], 'spec': [], 'auc': []}
}

for run in range(N_RUNS):
    rng = np.random.RandomState(run)
    h_idx = np.where(y_bal == 0)[0]
    m_idx = np.where(y_bal == 1)[0]
    test_h   = rng.choice(h_idx, size=6, replace=False)
    test_m   = rng.choice(m_idx, size=6, replace=False)
    test_idx = np.concatenate([test_h, test_m])
    dev_idx  = np.setdiff1d(np.arange(len(y_bal)), test_idx)

    X_dev_r  = X_bal[dev_idx]
    y_dev_r  = y_bal[dev_idx]
    X_test_r = X_bal[test_idx]
    y_test_r = y_bal[test_idx]

    for model_name, param_grid in [('SVM', svm_params), ('MLP', mlp_params)]:

        # Phase 1: 5-fold CV on dev set to select best hyperparameters
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=run)
        best_auc   = -1
        best_params = param_grid[0]

        for params in param_grid:
            fold_probs, fold_labels = [], []

            for tr_idx, val_idx in skf.split(X_dev_r, y_dev_r):
                X_tr, y_tr = X_dev_r[tr_idx], y_dev_r[tr_idx]
                X_val, y_val = X_dev_r[val_idx], y_dev_r[val_idx]

                sc = StandardScaler()
                X_tr_sc  = sc.fit_transform(X_tr)
                X_val_sc = sc.transform(X_val)

                if model_name == 'SVM':
                    clf = SVC(probability=True, **params)
                else:
                    clf = MLPClassifier(**params)

                clf.fit(X_tr_sc, y_tr)
                probs = clf.predict_proba(X_val_sc)[:, 1]
                fold_probs.extend(probs)
                fold_labels.extend(y_val)

            auc = roc_auc_score(fold_labels, fold_probs)
            if auc > best_auc:
                best_auc    = auc
                best_params = params

        # Phase 2: LOO on 12 test subjects with best params
        preds, probs_all, labels = [], [], []

        for i in range(len(X_test_r)):
            X_others    = np.delete(X_test_r, i, axis=0)
            y_others    = np.delete(y_test_r, i)
            X_train_loo = np.vstack([X_dev_r, X_others])
            y_train_loo = np.concatenate([y_dev_r, y_others])

            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_train_loo)
            X_te_sc = sc.transform(X_test_r[[i]])

            if model_name == 'SVM':
                clf = SVC(probability=True, **best_params)
            else:
                clf = MLPClassifier(**best_params)

            clf.fit(X_tr_sc, y_train_loo)
            prob = clf.predict_proba(X_te_sc)[0, 1]
            pred = int(prob > 0.48)

            preds.append(pred)
            probs_all.append(prob)
            labels.append(y_test_r[i])

        preds  = np.array(preds)
        probs_all = np.array(probs_all)
        labels = np.array(labels)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        results[model_name]['acc'].append(np.mean(preds == labels))
        results[model_name]['sens'].append(tp / (tp + fn))
        results[model_name]['spec'].append(tn / (tn + fp))
        results[model_name]['auc'].append(roc_auc_score(labels, probs_all))

    if (run + 1) % 10 == 0:
        print(f"Run {run+1}/{N_RUNS}")

print("\nComparison Results (same protocol as KAN):")
for name in ['SVM', 'MLP']:
    print(f"\n{name}:")
    print(f"  Accuracy:    {np.mean(results[name]['acc']):.4f} ± {np.std(results[name]['acc']):.4f}")
    print(f"  Sensitivity: {np.mean(results[name]['sens']):.4f} ± {np.std(results[name]['sens']):.4f}")
    print(f"  Specificity: {np.mean(results[name]['spec']):.4f} ± {np.std(results[name]['spec']):.4f}")
    print(f"  AUC-ROC:     {np.mean(results[name]['auc']):.4f} ± {np.std(results[name]['auc']):.4f}")