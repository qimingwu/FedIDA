import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations
from sklearn.metrics import roc_auc_score


# Function to compute variance of proportion-based fairness metrics
def proportion_variance(p, n):
    return (p * (1 - p)) / n if n > 0 else 0


# Function to compute confidence intervals and return variance, ensuring values are between 0 and 1
def compute_ci(est, var, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    std_err = np.sqrt(var)

    ci_lower = max(0, est - z * std_err)
    ci_upper = min(1, est + z * std_err)

    return (ci_lower, ci_upper), var


# Function to compute logit-based confidence intervals for probability ratios (DPR, EOR)
def compute_ci_logit_prob(est, var, alpha=0.05):
    if est <= 0 or est >= 1:
        return (max(0, min(1, est)), max(0, min(1, est))), var  # Avoid invalid values

    logit_est = np.log(est / (1 - est))
    std_err = np.sqrt(var) / (est * (1 - est))  # Delta method for logit variance

    z = stats.norm.ppf(1 - alpha / 2)
    logit_ci_lower = logit_est - z * std_err
    logit_ci_upper = logit_est + z * std_err

    # Transform back to probability space
    ci_lower = 1 / (1 + np.exp(-logit_ci_lower))
    ci_upper = 1 / (1 + np.exp(-logit_ci_upper))

    return (ci_lower, ci_upper), var


# Fairness metric variance calculations
def fairness_metrics(y_true, y_pred, groups, alpha=0.05):
    results = {}
    unique_groups = np.unique(groups)

    max_dpd, max_eod, max_fpr, max_ppv = float('-inf'), float('-inf'), float('-inf'), float('-inf')
    min_dpr, min_eor = float('inf'), float('inf')

    for g1, g2 in combinations(unique_groups, 2):
        mask_g1 = (groups == g1)
        mask_g2 = (groups == g2)

        # Demographic Parity (DPD)
        p1 = np.mean(y_pred[mask_g1]) 
        p2 = np.mean(y_pred[mask_g2])
        dpd = p1 - p2
        var_dpd = proportion_variance(p1, np.sum(mask_g1)) + proportion_variance(p2, np.sum(mask_g2))
        max_dpd = max(max_dpd, abs(dpd))

        # Equalized Odds Difference (EOD)
        mask_pos = (y_true == 1)
        p1_tpr = np.mean(y_pred[mask_g1 & mask_pos])
        p2_tpr = np.mean(y_pred[mask_g2 & mask_pos])
        eod = p1_tpr - p2_tpr
        var_eod = proportion_variance(p1_tpr, np.sum(mask_g1 & mask_pos)) + proportion_variance(p2_tpr, np.sum(
            mask_g2 & mask_pos))
        max_eod = max(max_eod, abs(eod))

        # Equal Opportunity Ratio (EOR) - should be minimized
        eor = p1_tpr / p2_tpr if p2_tpr > 0 else np.nan
        var_eor = (var_eod / (p2_tpr ** 2)) if p2_tpr > 0 else np.nan
        min_eor = min(min_eor, eor)

        # False Positive Rate (FPR) Difference
        mask_neg = (y_true == 0)
        p1_fpr = np.mean(y_pred[mask_g1 & mask_neg])
        p2_fpr = np.mean(y_pred[mask_g2 & mask_neg])
        fpr_diff = p1_fpr - p2_fpr
        var_fpr = proportion_variance(p1_fpr, np.sum(mask_g1 & mask_neg)) + proportion_variance(p2_fpr, np.sum(
            mask_g2 & mask_neg))
        max_fpr = max(max_fpr, abs(fpr_diff))

        # Positive Predictive Value (PPV) Difference
        mask_pred_pos = (y_pred == 1)
        p1_ppv = np.mean(y_true[mask_g1 & mask_pred_pos])
        p2_ppv = np.mean(y_true[mask_g2 & mask_pred_pos])
        ppv_diff = p1_ppv - p2_ppv
        var_ppv = proportion_variance(p1_ppv, np.sum(mask_g1 & mask_pred_pos)) + proportion_variance(p2_ppv, np.sum(
            mask_g2 & mask_pred_pos))
        max_ppv = max(max_ppv, abs(ppv_diff))

        # Demographic Parity Ratio (DPR) - should be minimized
        dpr = p1 / p2 if p2 > 0 else np.nan
        var_dpr = (var_dpd / (p2 ** 2)) if p2 > 0 else np.nan
        min_dpr = min(min_dpr, dpr)

    results['Max DPD'] = (max_dpd, *compute_ci(max_dpd, var_dpd, alpha))
    results['Min DPR'] = (min_dpr, *compute_ci_logit_prob(min_dpr, var_dpr, alpha))
    results['Max EOD'] = (max_eod, *compute_ci(max_eod, var_eod, alpha))
    results['Min EOR'] = (min_eor, *compute_ci_logit_prob(min_eor, var_eor, alpha))
    results['Max FPR Difference'] = (max_fpr, *compute_ci(max_fpr, var_fpr, alpha))
    results['Max PPV Difference'] = (max_ppv, *compute_ci(max_ppv, var_ppv, alpha))

    return results

data = pd.read_csv('../outputs/US_hete_race_age_sex1/group/FL_4_sites/FedAvg/FairFML_FedAvg_1.0_0.0112_1.0_0.0112_output_1.csv')
y_test, y_pred, y_prob, sensitive_test = data['y_true'], data['y_pred'], data['y_prob'], data['sensitive_feature']
results = fairness_metrics(y_test, y_pred, sensitive_test)
for metric, (value, ci, var) in results.items():
    print(f"{metric}: {value:.4f}, CI: {ci}, Variance: {var:.6f}")
