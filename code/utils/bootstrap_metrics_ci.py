import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, balanced_accuracy_score
from fairlearn.metrics import (
    make_derived_metric,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate_difference,
    MetricFrame
)
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/Users/qimingwu/DukeNUS/FedScore-Meta/PFL-Non-IID/system')

class BinaryLogisticRegression(nn.Module):
    def __init__(self, n_inputs):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def read_US_data(idx, is_train=True, setting='US_hete4', sensitive_feature='SEX'):
    data_dir = f'~/Aug22/{setting}/'
    if is_train:
        train_file = data_dir + f'C{idx + 1}_train.csv'
        train_tmp = pd.read_csv(train_file, index_col=0)
        X = train_tmp[["AGE", "cause_cardiac", "witnessed", "init_rhythm", "BCPR", "resp_time", "PHDEPIN"]]
        X = pd.get_dummies(X, dtype='int', drop_first=True)
        X = X.sort_index(axis=1)
        y = train_tmp['outcome_neurological'].to_numpy()
        if sensitive_feature == 'race_eth':
            train_data = {'x': X, 'y': y, 'sensitive_feature': pd.get_dummies(train_tmp[sensitive_feature]).drop('WHITE', axis=1)}
        elif sensitive_feature == 'SEX':
            train_data = {'x': X, 'y': y, 'sensitive_feature': train_tmp['SEX'].map({'Male': 1, 'Female': 0})}
        X_train = torch.Tensor(np.array(train_data['x'])).type(torch.float32)
        y_train = torch.Tensor(np.array(train_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(train_data['sensitive_feature'])).type(torch.float32)
        train_data = [(x, y, z) for x, y, z in zip(X_train, y_train, sensitive_feature)]
        return train_data
    else:
        test_file = data_dir + f'C{idx + 1}_test.csv'
        test_tmp = pd.read_csv(test_file, index_col=0)
        X = test_tmp[["AGE", "cause_cardiac", "witnessed", "init_rhythm", "BCPR", "resp_time", "PHDEPIN"]]
        X = pd.get_dummies(X, dtype='int', drop_first=True)
        X = X.sort_index(axis=1)
        y = test_tmp['outcome_neurological'].to_numpy()
        if sensitive_feature == 'race_eth':
            test_data = {'x': X, 'y': y, 'sensitive_feature': pd.get_dummies(test_tmp[sensitive_feature]).drop('WHITE', axis=1)}
        elif sensitive_feature == 'SEX':
            test_data = {'x': X, 'y': y, 'sensitive_feature': test_tmp['SEX'].map({'Male': 1, 'Female': 0})}
        X_test = torch.Tensor(np.array(test_data['x'])).type(torch.float32)
        y_test = torch.Tensor(np.array(test_data['y'])).type(torch.int64)
        sensitive_feature = torch.Tensor(np.array(test_data['sensitive_feature'])).type(torch.float32)
        test_data = [(x, y, z) for x, y, z in zip(X_test, y_test, sensitive_feature)]
        return test_data

# Define a function to compute all metrics
def fairness_and_performance_metrics(y_true, y_pred, y_prob, sensitive_features):
    # Performance metrics
    if len(set(y_true)) > 1:  # Check if there are at least two classes
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = None  # Placeholder when AUC cannot be calculated
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Fairness metrics
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    fpr_diff = false_positive_rate_difference(y_true, y_pred, sensitive_features=sensitive_features)
    # Group-based precision (using MetricFrame for precision difference)
    precision = MetricFrame(metrics=precision_score, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    precision_diff = precision.difference()

    return {
        'AUC': auc,
        'balanced_accuracy': balanced_acc,
        'demographic_parity_difference': dp_diff,
        'demographic_parity_ratio': dp_ratio,
        'equalized_odds_difference': eo_diff,
        'equalized_odds_ratio': eo_ratio,
        'false_positive_rate_difference': fpr_diff,
        'precision_difference': precision_diff
    }

# Bootstrapping function
def bootstrap_fairness_ci(y_true, y_pred, y_prob, sensitive_features, n_bootstraps=100, ci=95):
    results = {key: [] for key in ['AUC',
                                   'balanced_accuracy',
                                   'demographic_parity_difference',
                                   'demographic_parity_ratio',
                                   'equalized_odds_difference',
                                   'equalized_odds_ratio',
                                   'false_positive_rate_difference',
                                   'precision_difference']}
    rng = np.random.default_rng()
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = rng.integers(0, n_samples, size=n_samples)
        y_true_sample = [y_true[i] for i in indices]
        y_pred_sample = [y_pred[i] for i in indices]
        y_prob_sample = [y_prob[i] for i in indices]
        if isinstance(sensitive_features[0], list):
            sensitive_features_sample = pd.DataFrame([sensitive_features[i] for i in indices], columns=['BLACK', 'ASIAN', 'Hispanic'])
            sensitive_features_sample = pd.from_dummies(sensitive_features_sample, default_category='WHITE')
        else:
            sensitive_features_sample = [sensitive_features[i] for i in indices]

        # Compute metrics for the bootstrap sample
        metrics = fairness_and_performance_metrics(y_true_sample, y_pred_sample, y_prob_sample, sensitive_features_sample)
        for key, value in metrics.items():
            results[key].append(value)

    # Compute confidence intervals
    ci_results = {}
    for key, values in results.items():
        # Exclude None values when calculating confidence intervals
        valid_values = [v for v in values if v is not None]
        if valid_values:
            lower = np.percentile(valid_values, (100 - ci) / 2)
            upper = np.percentile(valid_values, 100 - (100 - ci) / 2)
            ci_results[key] = (lower, upper)
        else:
            ci_results[key] = (None, None)  # No valid values to compute CI

    return ci_results

def fairness_metrics(y_true, y_pred, y_prob, sensitive_feature):
    res_dict = {}
    res_dict['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    res_dict['AUC'] = roc_auc_score(y_true, y_prob)
    precision_difference = make_derived_metric(metric=precision_score, transform='difference')
    res_dict['demographic_parity_difference'] = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['demographic_parity_ratio'] = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['equalized_odds_difference'] = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['equalized_odds_ratio'] = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['FPR_difference'] = false_positive_rate_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['precision_difference'] = precision_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    ci = bootstrap_fairness_ci(y_true, y_pred, y_prob, sensitive_features=sensitive_feature)
    return res_dict, ci

def main(num_clients, strategy, setting, lambda_val, gamma_val, sensitive_feature_name='SEX'):
    if lambda_val == 0 and gamma_val == 0:
        model_type = strategy
    else:
        model_type = 'FairFML_' + strategy + '_' + str(lambda_val) + '_' + str(gamma_val)
    result_lst = []
    for site in range(num_clients):
        print('Client %d' % site)
        test_data = read_US_data(site, setting=setting, is_train=False)
        test_loader = DataLoader(dataset=test_data, batch_size=1024, shuffle=False)
        model_directory = f'../outputs/{setting}/models/group/{strategy}/lambda_{lambda_val}/gamma_{gamma_val}/'
        if strategy == 'FedAvg':
            model_name = 'FedAvg_server_10.pt'
            model_path = os.path.join(model_directory, model_name)
            model = torch.load(model_path)
        elif strategy == 'PerAvg':
            model_name = f'PerAvg_client{site}_10.pt'
            model_path = os.path.join(model_directory, model_name)
            model = BinaryLogisticRegression(n_inputs=7)
            model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError('Invalid strategy')
        model.eval()
        y_prob, y_true, y_pred, all_sensitive_feature, X = [], [], [], [], []
        for x, y, sensitive_feature in test_loader:
            X.extend(x.numpy())
            output = model(x).reshape(1, -1)[0].type(torch.float32)
            y_prob.append(output.detach())
            y_true.append(y.detach())
            all_sensitive_feature.append(sensitive_feature.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_prob >= 0.1).astype('int')
        all_sensitive_feature = np.concatenate(all_sensitive_feature, axis=0)
        if sensitive_feature_name == 'race_eth':
            sensitive_features = pd.DataFrame(all_sensitive_feature, columns=['BLACK', 'ASIAN', 'Hispanic'])
            sensitive_features = pd.from_dummies(sensitive_features, default_category='WHITE').to_numpy()
            sensitive_features = [x[0] for x in sensitive_features]
        elif sensitive_feature_name == 'SEX':
            sensitive_features = pd.Series(all_sensitive_feature)
        df = pd.DataFrame({'y_true': list(y_true), 'y_pred': list(y_pred), 'y_prob': list(y_prob), 'sensitive_feature': list(sensitive_features)})
        if sensitive_feature_name == 'SEX':
            df.to_csv(f'../outputs/{setting}/group/FL_{num_clients}_sites/{strategy}_sex/{model_type}_{lambda_val}_{gamma_val}_output_{site}.csv', index=False)
        else:
            df.to_csv(f'../outputs/{setting}/group/FL_{num_clients}_sites/{strategy}/{model_type}_{lambda_val}_{gamma_val}_output_{site}.csv', index=False)

        fairness, ci = fairness_metrics(y_true, y_pred, y_prob, all_sensitive_feature)
        result_lst.append([site, model_type, fairness['AUC'], fairness['demographic_parity_difference'], fairness['demographic_parity_ratio'],
                           fairness['equalized_odds_difference'], fairness['equalized_odds_ratio'],
                           fairness['FPR_difference'], fairness['precision_difference'],
                           ci['AUC'][0], ci['AUC'][1], ci['demographic_parity_difference'][0], ci['demographic_parity_difference'][1],
                           ci['demographic_parity_ratio'][0], ci['demographic_parity_ratio'][1],
                           ci['equalized_odds_difference'][0], ci['equalized_odds_difference'][1],
                           ci['equalized_odds_ratio'][0], ci['equalized_odds_ratio'][1],
                           ci['false_positive_rate_difference'][0], ci['false_positive_rate_difference'][1],
                           ci['precision_difference'][0], ci['precision_difference'][1]])
    result_df = pd.DataFrame.from_records(result_lst, columns=['site', 'model', 'AUC', 'DPD', 'DPR', 'EOD', 'EOR', 'FPR_diff', 'PPV_diff', 'AUC_lower', 'AUC_upper',
                                                               'DPD_lower', 'DPD_upper', 'DPR_lower', 'DPR_upper', 'EOD_lower', 'EOD_upper',
                                                               'EOR_lower', 'EOR_upper', 'FPR_lower', 'FPR_upper', 'PPV_lower', 'PPV_upper'])
    print(result_df)
    if sensitive_feature_name == 'SEX':
        result_df.to_csv(f'../outputs/{setting}/group/FL_{num_clients}_sites/{strategy}_sex/{model_type}_{lambda_val}_{gamma_val}_CI.csv', index=False)
    else:
        result_df.to_csv(f'../outputs/{setting}/group/FL_{num_clients}_sites/{strategy}/{model_type}_{lambda_val}_{gamma_val}_CI.csv', index=False)


if __name__ == '__main__':
    strategy = sys.argv[1]
    setting = sys.argv[2]
    lambda_val = float(sys.argv[3])
    gamma_val = float(sys.argv[4])
    if setting == 'US_hete3' or setting == 'US_hete4':
        num_clients = 6
    else:
        num_clients = 4
    main(num_clients, strategy, setting, lambda_val, gamma_val)
