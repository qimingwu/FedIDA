import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference, equalized_odds_ratio, equalized_odds_difference
from aif360.sklearn.metrics import consistency_score, generalized_entropy_error


class MulticlassGroupFairnessLoss(nn.Module):
    def __init__(self, fairness_lambda=1, L2_gamma=0, average='ovr'):
        super(MulticlassGroupFairnessLoss, self).__init__()
        self.fairness_lambda = fairness_lambda
        self.L2_gamma = L2_gamma
        self.average = average

    def forward(self, y_pred, y_true, dataset, sensitive_cat, model):
        if self.fairness_lambda > 0:
            fairness_loss_list = []
            if self.average == 'ovr':  # One-vs-rest
                all_classes = np.unique(sensitive_cat)
                for c in all_classes:
                    fairness_loss = 0
                    pos_data, neg_data = dataset[dataset[c] == 1], dataset[dataset[c] == 0]
                    pos_target, neg_target = pos_data['y'].reset_index(drop=True), neg_data['y'].reset_index(drop=True)
                    pos_data, neg_data = pos_data.drop(['y'] + sensitive_cat, axis=1), neg_data.drop(['y'] + sensitive_cat, axis=1)
                    pos_len, neg_len = len(pos_data), len(neg_data)
                    if min(pos_len, neg_len) == 0:
                        continue
                    pos_data, neg_data = torch.tensor(pos_data.values).to(torch.float), torch.tensor(neg_data.values).to(torch.float)
                    pos_data_output, neg_data_output = ([model(pos_data[i]) for i in range(len(pos_data))],
                                                        [model(neg_data[i]) for i in range(len(neg_data))])
                    selected_pairs = np.random.choice(pos_len * neg_len, 2 * min(pos_len, neg_len), replace=False)
                    idx = 0
                    for i in range(pos_len):
                        for j in range(neg_len):
                            if idx in selected_pairs and pos_target[i] == neg_target[j]:
                                fairness_loss += (pos_data_output[i] - neg_data_output[j]) ** 2
                            idx += 1
                    with torch.no_grad():
                        c_scaled_fairness_loss = self.fairness_lambda * (fairness_loss / (2 * min(pos_len, neg_len))) ** 2
                        fairness_loss_list.append((pos_len, c_scaled_fairness_loss))
            elif self.average == 'ovo':  # One-vs-one
                all_classes = list(itertools.combinations(sensitive_cat, r=2))
                for c1, c2 in all_classes:
                    fairness_loss = 0
                    pos_data, neg_data = dataset[dataset[c1] == 1], dataset[dataset[c2] == 1]
                    pos_target, neg_target = pos_data['y'].reset_index(drop=True), neg_data['y'].reset_index(drop=True)
                    pos_data, neg_data = pos_data.drop(['y'] + sensitive_cat, axis=1), neg_data.drop(['y'] + sensitive_cat, axis=1)
                    pos_len, neg_len = len(pos_data), len(neg_data)
                    if min(pos_len, neg_len) == 0:
                        continue
                    pos_data, neg_data = torch.tensor(pos_data.values).to(torch.float), torch.tensor(neg_data.values).to(torch.float)
                    pos_data_output, neg_data_output = ([model(pos_data[i]) for i in range(len(pos_data))],
                                                        [model(neg_data[i]) for i in range(len(neg_data))])
                    selected_pairs = np.random.choice(pos_len * neg_len, 2 * min(pos_len, neg_len), replace=False)
                    idx = 0
                    for i in range(pos_len):
                        for j in range(neg_len):
                            if idx in selected_pairs and pos_target[i] == neg_target[j]:
                                fairness_loss += (pos_data_output[i] - neg_data_output[j]) ** 2
                            idx += 1
                    with torch.no_grad():
                        c_scaled_fairness_loss = self.fairness_lambda * (fairness_loss / (2 * min(pos_len, neg_len))) ** 2
                        fairness_loss_list.append((pos_len + neg_len, c_scaled_fairness_loss))
            else:
                raise ValueError('multiclass must be "ovr" or "ovo"')
            if sum([x[0] for x in fairness_loss_list]) == 0:
                scaled_fairness_loss = 0
            else:
                scaled_fairness_loss = sum([x[0] * x[1] for x in fairness_loss_list]) / sum([x[0] for x in fairness_loss_list])
        else:
            scaled_fairness_loss = 0
        L2_penalty = self.L2_gamma * sum([(p**2).sum() for p in model.parameters()])
        return F.binary_cross_entropy(y_pred, y_true) + scaled_fairness_loss + L2_penalty

def fairness_metrics(y_true, y_pred, sensitive_feature, X):
    res_dict = {}
    res_dict['demographic_parity_difference'] = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['demographic_parity_ratio'] = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['equalized_odds_difference'] = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['equalized_odds_ratio'] = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
    res_dict['consistency'] = consistency_score(X, y_pred, n_neighbors=5)
    res_dict['generalized_entropy_error'] = generalized_entropy_error(y_true, y_pred)
    return res_dict
