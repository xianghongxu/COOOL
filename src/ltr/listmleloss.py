import torch
import torch.nn as nn
'''
refered
https://github.com/allegro/allRank/blob/2923985135c8afcdae6392e5b810b62da7851276/allrank/models/losses/listMLE.py#L1
'''

EPS = 1e-15
PADDED_Y_VALUE = -1.0

def listMLE(y_pred, y_true, eps=EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    max_pred_values, _ = y_pred.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = y_pred - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums+EPS) - preds_sorted_by_true_minus_max
    #print("observation:", observation_loss)
    #print("mask:", mask)
    observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1))

def listMLE_legacy(y_pred, y_true, eps=EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))