# Copyright 2023 Bytedance Ltd. and/or its affiliates 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import torch
import torch.nn as nn
def pairwiseMLE(y_pred, y_true, weights = None):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, 2]
    :param y_true: ground truth labels, shape [batch_size, 2]
    :return: loss value, a torch.Tensor
    """
    y_true_sorted, indices = y_true.sort(descending=True, dim=-1)
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums) - preds_sorted_by_true_minus_max
    if weights is not None:
        return torch.dot(weights, torch.sum(observation_loss, dim = 1))
    else:
        return torch.mean(torch.sum(observation_loss, dim = 1))

if __name__ == "__main__":
    y_pred = torch.tensor([[3, 2, 1, 1], [1, 2, 4, 2.5]])
    y_true = torch.tensor([[1, 0.5, -1, -1], [4, 3, 2, -1]])
    print(pairwiseMLE(y_pred, y_true))
