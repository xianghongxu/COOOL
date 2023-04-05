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
import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import copy
from torch.nn import functional
import torch.nn as nn






def clones(module, N):
    "生成N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Dict(dict):
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, val):
        return self.__setitem__(key, val)

    def __setitem__(self, key, val):
        if type(val) is dict:
            val = Dict(val)
        super().__setitem__(key, val)

    def __str__(self):
        import re

        _str = ''
        for key, val in sorted(self.items()):
            if type(key) is not str:
                continue
            val_str = str(val)
            if len(val_str) > 80:
                val_str = re.sub(r'\s+', ' ', val_str.strip())[:60]
                val_str = f'"{val_str}..."'
            _str += f"{key}: {val_str}\n"
        return _str