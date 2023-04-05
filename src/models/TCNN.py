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
'''
This file solely consists the Value Network model, i.e., all parameters.
A scalar <- ValueNetwork(plan)
copied from Bao
'''

from src.models.model_tools import *
import torch.nn as nn
from src.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from src.TreeConvolution.tcnn import TreeActivation, DynamicPooling
from src.TreeConvolution.util import prepare_trees

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    return x[0]

# main model, import_lib applicable
class TCNN(nn.Module):
    def __init__(self, in_channels):
        super(TCNN, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        )
        self.layer1 = nn.Linear(64, 32)
        self.activation = nn.LeakyReLU()
        self.layer2 = nn.Linear(32, 1)

    def in_channels(self):
        return self.__in_channels
        
    def forward(self, x):
        '''
        x: trees
        output: scalar
        '''
        trees = prepare_trees(x, features, left_child, right_child,
                              cuda=self.__cuda)
        plan_emb = self.tree_conv(trees)
        latent = self.layer1(plan_emb)
        latent = self.activation(latent)
        score = self.layer2(latent)
        return score

    def cuda(self):
        self.__cuda = True
        return super().cuda()



config = Dict({
    'Model': TCNN,
})