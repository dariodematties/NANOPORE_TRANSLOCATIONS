"""
Backbone module
"""

import sys
import torch
import torch.nn as nn
sys.path.append('../../../')
from Model_Util import Identity

class Backbone(nn.Module):
    def __init__(self, pulse_counter, feature_predictor, num_channels):
        super(Backbone, self).__init__()
        self.pulse_counter = pulse_counter
        self.pulse_counter.eval()
        self.feature_predictor = feature_predictor
        self.feature_predictor.linear1 = Identity()
        self.feature_predictor.linear2 = Identity()
        self.feature_predictor.train()
        self.num_channels = num_channels

    def forward(self, x):
        num_of_pulses = self.pulse_counter(x)
        return self.feature_predictor(x, torch.reshape(num_of_pulses, [num_of_pulses.shape[0],1]).round())


def build_backbone(pulse_counter, feature_predictor, num_channels):
    return Backbone(pulse_counter=pulse_counter, feature_predictor=feature_predictor, num_channels=num_channels)
