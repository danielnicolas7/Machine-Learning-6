import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
import csv
import itertools
import collections
import pywt
from scipy import stats
import pandas as pd
import os
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report

plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 

def evaluate_model(test_targets, test_preds, labels):
    label_indices = list(range(len(labels)))
    return classification_report(
        test_targets,
        test_preds,
        labels=label_indices,
        target_names=labels,
        zero_division=0
    )

def denoise(data): 
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.03
    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    return pywt.waverec(coeffs, 'sym4')

def add_ecg_noise(signal, noise_type='baseline_wander'):
    if noise_type == 'baseline_wander':
        noise = 0.2 * np.sin(2 * np.pi * 0.5 * np.arange(len(signal)))
    elif noise_type == 'electrode_motion':
        noise = 0.2 * np.random.randn(len(signal)) * np.sin(2 * np.pi * 1 * np.arange(len(signal)))
    return signal + noise

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ECGTransformer(nn.Module):
    def __init__(self, configs, hparams):
        super(ECGTransformer, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs['input_channels'], configs['mid_channels'], kernel_size=filter_sizes[0],
                               stride=configs['stride'], bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs['input_channels'], configs['mid_channels'], kernel_size=filter_sizes[1],
                               stride=configs['stride'], bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs['input_channels'], configs['mid_channels'], kernel_size=filter_sizes[2],
                               stride=configs['stride'], bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(0.2)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(128, 5)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)
        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))
        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)
        x = self.crm(x)
        x = x.permute(0, 2, 1)
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [2]))
        x = x1 + x2
        x = x.permute(0, 2, 1)
        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        return x_out

if __name__ == "__main__":
    # Seluruh kode pelatihan dan evaluasi diletakkan di sini
    pass
