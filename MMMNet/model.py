import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


# Define alignment model
class Alignment(nn.Module):
    def __init__(self):
        super(Alignment, self).__init__()
        self.text_alignment = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.pic_alignment = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )

    def forward(self, text, pic):
        text_out = self.text_alignment(text)
        pic_out = self.pic_alignment(pic)
        return text_out, pic_out


# Definition of cross modal fusion module
class CrossModule(nn.Module):
    def __init__(self):
        super(CrossModule, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.correlation_layer = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        # self.fusion_layer = nn.Sequential (
        #     nn.Linear(16, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        # )

    def forward(self, text, image):
        text_input = text.unsqueeze(2)
        image_input = image.unsqueeze(1)
        feature_dim = text.shape[1]
        similarity = torch.matmul(text_input, image_input) / math.sqrt(feature_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        fusion = self.correlation_layer(correlation_p)
        return fusion


# Define classifier model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fusion_module = CrossModule()
        # self.fc1 = nn.Linear(48, 32)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.fc2 = nn.Linear(32, 16)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.fc3 = nn.Linear(16, 1)
        self.text_uni = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.pic_uni = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(64, 20),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, text, image, t, p):
        fusion = self.fusion_module(text, image)
        # x = torch.cat([text, image, fusion], 1)
        t_uni = self.text_uni(t)
        p_uni = self.pic_uni(p)
        # x = torch.cat([p_uni, t_uni, fusion], 1)
        x = fusion
        x = self.classifier_1(x)

        # x = torch.cat([text, image], 1)
        # x = torch.sigmoid(self.bn1(self.fc1(x)))
        # x = torch.sigmoid(self.bn2(self.fc2(x)))
        # x = torch.sigmoid(self.fc3(x))
        return x

# Define classifier model with mask
class Classifier_M(nn.Module):
    def __init__(self):
        super(Classifier_M, self).__init__()
        self.fusion_module = CrossModule()
        # self.fc1 = nn.Linear(48, 32)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.fc2 = nn.Linear(32, 16)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.fc3 = nn.Linear(16, 1)
        self.text_uni = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.pic_uni = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(64, 20),
            nn.BatchNorm1d(20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, text, image, t, p, auto_mask):
        fusion = self.fusion_module(text, image)
        # x = torch.cat([text, image, fusion], 1)
        t_uni = self.text_uni(t)
        p_uni = self.pic_uni(p)
        # x = torch.cat([p_uni, t_uni, fusion], 1)
        x = fusion * auto_mask
        x = self.classifier_1(x)

        # x = torch.cat([text, image], 1)
        # x = torch.sigmoid(self.bn1(self.fc1(x)))
        # x = torch.sigmoid(self.bn2(self.fc2(x)))
        # x = torch.sigmoid(self.fc3(x))
        return x
