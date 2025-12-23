from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
from matplotlib import pyplot as plt
from torch.nn import functional as F
from model.ResUNet import ResUNet_Encoder, ResUNet_Decoder
# from postprocessing import visual
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
from model.EGM import EGAB
from model.DSSM import DAGB
from model.PositiveAttention import ChannelAttention3D, MultiChannelSpatialAttention
from model.DEM import DEAB


def concat(*args):
    return torch.cat(args, dim=1)


class ResUNet_EDGE_Encoder(ResUNet_Encoder):
    def __init__(self,
                 in_channels,
                 num_channels=[8, 16, 32, 64, 128],
                 window_size=(8, 8, 4),
                 conv_bias=False,
                 dropout=0.3,
                 att_dropout=0.25):
        super(ResUNet_EDGE_Encoder, self).__init__(in_channels, num_channels, conv_bias, dropout)
        # self.dem1_ct = SDM(num_channels[0],num_channels[0])

        self.egm1_ct = EGAB(num_channels[0])
        self.egm2_ct = EGAB(num_channels[1])
        self.egm3_ct = EGAB(num_channels[2])
        self.egm4_ct = EGAB(num_channels[3])
        self.egm5_ct = EGAB(num_channels[4])

        self.egm1_pet = EGAB(num_channels[0])
        self.egm2_pet = EGAB(num_channels[1])
        self.egm3_pet = EGAB(num_channels[2])
        self.egm4_pet = EGAB(num_channels[3])
        self.egm5_pet = EGAB(num_channels[4])

        self.dem2_ct = DAGB(num_channels[1], num_channels[1])
        self.dem3_ct = DAGB(num_channels[2], num_channels[2])
        self.dem4_ct = DAGB(num_channels[3], num_channels[3])
        self.dem5_ct = DAGB(num_channels[4], num_channels[4])

        # self.dem1_pet = SDM(num_channels[0],num_channels[0])
        self.dem2_pet = DAGB(num_channels[1], num_channels[1])
        self.dem3_pet = DAGB(num_channels[2], num_channels[2])
        self.dem4_pet = DAGB(num_channels[3], num_channels[3])
        self.dem5_pet = DAGB(num_channels[4], num_channels[4])


        self.mlp5_ct = nn.Linear(in_features=num_channels[4] * 2, out_features=num_channels[4])
        self.mlp4_ct = nn.Linear(in_features=num_channels[3] * 2, out_features=num_channels[3])
        self.mlp3_ct = nn.Linear(in_features=num_channels[2] * 2, out_features=num_channels[2])
        self.mlp2_ct = nn.Linear(in_features=num_channels[1] * 2, out_features=num_channels[1])
        self.mlp1_ct = nn.Linear(in_features=num_channels[0] * 2, out_features=num_channels[0])

        self.mlp5_pet = nn.Linear(in_features=num_channels[4] * 2, out_features=num_channels[4])
        self.mlp4_pet = nn.Linear(in_features=num_channels[3] * 2, out_features=num_channels[3])
        self.mlp3_pet = nn.Linear(in_features=num_channels[2] * 2, out_features=num_channels[2])
        self.mlp2_pet = nn.Linear(in_features=num_channels[1] * 2, out_features=num_channels[1])
        self.mlp1_pet = nn.Linear(in_features=num_channels[0] * 2, out_features=num_channels[0])

        # self.ln1_ct = nn.LayerNorm(num_channels[0]*2)
        # self.ln2_ct = nn.LayerNorm(num_channels[1]*2)
        # self.ln3_ct = nn.LayerNorm(num_channels[2]*2)
        # self.ln4_ct = nn.LayerNorm(num_channels[3]*2)
        # self.ln5_ct = nn.LayerNorm(num_channels[4]*2)
        # self.ln1_pet = nn.LayerNorm(num_channels[0]*2)
        # self.ln2_pet = nn.LayerNorm(num_channels[1]*2)
        # self.ln3_pet = nn.LayerNorm(num_channels[2]*2)
        # self.ln4_pet = nn.LayerNorm(num_channels[3]*2)
        # self.ln5_pet = nn.LayerNorm(num_channels[4]*2)
        # self.bn2_ct = nn.BatchNorm3d(num_channels[1],track_running_stats=False)
        # self.bn3_ct = nn.BatchNorm3d(num_channels[2],track_running_stats=False)
        # self.bn4_ct = nn.BatchNorm3d(num_channels[3],track_running_stats=False)
        # self.bn5_ct = nn.BatchNorm3d(num_channels[4],track_running_stats=False)

        # self.bn1_pet = nn.BatchNorm3d(num_channels[0],track_running_stats=False)
        # self.bn2_pet = nn.BatchNorm3d(num_channels[1],track_running_stats=False)
        # self.bn3_pet = nn.BatchNorm3d(num_channels[2],track_running_stats=False)
        # self.bn4_pet = nn.BatchNorm3d(num_channels[3],track_running_stats=False)
        # self.bn5_pet = nn.BatchNorm3d(num_channels[4],track_running_stats=False)

        self.lka1 = DEAB(num_channels[0])
        self.lka2 = DEAB(num_channels[1])
        self.lka3 = DEAB(num_channels[2])
        self.lka4 = DEAB(num_channels[3])
        self.lka5 = DEAB(num_channels[4])


    def forward(self, ct, pet):
        # Encode
        # enc1_ct = self.sa_multi0(ct)
        enc1_ct = self.conv1_ct(ct)

        # enc1_pet = self.sa_multi0(pet)
        enc1_pet = self.conv1_pet(pet)
        global enc2_ct, enc2_pet
        # enc2_ct = self.sa_multi1(enc1_ct)
        enc2_ct = self.conv2_ct(enc1_ct)
        # enc2_pet = self.sa_multi1(enc1_pet)
        enc2_pet = self.conv2_pet(enc1_pet)

        enc2_ct = self.dem2_ct(enc2_ct, enc2_pet)
        enc2_pet = self.dem2_pet(enc2_pet, enc2_ct)

        # enc2_ct = self.ca2(enc2_ct)
        # enc2_pet = self.ca2(enc2_pet)

        # enc3_ct = self.sa_multi2(enc2_ct)
        enc3_ct = self.conv3_ct(enc2_ct)
        # enc3_pet = self.sa_multi2(enc2_pet)
        enc3_pet = self.conv3_pet(enc2_pet)
        enc3_ct = self.dem3_ct(enc3_ct, enc3_pet)
        enc3_pet = self.dem3_pet(enc3_pet, enc3_ct)

        # enc3_ct = self.ca3(enc3_ct)
        # enc3_pet = self.ca3(enc3_pet)

        # enc4_ct = self.sa_multi3(enc3_ct)
        enc4_ct = self.conv4_ct(enc3_ct)
        # enc4_pet = self.sa_multi3(enc3_pet)
        enc4_pet = self.conv4_pet(enc3_pet)
        enc4_ct = self.dem4_ct(enc4_ct, enc4_pet)
        enc4_pet = self.dem4_pet(enc4_pet, enc4_ct)

        # enc4_ct = self.ca4(enc4_ct)
        # enc4_pet = self.ca4(enc4_pet)

        # enc5_ct = self.sa_multi4(enc4_ct)
        enc5_ct = self.conv5_ct(enc4_ct)
        # enc5_pet = self.sa_multi4(enc4_pet)
        enc5_pet = self.conv5_pet(enc4_pet)
        enc5_ct = self.dem5_ct(enc5_ct, enc5_pet)
        enc5_pet = self.dem5_pet(enc5_pet, enc5_ct)

        # enc5_ct = self.ca5(enc5_ct)
        # enc5_pet = self.ca5(enc5_pet)

        # enc5_ct = self.sa_multi5(enc5_ct)
        # enc5_pet = self.sa_multi5(enc5_pet)

        # print("enc1_ct shape:", enc1_ct.shape)

        # enc1_ct = self.cbam1(enc1_ct)  # 应用CBAM
        # enc2_ct = self.cbam2(enc2_ct)
        # enc3_ct = self.cbam3(enc3_ct)
        # enc4_ct = self.cbam4(enc4_ct)
        # enc5_ct = self.cbam5(enc5_ct)
        #
        # enc1_pet = self.cbam1(enc1_pet)
        # enc2_pet = self.cbam2(enc2_pet)
        # enc3_pet = self.cbam3(enc3_pet)
        # enc4_pet = self.cbam4(enc4_pet)
        # enc5_pet = self.cbam5(enc5_pet)

        edge1_ct = self.conv1_ct_edge(ct)
        edge1_pet = self.conv1_pet_edge(pet)
        edge1_ct = self.egm1_ct(edge1_ct)
        edge1_pet = self.egm1_pet(edge1_pet)

        edge2_ct = self.conv2_ct_edge(enc1_ct)
        edge2_pet = self.conv2_pet_edge(enc1_pet)
        edge2_ct = self.egm2_ct(edge2_ct)
        edge2_pet = self.egm2_pet(edge2_pet)

        edge3_ct = self.conv3_ct_edge(enc2_ct)
        edge3_pet = self.conv3_pet_edge(enc2_pet)
        edge3_ct = self.egm3_ct(edge3_ct)
        edge3_pet = self.egm3_pet(edge3_pet)

        edge4_ct = self.conv4_ct_edge(enc3_ct)
        edge4_pet = self.conv4_pet_edge(enc3_pet)
        edge4_ct = self.egm4_ct(edge4_ct)
        edge4_pet = self.egm4_pet(edge4_pet)

        edge5_ct = self.conv5_ct_edge(enc4_ct)
        edge5_pet = self.conv5_pet_edge(enc4_pet)
        edge5_ct = self.egm5_ct(edge5_ct)
        edge5_pet = self.egm5_pet(edge5_pet)

        # enc1_ct = enc1_ct + edge1_ct
        # enc1_pet = enc1_pet + edge1_pet
        # enc2_ct = enc2_ct + edge2_ct
        # enc2_pet = enc2_pet + edge2_pet
        # enc3_ct = enc3_ct + edge3_ct
        # enc3_pet = enc3_pet + edge3_pet
        # enc4_ct = enc4_ct + edge4_ct
        # enc4_pet = enc4_pet + edge4_pet
        # enc5_ct = enc5_ct + edge5_ct
        # enc5_pet = enc5_pet + edge5_pet

        N, C, H, W, D = enc1_ct.shape
        x=enc1_ct
        enc1_ct = concat(enc1_ct,edge1_ct)
        enc1_ct = enc1_ct.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[0] * 2)
        # enc1_ct = self.ln1_ct(enc1_ct)
        enc1_ct = self.mlp1_ct(enc1_ct)
        enc1_ct = enc1_ct.reshape(N, H, W, D, self.num_channels[0]).permute(0, 4, 1, 2, 3)
        enc1_ct += x
        # enc1_ct = self.activation(enc1_ct)
        x=enc1_pet
        enc1_pet = concat(enc1_pet, edge1_pet)
        enc1_pet = enc1_pet.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[0] * 2)
        # enc1_pet = self.ln1_pet(enc1_pet)
        enc1_pet = self.mlp1_pet(enc1_pet)
        enc1_pet = enc1_pet.reshape(N, H, W, D, self.num_channels[0]).permute(0, 4, 1, 2, 3)
        enc1_pet += x
        # enc1_pet = self.activation(enc1_pet)

        N, C, H, W, D = enc2_ct.shape
        x=enc2_ct
        enc2_ct = concat(enc2_ct,edge2_ct)
        enc2_ct = enc2_ct.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[1] * 2)
        # enc2_ct = self.ln2_ct(enc2_ct)
        enc2_ct = self.mlp2_ct(enc2_ct)
        enc2_ct = enc2_ct.reshape(N, H, W, D, self.num_channels[1]).permute(0, 4, 1, 2, 3)
        enc2_ct += x
        # enc2_ct = self.activation(enc2_ct)
        x=enc2_pet
        enc2_pet = concat(enc2_pet, edge2_pet)
        enc2_pet = enc2_pet.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[1] * 2)
        # enc2_pet = self.ln2_pet(enc2_pet)
        enc2_pet = self.mlp2_pet(enc2_pet)
        enc2_pet = enc2_pet.reshape(N, H, W, D, self.num_channels[1]).permute(0, 4, 1, 2, 3)
        enc2_pet += x
        # enc2_pet = self.activation(enc2_pet)
        N, C, H, W, D = enc3_ct.shape
        x=enc3_ct
        enc3_ct = concat(enc3_ct,edge3_ct)
        enc3_ct = enc3_ct.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[2] * 2)
        # enc3_ct = self.ln3_ct(enc3_ct)
        enc3_ct = self.mlp3_ct(enc3_ct)
        enc3_ct = enc3_ct.reshape(N, H, W, D, self.num_channels[2]).permute(0, 4, 1, 2, 3)
        enc3_ct += x
        # enc3_ct = self.activation(enc3_ct)
        x=enc3_pet
        enc3_pet = concat(enc3_pet, edge3_pet)
        enc3_pet = enc3_pet.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[2] * 2)
        # enc3_pet = self.ln3_pet(enc3_pet)
        enc3_pet = self.mlp3_pet(enc3_pet)
        enc3_pet = enc3_pet.reshape(N, H, W, D, self.num_channels[2]).permute(0, 4, 1, 2, 3)
        enc3_pet += x
        # enc3_pet = self.activation(enc3_pet)
        N, C, H, W, D = enc4_ct.shape
        x=enc4_ct
        enc4_ct = concat(enc4_ct,edge4_ct)
        enc4_ct = enc4_ct.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[3] * 2)
        # enc4_ct = self.ln4_ct(enc4_ct)
        enc4_ct = self.mlp4_ct(enc4_ct)
        enc4_ct = enc4_ct.reshape(N, H, W, D, self.num_channels[3]).permute(0, 4, 1, 2, 3)
        enc4_ct += x
        # enc4_ct = self.activation(enc4_ct)
        x=enc4_pet
        enc4_pet = concat(enc4_pet, edge4_pet)
        enc4_pet = enc4_pet.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[3] * 2)
        # enc4_pet = self.ln4_pet(enc4_pet)
        enc4_pet = self.mlp4_pet(enc4_pet)
        enc4_pet = enc4_pet.reshape(N, H, W, D, self.num_channels[3]).permute(0, 4, 1, 2, 3)
        enc4_pet += x
        # enc4_pet = self.activation(enc4_pet)
        N, C, H, W, D = enc5_ct.shape
        x=enc5_ct
        enc5_ct = concat(enc5_ct,edge5_ct)
        enc5_ct = enc5_ct.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[4] * 2)
        # enc5_ct = self.ln5_ct(enc5_ct)
        enc5_ct = self.mlp5_ct(enc5_ct)
        enc5_ct = enc5_ct.reshape(N, H, W, D, self.num_channels[4]).permute(0, 4, 1, 2, 3)
        enc5_ct += x
        # enc5_ct = self.activation(enc5_ct)
        x=enc5_pet
        enc5_pet = concat(enc5_pet, edge5_pet)
        enc5_pet = enc5_pet.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels[4] * 2)
        # enc5_pet = self.ln5_pet(enc5_pet)
        enc5_pet = self.mlp5_pet(enc5_pet)
        enc5_pet = enc5_pet.reshape(N, H, W, D, self.num_channels[4]).permute(0, 4, 1, 2, 3)
        enc5_pet += x
        # enc5_pet = self.activation(enc5_pet)

        enc1_pet, enc1_ct = self.lka1(enc1_pet, enc1_ct)
        enc2_pet, enc2_ct = self.lka2(enc2_pet, enc2_ct)
        enc3_pet, enc3_ct = self.lka3(enc3_pet, enc3_ct)
        enc4_pet, enc4_ct = self.lka4(enc4_pet, enc4_ct)
        enc5_pet, enc5_ct = self.lka5(enc5_pet, enc5_ct)


        return enc1_ct, enc2_ct, enc3_ct, enc4_ct, enc5_ct, enc1_pet, enc2_pet, enc3_pet, enc4_pet, enc5_pet
# def ri():
#     return enc2_ct, enc2_pet
