import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from PointNet_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointNet_base(nn.Module):
    """
    PointNet backbone
    add cls, seg head later
    """
    def __init__(self, mlp_layers = (64, 64, 128, 1024), feat_dim = 64, input_dim = 3,):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.mlp_layers = mlp_layers
        
        self.input_transform = transform((input_dim, input_dim))
        self.feature_transform = transform((feat_dim, feat_dim))

        self.conv1 = nn.Conv2d(1, 64, (1, 3), padding = 'valid')
        self.conv2 = nn.Conv2d(64, 64, 1, padding = 'valid')
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.mlp_layers = mlp_layers_compile(mlp_layers)

        
    def forward(self, x):
        input_transform = self.input_transform(x.contiguous().permute(0, 2, 1))
        x = torch.matmul(x, input_transform).unsqueeze(3) # bs x N x 3 x 1
        out = self.relu(self.bn1(self.conv1(x.contiguous().permute(0, 3, 1, 2)))) # bs x 1 x N x 3 -> bs x 64 x N x 1 
        out = self.relu(self.bn2(self.conv2(out))).squeeze(-1) # bs x 64 x N x 1 -> bs x 64 x N

        feat_transform = self.feature_transform(out) # bs x 64 x 64
        feat_out = torch.matmul(out.contiguous().permute(0, 2, 1), feat_transform) # bs x N x 64
        out = self.mlp_layers(feat_out.contiguous().permute(0, 2, 1))[-1] # bs x 64 x N -> bs x 1024 x N

        out = torch.max(out.contiguous().permute(0, 2, 1), 1)[0] # "maxpool", bs x N x 1024 -> bs x 1024
        return out, feat_transform, feat_out
    

class PointNet_cls(nn.Module):
    """
    input size: bs x N x C
    """
    def __init__(self, fc_layers = (1024, 512, 256, 40)):
        super(PointNet_cls, self).__init__()
        self.base = PointNet_base()
        # paper: only dropout before the last fc layer, 
        # imp in tf: dropout after each fc layer
        self.fc = cls_mlp(fc_layers)
    
    def forward(self, x):
        out, end_point, _ = self.base(x)
        return self.fc(out)[-1], end_point
    

class PointNet_seg(nn.Module):
    def __init__(self, mlp_layers = (1088, 512, 256, 128, 50)):
        super().__init__()
        self.base = PointNet_base()
        
        self.mlp_1 = mlp_layers_compile(mlp_layers[:-1])
        
        self.mlp_2 = nn.Sequential(nn.Conv1d(mlp_layers[-2], mlp_layers[-2], 1),
                                   nn.BatchNorm1d(mlp_layers[-2]),
                                   nn.ReLU(),
                                   nn.Conv1d(mlp_layers[-2], mlp_layers[-1], 1))
    def forward(self, x):
        out, end_point, feat_out = self.base(x)
        out = out.unsqueeze(2).expand(-1, -1, x.shape[1]) # bs x 1024 x N
        concat_x = torch.cat([out.permute(0, 2, 1), feat_out], dim = -1) # bs x N x (max_pool + 64)
        out = self.mlp_1(concat_x.contiguous().permute(0, 2, 1))[-1] # bs x N x 128
        out = self.mlp_2(out) # bs x N x 50
        return out, end_point
    


class part_seg(PointNet_base):
    def __init__(self, mlp_layers_one =(64, 128, 128), 
                 mlp_layers_two = (128, 512, 2048), 
                 cls_layers = (2048, 256, 256, 16), 
                 seg_layers = (256, 256, 128, 50), 
                 input_dim = 3, feat_dim = 128):
        super().__init__(mlp_layers= mlp_layers_two, feat_dim = feat_dim, input_dim = input_dim)

        self.conv1 = nn.Conv2d(1, mlp_layers_one[0], (1, input_dim), padding = 'valid')
        self.bn1 = nn.BatchNorm2d(mlp_layers_one[0])
        self.relu = nn.ReLU()

        self.mlp_layers_one = mlp_layers_compile(mlp_layers_one)
        self.mlp_layers_two = mlp_layers_compile(mlp_layers_two)

        self.cls_mlp = cls_mlp(cls_layers)
        seg_input_dim = np.sum(mlp_layers_one)
        seg_input_dim += np.sum(mlp_layers_two[1:])
        seg_input_dim += mlp_layers_two[-1]
        seg_input_dim += cls_layers[-1] 
        seg_layers = (seg_input_dim,) + seg_layers # out from all convs + max_pool + input_label_dim
        self.seg_layers = seg_layers
        self.seg_mlp_1 = mlp_layers_compile(seg_layers[:-2], dropout = .2)
        self.seg_mlp_2 = mlp_layers_compile(seg_layers[-3:], batch_norm=False)

    
    def forward(self, x, labels):
        _, N, _ =  x.shape
        all_outs = []
        input_transform = self.input_transform(x.contiguous().permute(0, 2, 1))
        x = torch.matmul(x, input_transform).unsqueeze(3)

        out = self.relu(self.bn1(self.conv1(x.contiguous().permute(0, 3, 1, 2)))).squeeze(-1)
        all_outs.append(out)
        all_outs_ = self.mlp_layers_one(out.contiguous()) 
        all_outs.extend(all_outs_ if isinstance(all_outs_, list) else [all_outs_])

        feat_transform = self.feature_transform(all_outs[-1]) # bs x 128 x 128
        feat_out = torch.matmul(all_outs[-1].contiguous().permute(0, 2, 1), feat_transform)
        all_outs_ = self.mlp_layers_two(feat_out.contiguous().permute(0, 2, 1)) 
        all_outs.extend(all_outs_ if isinstance(all_outs_, list) else [all_outs_])

        out = torch.max(all_outs[-1].contiguous().permute(0, 2, 1), 1)[0] # "maxpool", bs x N x 2048 -> bs x 2048
        cls_out = self.cls_mlp(out)[-1]

        seg_input = torch.cat([out.unsqueeze(2).expand(-1,-1, N), 
                               F.one_hot(labels.to(torch.int64), num_classes = 16).float().to(device).unsqueeze(2).expand(-1, -1, N), # bs x 16 x N
                               torch.cat(all_outs, dim = 1)], dim = 1) # bs x seg_input_dim x N

        seg_out = self.seg_mlp_1(seg_input)[-1]
        seg_out = self.seg_mlp_2(seg_out)[-1] # bs x seg_out x N
        
        return cls_out, seg_out, feat_transform



class sem_seg(PointNet_base):
    def __init__(self, mlp_layers =(64, 64, 64, 128, 1024), 
                 mlp_layers_two = (512, 256, 13),
                 fc_layers = (1024, 256, 128), 
                 input_dim = 9, feat_dim = 128):
        super().__init__(mlp_layers= mlp_layers, feat_dim = feat_dim, input_dim = input_dim)

        self.conv1 = nn.Conv2d(1, mlp_layers[0], (1, input_dim), padding = 'valid')
        self.bn1 = nn.BatchNorm2d(mlp_layers[0])
        self.relu = nn.ReLU()

        self.mlp_layers_one = mlp_layers_compile(mlp_layers)
        mlp_layers_two = (np.sum([mlp_layers[-1], fc_layers[-1]]),) + mlp_layers_two
        self.mlp_layers_two = mlp_layers_compile(mlp_layers_two[:-1])
        self.conv2 = nn.Conv1d(mlp_layers_two[-2], mlp_layers_two[-1], 1)

        self.fc = cls_mlp(fc_layers, dropout=0)
        self.dropout = nn.Dropout(.3)
    
    def forward(self, x,):
       _, N, _ = x.shape
       x = x.unsqueeze(3)
       out = self.relu(self.bn1(self.conv1(x.contiguous().permute(0, 3, 1, 2)))).squeeze(-1)

       point_feat = self.mlp_layers_one(out.contiguous())[-1] # bs x 1024 x N

       pc_feat = torch.max(point_feat, -1)[0] 
       pc_feat = self.fc(pc_feat)[-1].unsqueeze(2).expand(-1, -1, N) # bs x 128 x N 
       points_pc = torch.cat([point_feat, pc_feat], dim = 1) # bs x 1152 x N

       out = self.dropout(self.mlp_layers_two(points_pc)[-1])
       out = self.conv2(out)
       return out
 
