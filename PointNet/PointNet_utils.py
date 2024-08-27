import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class transform(nn.Module):
    def __init__(self, out = (3, 3)):
        super(transform, self).__init__()
        """
        input transform (3, 3) or feature transform (64, 64)
        input size: bs x C x N
        output size:
            input transform: bs x 3 x 3 
            feature transform: bs x 64 x 64

        """
        self.out_shape = out
        self.conv1 = nn.Conv1d(out[0], 64, 1, padding = 'valid')
        self.conv2 = nn.Conv1d(64, 128, 1, padding = 'valid')
        self.conv3 = nn.Conv1d(128, 1024, 1, padding = 'valid')
        
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256, np.prod(out), bias =False))
        self.bias = nn.Parameter(torch.eye(out[0], out[1]).view(-1),)
        self.bias.to(device)
        
    def forward(self, x):
        bs = x.shape[0]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out, _ = torch.max(out, -1)

        out = self.fc(out.view(bs, -1))
        out = out + self.bias
        return out.reshape(bs, *self.out_shape)
    

def get_loss(pred, target, feat = None, reg_weight=0.001):
    loss_keys = ['cls loss', 'reg loss', 'total loss']
    cls_loss = nn.CrossEntropyLoss()(pred, target)
    if feat:
        reg_loss = nn.MSELoss()(torch.eye(feat.shape[1],).to(device), 
                                torch.matmul(feat, feat.permute(0, 2, 1)))
    else:
        reg_loss = torch.zeros(1).to(device)
    total_loss = cls_loss + reg_weight * reg_loss
    with open('losses.pkl', 'wb') as f:
        pickle.dump({loss_keys[0]: cls_loss, loss_keys[1]: 
                     reg_loss, loss_keys[2]: total_loss}, f)
    return total_loss

def get_partseg_loss(l_pred, s_pred, l_target, s_target, feat = None, reg_weight=0.001, weight = 1):
    loss_keys = ['cls loss', 'seg loss', 'reg loss', 'total loss']
    bs, npart, pn = s_pred.shape
    cls_loss = nn.CrossEntropyLoss()(l_pred, l_target.to(torch.int64))
    seg_loss = nn.CrossEntropyLoss()(s_pred, s_target.to(torch.int64))

    if feat is not None:
        reg_loss = nn.MSELoss()(torch.eye(feat.shape[1],).to(device), 
                                torch.matmul(feat, feat.permute(0, 2, 1)))
    else:
        reg_loss = torch.zeros(1).to(device)
    total_loss = (1 - weight) * cls_loss + weight * seg_loss + reg_weight * reg_loss
    with open('all_losses.pkl', 'ab') as f:
        pickle.dump({loss_keys[0]: cls_loss.cpu().item(), 
                     loss_keys[1]: seg_loss.cpu().item(),
                     loss_keys[2]: 
                     reg_loss.cpu().item(), loss_keys[3]: total_loss.cpu().item()}, f)
    return total_loss, cls_loss, seg_loss
    

class mlp_layers_compile(nn.Module):
    def __init__(self, layers, dropout=0.0, batch_norm = True):
        super().__init__() 
        self.mlp_layer_list = nn.ModuleList()
        for i in range(len(layers)-1):
            self.mlp_layer_list.append(nn.Conv1d(layers[i], layers[i+1], 1, padding = 'valid'))
            if batch_norm or i < len(layers) - 2:
                self.mlp_layer_list.append(nn.BatchNorm1d(layers[i+1]))
                self.mlp_layer_list.append(nn.ReLU())
            if dropout > 0:
                self.mlp_layer_list.append(nn.Dropout(dropout))

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.mlp_layer_list):
            x = layer(x)
            if isinstance(layer, nn.Dropout):
                outputs.append(x)
            elif isinstance(layer, nn.ReLU):
                outputs.append(x)
            elif i == len(self.mlp_layer_list) -1:
                outputs.append(x)
        return outputs 


class cls_mlp(nn.Module):
    def __init__(self, layers, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()  # Using ModuleList to store layers
        for i in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.BatchNorm1d(layers[i+1]))
            self.layers.append(nn.ReLU())
        if dropout:
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Dropout):
                outputs.append(x)
            elif isinstance(layer, nn.ReLU):
                outputs.append(x)
            elif i == len(self.layers) -1:
                outputs.append(x)
        return outputs  # Return all outputs


def eval_iou_accuracy():
    pred_data_label_filenames = [line.rstrip() for line in open('all_pred_data_label_filelist.txt')]
    gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]
    num_room = len(gt_label_filenames)


    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]
    for i in range(num_room):
        print(i)
        data_label = np.loadtxt(pred_data_label_filenames[i])
        pred_label = data_label[:,-1]
        gt_label = np.loadtxt(gt_label_filenames[i])
        print(gt_label.shape)
        for j in range(gt_label.shape[0]):
            gt_l = int(gt_label[j])
            pred_l = int(pred_label[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l==pred_l)


    print(gt_classes)
    print(positive_classes)
    print(true_positive_classes)


    print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

    print('IoU:')
    iou_list = []
    for i in range(13):
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
        print(iou)
        iou_list.append(iou)

    print(sum(iou_list)/13.0)

def bn_momentum(bn_init_decay, bn_decay_rate, bn_decay_step, global_step, model):
    new_momentum = bn_init_decay * (bn_decay_rate ** (global_step // bn_decay_step))
    new_momentum = max(new_momentum, 0.01)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.momentum = new_momentum



# checkpoint
def save_checkpoint(model, optimizer, epoch, args, ):
    checkpoint_path = f'{args.output_dir}/trained_{args.label}_checkpoint_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'lr_scheduler_state_dict': lr_scheduler.state.dict(),
    }, checkpoint_path)

    
def load_checkpoint(model, optimizer, args,):
    checkpoint_path = f'{args.output_dir}/trained_{args.label}_checkpoint_{args.cp}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'], )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], )
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler.state.dict'],)
        epoch = checkpoint['epoch']
        print(f"Checkpoint found. Resuming training from epoch {epoch}.")
        return model, optimizer, epoch
    else:
        return model, optimizer, 0


def initialize_weights(m):
    if any([isinstance(m, nn.Conv2d), isinstance(m, nn.Linear), isinstance(m, nn.Conv1d)]):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
