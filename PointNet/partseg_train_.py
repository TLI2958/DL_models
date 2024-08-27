import argparse
import torch 
import torch.nn as nn
import logging
from tqdm import tqdm
from datetime import datetime
import json
import os
import sys
from IPython.display import HTML

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from PointNet_data import * 
from PointNet import *

def train(model, optimizer, train_dataloader):
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, 
                                                                  args)

    model.to(device)

    pbar = tqdm(range(max(1, start_epoch + 1), args.epoch + 1))
    
    for epoch in pbar:
        total_loss = 0.0
        total_label_loss = 0.0
        total_seg_loss = 0.0
        total_label_acc = 0.0
        total_seg_acc = 0.0
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), 
                            total = len(train_dataloader)):
            data, label, seg = batch.values()
            data, label, seg = data.to(device), label.to(device), seg.to(device)
            labels_pred, seg_pred, end_points = model(data, label)
            loss, cls_loss, seg_loss = get_partseg_loss(l_pred = labels_pred, s_pred = seg_pred,
                                                        l_target = label, s_target = seg, feat = end_points, 
                                                        reg_weight=0.001, weight = .999)
            total_loss += loss.detach().to('cpu').item()
            total_label_loss += cls_loss.detach().to('cpu').item()
            total_seg_loss += seg_loss.detach().to('cpu').item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # acc
            bs, npart, pn = seg_pred.shape
            labels_pred_, seg_pred_ = torch.argmax(labels_pred, 1), seg_pred.argmax(1)

            per_instance_seg_acc = seg_pred_.eq(seg.to(torch.int64)).float().mean(dim = 1)
            seg_acc = per_instance_seg_acc.mean()
            
            total_label_acc += (labels_pred_.eq(label)).float().mean().cpu().item()
            total_seg_acc += seg_acc.cpu().item()

        avg_total_loss, avg_label_loss, avg_seg_loss = total_loss / len(train_dataloader), total_label_loss / len(train_dataloader), total_seg_loss / len(train_dataloader)
        avg_label_acc, avg_seg_acc = total_label_acc / len(train_dataloader), total_seg_acc / len(train_dataloader)
            
        pbar.set_postfix({
            'train_loss': avg_total_loss,
            'label_loss': avg_label_loss,
            'seg_loss': avg_seg_loss,
            'label_acc': avg_label_acc,
            'seg_acc': avg_seg_acc})
        
        with open(args.output_dir + f'/train.log.pkl', 'a') as f:
            f.write(f'\nepoch: {epoch-1}\ttotal loss: {avg_total_loss:.6f}\tlabel loss: {avg_label_loss:.6f}\tseg loss: {avg_seg_loss:.6f}\nlabel acc: {avg_label_acc:.6f}\tseg acc: {avg_seg_acc:.6f}\n')

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, args) 
            eval(model, train_dataloader, train_eval = True, epoch = epoch)
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(base_learning_rate *(decay_rate **(epoch//decay_step)), 
                                    learning_rate_clip)
        print(param_group['lr'])
        bn_momentum(bn_init_decay, bn_decay_rate, bn_decay_step, global_step = epoch, model = model)
                

            
def eval(model, eval_dataloader, train_eval = False, epoch = 0):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_label_loss = 0.0
    total_seg_loss = 0.0
    total_label_acc = 0.0
    total_seg_acc = 0.0
    
    total_label_acc_per_cat = np.zeros((num_cat)).astype(np.float32)
    total_seg_acc_per_cat = np.zeros((num_cat)).astype(np.float32)
    total_seen_per_cat = np.zeros((num_cat)).astype(np.int32)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader),
                            total = len(eval_dataloader)):
            data, label, seg = batch.values()
            data, label, seg = data.to(device), label.to(device), seg.to(device)
            labels_pred, seg_pred, end_points = model(data, label)
            loss, cls_loss, seg_loss = get_partseg_loss(l_pred = labels_pred, s_pred = seg_pred, 
                                                        l_target = label, s_target = seg, 
                                                        feat = end_points, reg_weight=0.001, weight = 0.999)
            # loss
            total_loss += loss.detach().to('cpu').item()
            total_label_loss += cls_loss.detach().to('cpu').item()
            total_seg_loss += seg_loss.detach().to('cpu').item()
            
            # acc
            bs, npart, pn = seg_pred.shape
            labels_pred_, seg_pred_ = labels_pred.argmax(1), seg_pred.argmax(1)
            
            per_instance_seg_acc = seg_pred_.eq(seg.to(torch.int64)).float().mean(dim = 1)
            seg_acc = per_instance_seg_acc.mean()
            
            total_label_acc += (labels_pred_.eq(label)).float().mean().cpu().item()
            total_seg_acc += seg_acc.cpu().item()

            total_seen_per_cat += np.bincount(label.cpu(), minlength = num_cat)
            for cat in range(num_cat):
                mask = (cat == label)
                if mask.any():
                    total_label_acc_per_cat[cat] += (labels_pred_[mask].eq(label[mask])).float().sum().cpu().item()          
                    total_seg_acc_per_cat[cat] += per_instance_seg_acc[mask].float().sum().cpu().item()
        avg_label_acc_per_cat = np.nan_to_num(total_label_acc_per_cat / total_seen_per_cat)
        avg_seg_acc_per_cat = np.nan_to_num(total_seg_acc_per_cat / total_seen_per_cat)

        avg_total_loss, avg_label_loss, avg_seg_loss = total_loss / len(eval_dataloader), total_label_loss / len(eval_dataloader), total_seg_loss / len(eval_dataloader)
        avg_label_acc, avg_seg_acc = total_label_acc / len(eval_dataloader), total_seg_acc / len(eval_dataloader)
        
        log_message = f'\nepoch: {max(epoch, args.cp)}\teval loss: {avg_total_loss:.6f}\tlabel loss: {avg_label_loss:.6f}\tseg loss: {avg_seg_loss:.6f}\nlabel acc: {avg_label_acc:.6f}\tseg acc: {avg_seg_acc:.6f}\n\nlabel_acc_per_cat: {avg_label_acc_per_cat}\n\nseg_acc_per_cat: {avg_seg_acc_per_cat}\n'
        log_message = '\n' + '='*80 + log_message

        with open(args.output_dir + '/eval.log.pkl', 'a') as f:
            f.write(log_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
    parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
    parser.add_argument('--epoch', type=int, default=50, help='Epoch to run [default: 50]')
    parser.add_argument('--cp', type=int, default=0, help='checkpoint to run [default: 0]')
    parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
    parser.add_argument('--output_dir', type=str, default='train_results/', help='Directory that stores all training logs and trained models')
    parser.add_argument('--label', type = str, default= 'partseg')
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--eval", action="store_true", help="eval a model on the val data")
    parser.add_argument("--debug", action="store_true", help="debug train")


    args = parser.parse_args()
    
    hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data/')

    num_cat = 16
    decay_step = 20
    decay_rate = 0.5
    
    learning_rate_clip = 1e-5
    bn_init_decay = 0.1
    bn_decay_rate = 0.5
    bn_decay_step = float(decay_step * 2)
    # bn_decay_step = 20
    bn_decay_clip = 0.99
    
    base_learning_rate = 1e-3
    training_epochs = args.epoch
    
    training_file_list = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
    testing_file_list = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')
    # inference_file_list = os.path.join(hdf5_data_dir, 'test_hdf5_file_list.txt')


    train_file_list = getDataFiles(training_file_list)
    num_train_file = len(train_file_list)
    test_file_list = getDataFiles(testing_file_list)
    num_test_file = len(test_file_list)
    # inf_file_list = getDataFiles(inference_file_list)
    # num_inf_file = len(inf_file_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare(args)
    
    model = part_seg()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate, 
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=1e-4)

    model.to(device)

    if args.train:
        train_dataset = load_data(hdf5_data_dir, train_file_list)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                       batch_size = args.batch, shuffle = True)
        
        bn_momentum(bn_init_decay, bn_decay_rate, bn_decay_step, 0, model)
        train(model, optimizer, train_dataloader)

    if args.debug:
        train_dataset = load_data(hdf5_data_dir, train_file_list)
        
        indices = torch.randint(len(train_dataset), (10,))
        
        # Use these indices to create a smaller dataset
        small_train_dataset = torch.utils.data.Subset(train_dataset, indices)
        small_train_dataloader = torch.utils.data.DataLoader(small_train_dataset, 
                                                       batch_size = args.batch, shuffle = True)

        bn_momentum(bn_init_decay, bn_decay_rate, bn_decay_step, 0, model)
        train(model, optimizer, small_train_dataloader,)

    if args.eval:
        eval_dataset = load_data(hdf5_data_dir, test_file_list)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, 
                                                batch_size = args.batch, shuffle = True)
        model, optimizer, epoch = load_checkpoint(model, optimizer, args)
        eval(model, eval_dataloader, )

