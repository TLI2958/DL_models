import argparse
import torch 
import torch.nn as nn
import logging
from tqdm import tqdm
from datetime import datetime
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

from PointNet_data import * 
from PointNet import *

def predict(model, test_dataloader):
    model.eval()
    total_acc = 0
    total_acc_iou = 0

    total_per_cat_acc = np.zeros((num_cats)).astype(np.float32)
    total_per_cat_iou = np.zeros((num_cats)).astype(np.float32)
    total_per_cat_seen = np.zeros((num_cats)).astype(np.int32)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), 
                             total = len(test_dataloader)):
            data, seg, label = batch.values()
            data, seg, label = data.to(device), seg.to(device), label.to(device)
            labels_pred, seg_pred, end_points = model(data, label)
            
            
            iou_oids = object2setofoid[objcats[label]]
            non_cat_labels = list(set(np.arange(num_parts)).difference(set(iou_oids)))
            seg_pred[:, non_cat_labels] = -1e9
            seg_pred_ = seg_pred[:, :, :seg.shape[1]].argmax(1)
            seg_acc = seg_pred_.eq(seg.to(torch.int64)).float().mean(-1).cpu().item()
            total_acc += seg_acc

            total_per_cat_seen[label] += 1
            total_per_cat_acc[label] += seg_acc

            avg_iou = iou(seg_pred_, seg, iou_oids)
            total_per_cat_iou[label] += avg_iou 
            total_acc_iou += avg_iou
        
        avg_acc = total_acc/len(test_dataloader)
        avg_iou = total_acc_iou/len(test_dataloader)
        log_message = '='*50
        log_message += f'\nEpoch:\t{args.cp}\nAccuracy:\t{avg_acc:.6f}\nIoU:\t{avg_iou:.6f}\n'

        for cat in range(num_cats):
            log_message += f'\n{objcats[cat]}\t\tTotal Number:\t{str(total_per_cat_seen[cat])}\n'
            if total_per_cat_seen[cat] > 0:
                log_message += f'Accuracy:\t{total_per_cat_acc[cat] / total_per_cat_seen[cat]:.6f}\n'
                log_message += f'IoU\t{total_per_cat_iou[cat] / total_per_cat_seen[cat]:.5f}\n'
        
        with open(args.output_dir + 'test_metric.pkl', 'a') as f:
            f.write(log_message)

        print(log_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
    parser.add_argument('--batch', type=int, default=1, help='Batch Size during inferece [default: 1]')
    parser.add_argument('--cp', type=int, default=60, help='checkpoint to run [default: 0]')
    parser.add_argument('--output_dir', type=str, default='train_results/', help='Directory that stores all training logs and trained models')
    parser.add_argument('--label', type = str, default= 'partseg')
    parser.add_argument("--pred", action="store_true", help='Run prediction on test data')


    args = parser.parse_args()
    
    hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data/')
    ply_data_dir = os.path.join(BASE_DIR, 'PartAnnotation/')
    output_dir = os.path.join(BASE_DIR, 'test_results/')
    test_file_list = os.path.join(BASE_DIR, 'testing_ply_file_list.txt')


    point_num = 3000  # the max number of points in the all testing data shapes

    num_cats = 16
    num_parts = 50
    
    learning_rate_clip = 1e-6
    base_learning_rate = 1e-4
    decay_rate = 0.5

    momentum = 0.9

    object2setofoid, objcats, objnames, on2oid, color_map, cpid2oid = prepare_test_data(hdf5_data_dir)
    pts_files, seg_files, labels = getTestDataFiles(test_file_list)
    test_dataloader = torch.utils.data.DataLoader(partseg_testset(pts_files, 
                                                                  seg_files, labels, 
                                                                  on2oid, objcats, 
                                                                  ply_data_dir), batch_size = args.batch, 
                                                                  shuffle = True, 
                                                                  collate_fn = partseg_collator(point_num))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = part_seg()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate, 
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=1e-4)
    model, optimizer, epoch = load_checkpoint(model, optimizer, args)
    model.to(device)
    if args.pred:
        predict(model, test_dataloader)