import numpy as np
import h5py
import torch
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data')
ply_data_dir = os.path.join(BASE_DIR, 'PartAnnotation')
test_file_list = os.path.join(BASE_DIR, 'testing_ply_file_list.txt')


oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
cpid2oid = json.load(open(os.path.join(hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))


def shuffle_data(data, labels):
    """ Shuffle data and labels
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def getTestDataFiles(list_filename):
    ffiles = open(list_filename, 'r')
    lines = [line.rstrip() for line in ffiles.readlines()]
    pts_files = [line.split()[0] for line in lines]
    seg_files = [line.split()[1] for line in lines]
    labels = [line.split()[2] for line in lines]
    ffiles.close()
    return pts_files, seg_files, labels

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


class partseg_dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, seg):
        self.data = data
        self.label = label
        self.seg = seg

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 
                'labels': self.label[idx], 
                'segments': self.seg[idx]}
    
    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return torch.utils.data.ConcatDataset([self, other])
    
def load_data(hdf5_data_dir, train_file_list):
    # lens = [0]
    for i in range(len(train_file_list)):
        cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[i])
        cur_data, cur_labels, cur_seg = loadDataFile_with_seg(cur_train_filename)
        cur_data, cur_labels, order = shuffle_data(cur_data, np.squeeze(cur_labels))
        cur_seg = cur_seg[order, ...]
        ds_ = partseg_dataset(data = cur_data, label = cur_labels, seg = cur_seg)
        # lens += [ds_.__len__()]

        try:
            ds.__add__(ds_)
        except:
            ds = partseg_dataset(data = cur_data, label = cur_labels, seg = cur_seg)
    # lens = np.cumsum(lens)
    return ds

def load_data_(hdf5_data_dir, train_file):
    # lens = [0]
    cur_train_filename = os.path.join(hdf5_data_dir, train_file)
    cur_data, cur_labels, cur_seg = loadDataFile_with_seg(cur_train_filename)
    cur_data, cur_labels, order = shuffle_data(cur_data, np.squeeze(cur_labels))
    cur_data[:, :3] = pc_normalize(cur_data[:, :3])
    cur_seg = cur_seg[order, ...]
    ds = partseg_dataset(data = cur_data, label = cur_labels, seg = cur_seg)
    return ds

def prepare_test_data(hdf5_data_dir, ):
    oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

    object2setofoid = {}
    for idx in range(len(oid2cpid)):
        objid, pid = oid2cpid[idx]
        if not objid in object2setofoid.keys():
            object2setofoid[objid] = []
        object2setofoid[objid].append(idx)

    all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
    fin = open(all_obj_cat_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    objcats = [line.split()[1] for line in lines]
    objnames = [line.split()[0] for line in lines]
    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()

    color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
    color_map = json.load(open(color_map_file, 'r'))

    cpid2oid = json.load(open(os.path.join(hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

    return object2setofoid, objcats, objnames, on2oid, color_map, cpid2oid


def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def load_pts_seg_files(pts_file, seg_file, catid):
    with open(pts_file, 'r') as f:
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
    with open(seg_file, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
        seg = np.array([cpid2oid[catid+'_'+str(x)] for x in part_ids])
    return pts, seg

def pc_augment_to_point_num(pts, pn):
    assert(pts.shape[0] <= pn)
    cur_len = pts.shape[0]
    res = pts
    while cur_len < pn:
        res = torch.cat((res, pts), 0)
        cur_len += pts.shape[0]
    return res[:pn, :]

class partseg_testset(torch.utils.data.Dataset):
    def __init__(self, data_file, seg_file, label, on2oid, objcats, ply_data_dir):
        self.data_file = data_file
        self.label = label
        self.seg_file = seg_file
        self.on2oid = on2oid
        self.objcats = objcats
        self.ply_data_dir = ply_data_dir

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        cur_gt_label = self.on2oid[self.label[idx]]
        pts_file_to_load = os.path.join(self.ply_data_dir, self.data_file[idx])
        seg_file_to_load = os.path.join(self.ply_data_dir, self.seg_file[idx])
        
        pts, segs = load_pts_seg_files(pts_file_to_load, seg_file_to_load,
                                        self.objcats[cur_gt_label])
        return {'data': torch.from_numpy(pts).float(), 'segments': torch.from_numpy(segs).long(), 
                'labels': torch.tensor(cur_gt_label).long()}

    
class partseg_collator:
    def __init__(self, max_length):
        self.max_length = max_length
    
    def __call__(self, batch):
        batched_data = [item['data'] for item in batch]
        batched_seg = [item['segments'] for item in batch]
        batched_label = [item['labels'] for item in batch]
        
        padded_data = []
        for data in batched_data:
            if data.shape[0] < self.max_length:
                padded_data.append(pc_augment_to_point_num(data, self.max_length))
            else:
                padded_data.append(data)
        
        # Stack all data and segment tensors
        batched_data_tensor = torch.stack(padded_data,)
        batched_seg = torch.stack(batched_seg, )
        batched_label = torch.stack(batched_label, )

        
        
        return {'data': batched_data_tensor,  
               'segments': batched_seg, 
               'labels': batched_label}
    
def iou(preds, targets, iou_oids):
    mask = preds.eq(targets)
    total_iou = 0 
    for oid in iou_oids:
        n_pred = preds.eq(oid).sum().cpu().item()
        n_target = targets.eq(oid).sum().cpu().item()
        n_intersect = (targets.eq(oid) * mask).sum().cpu().item()
        n_union = n_pred + n_target - n_intersect
        total_iou = total_iou + n_intersect/n_union if n_union else total_iou + 1
    avg_iou = total_iou/len(iou_oids)
    return avg_iou
