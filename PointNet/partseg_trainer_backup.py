# based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.pys
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
import os
import time
import logging
import copy
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from detectron2.utils.events import get_event_storage, EventStorage, EventWriter, CommonMetricPrinter, JSONWriter
from detectron2.utils.file_io import PathManager
from detectron2.engine.train_loop import SimpleTrainer
import detectron2.utils.comm as comm
from fvcore.common.checkpoint import Checkpointer
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Mapping
import yaml
import sys
import json
import re

from partseg_hook import *
from PointNet_data import *
from PointNet_utils import *
from PointNet import part_seg
from partseg_build import build_optimizer as _build_optimizer

__all__ = ["dict_to_namespace", "transform_namespace", "default_argument_parser", "default_writers", "PartSegTrainer"]


def dict_to_namespace(config_dict):
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            value = dict_to_namespace(value)
        setattr(namespace, key, value)
    return namespace
    
def transform_namespace(ns):
    for key, value in vars(ns).items():
        if isinstance(value, dict):
            setattr(ns, key, dict_to_namespace(value))
    return ns
    
def default_argument_parser(epilog=None):
    """
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval", action="store_true", help="perform evaluation only")

    return parser


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
    ]


class PartSegTrainer(SimpleTrainer):
    def __init__(self, model, args, **kwargs):
        self.args = args
        self.model = model
        super().__init__(model, data_loader = self.build_train_loader(args), 
                         optimizer = self.build_optimizers(), **kwargs)
        self.val_data_loader = self.build_val_loader(args)
        self.num_classes = args.num_cat
        self.lr_scheduler = self.build_lr_scheduler(args, self.optimizer)

        self.start_iter = 0
        # self.max_iter = len(self.data_loader) * args.epoch
        self.max_iter = args.max_iter
        self.checkpointer = Checkpointer(
            model=self.model, save_dir=args.output_dir, 
            optimizer=self.optimizer, scheduler=self.lr_scheduler
        )

        self.register_hooks(self.build_hooks(self.args))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def resume_or_load(self, resume = True):
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.checkpointer.resume_or_load(path = self.args.output_dir, resume= resume)
            curr_iter = self.checkpointer.get_checkpoint_file()
            curr_iter = ''.join(re.findall(r'[0-9]', curr_iter))
            self.iter = int(curr_iter) if len(curr_iter) > 0 else self.max_iter
            self.start_iter = self.iter + 1
        # else:
        #     self.model.apply(initialize_weights)

    def predict(self):
        self.model.eval()
        pred_args = self.args.pred_args
        total_acc = 0
        total_acc_iou = 0

        total_per_cat_acc = np.zeros((self.num_classes)).astype(np.float32)
        total_per_cat_iou = np.zeros((self.num_classes)).astype(np.float32)
        total_per_cat_seen = np.zeros((self.num_classes)).astype(np.int32)

        test_dataset, test_dict = self._test_dataset(self.args)
        object2setofoid, objcats, objnames, on2oid, cpid2oid = test_dict.values()
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size = 1, 
                                                      shuffle = True, 
                                                      collate_fn = partseg_collator(pred_args.point_num))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_dataloader), 
                                total = len(test_dataloader)):
                data, seg, label = batch.values()
                data, seg, label = data.to(device), seg.to(device), label.to(device)
                labels_pred, seg_pred, end_points = self.model(data, label)
                
                
                iou_oids = object2setofoid[objcats[label]]
                non_cat_labels = list(set(np.arange(pred_args.num_parts)).difference(set(iou_oids)))
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
            log_message += f'\niteration:\t{self.iter + 1}\nAccuracy:\t{avg_acc:.6f}\nIoU:\t{avg_iou:.6f}\n'

            for cat in range(self.num_classes):
                log_message += f'\n{objcats[cat]}\t\tTotal Number:\t{str(total_per_cat_seen[cat])}\n'
                if total_per_cat_seen[cat] > 0:
                    log_message += f'Accuracy:\t{total_per_cat_acc[cat] / total_per_cat_seen[cat]:.6f}\n'
                    log_message += f'IoU\t{total_per_cat_iou[cat] / total_per_cat_seen[cat]:.5f}\n'
            
            with open(args.output_dir + 'test_metric.pkl', 'a') as f:
                f.write(log_message)

    def build_train_loader(self, args):
        cl_args = args.train_collator 
        training_file_list = os.path.join(args.hdf5_data_dir, 'train_hdf5_file_list.txt')
        train_file_list = getDataFiles(training_file_list)
        train_dataset = load_data(args.hdf5_data_dir, train_file_list, dataset_type=args.dataset_type)
        return torch.utils.data.DataLoader(train_dataset, 
                                        batch_size = args.batch, 
                                        # collate_fn = partseg_train_collator(2048, 
                                        #                                     rotation_prob = cl_args.rotation_prob,
                                        #                                     jitter_prob = cl_args.jitter_prob,)
                                          )
    
    def build_val_loader(self, args):
        cl_args = args.train_collator 
        validation_file_list = os.path.join(args.hdf5_data_dir, 'val_hdf5_file_list.txt')
        val_file_list = getDataFiles(validation_file_list)
        val_dataset = load_data(args.hdf5_data_dir, val_file_list, dataset_type = 'dataset')
        return torch.utils.data.DataLoader(val_dataset, 
                                        batch_size = args.batch, 
                                        # collate_fn = partseg_train_collator(2048, 
                                        #                                     rotation_prob = cl_args.rotation_prob,
                                        #                                     jitter_prob = cl_args.jitter_prob,)
                                          )    
    def _test_dataset(self, args):
        ply_data_dir = os.path.join(args.base_dir, 'PartAnnotation/')
        test_file_list = os.path.join(args.base_dir, 'testing_ply_file_list.txt')
        object2setofoid, objcats, objnames, on2oid, color_map, cpid2oid = prepare_test_data(hdf5_data_dir)
        pts_files, seg_files, labels = getTestDataFiles(test_file_list)
        test_dict = dict(zip(['object2setofoid', 'objcats', 'objnames', 'on2oid', 'cpid2oid'],
                        [object2setofoid, objcats, objnames, on2oid, cpid2oid]))
        return partseg_testset(pts_files, seg_files, labels, on2oid, 
                               objcats, ply_data_dir), test_dict
        
    def build_hooks(self, args):
        def eval_fn():
            return self.evaluate_model()

        bn_args = args.bn_momentum
        period_args = args.period
        hooks = [
            PeriodicCheckpointer(self.checkpointer, period= period_args.eval_period), 
            LRSchedulerHook(self.optimizer, self.lr_scheduler),
            BNMomentumHook(
                self.model,
                bn_init_decay=bn_args.bn_init_decay, 
                bn_decay_rate=bn_args.bn_decay_rate, 
                decay_step=bn_args.bn_decay_step, 
                clip=bn_args.bn_decay_clip
            ),
        ]

        hooks.append(EvalHook(eval_period= period_args.eval_period, eval_function=eval_fn, args = args)) # Do evaluation after checkpointer for debug
        # hooks.append(BestCheckpointer(eval_period= period_args.eval_period, 
        #                               checkpointer=self.checkpointer, val_metric = 'acc'))
        hooks.append(PeriodicWriter(self.build_writers(), period= period_args.write_period))
        
        return hooks


    def build_writers(self, ):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.args.output_dir, self.max_iter)
    
    def build_optimizers(self,):
        return _build_optimizer(self.args, self.model)

    def build_lr_scheduler(self, args, optimizer: torch.optim.Optimizer):
        assert args.lr_scheduler.name == "EMAclip", "lr scheduler only support EMAclip!"
        return EMAclip(optimizer, args)
    
    def _init_metrics(self):
        return{  
            "total_loss": 0.0,
            "total_label_loss": 0.0,
            "total_seg_loss": 0.0,
            "total_label_acc": 0.0,
            "total_seg_acc": 0.0,
            "total_label_acc_per_cat": np.zeros((self.num_classes)).astype(np.float32),
            "total_seg_acc_per_cat": np.zeros((self.num_classes)).astype(np.float32),
            "total_seen_per_cat": np.zeros((self.num_classes)).astype(np.int32),
        }

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                PartSegTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise
                
    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, float],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = copy.deepcopy(loss_dict)
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            # total_losses_reduced = sum(metrics_dict.values())
            # if not np.isfinite(total_losses_reduced):
            #     raise FloatingPointError(
            #         f"Loss became infinite or NaN at iteration={cur_iter}!\n"
            #         f"loss_dict = {metrics_dict}"
            #     )

            # storage.put_scalar(
            #     "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            # )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)
    

if __name__ == '__main__':
    parser = default_argument_parser()
    args = parser.parse_args()

    # Load json configuration
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**json.load(f))
            
        args = parser.parse_args()
        args = transform_namespace(args)

    trainer = PartSegTrainer(model=part_seg(), args= args)
    try:
        trainer.resume_or_load(resume = True)
    except:
        trainer.resume_or_load(resume = False)
    trainer.predict()
    # trainer.train()