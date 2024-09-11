## based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/hooks.py
import datetime
import itertools
import logging
import math
import operator
import os
import pickle
import tempfile
import time
import warnings
from collections import Counter
import tqdm
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from fvcore.common.checkpoint import Checkpointer
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from contextlib import contextmanager

import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils.events import EventWriter
from detectron2.utils.file_io import PathManager

from detectron2.engine.train_loop import HookBase


__all__ = [
    "CallbackHook",
    "NextListHook",
    "tqdmHook",
    "IterationTimer",
    "PeriodicWriter",
    "PeriodicCheckpointer",
    "BestCheckpointer",
    "LRSchedulerHook",
    "EvalHook",
    "BNMomentumHook",
    "EMAclip",
    "BNMomentum",
]

# todo: cook a `tqdm` Hook
class CallbackHook(HookBase):
    """
    Create a hook using callback functions provided by the user.
    """

    def __init__(self, *, before_train=None, after_train=None, before_step=None, after_step=None):
        """
        Each argument is a function that takes one argument: the trainer.
        """
        self._before_train = before_train
        self._before_step = before_step
        self._after_step = after_step
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        # The functions may be closures that hold reference to the trainer
        # Therefore, delete them to avoid circular reference.
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)

class NextListHook(HookBase):
    def __init__(self, period):
        self.list_period = period
    
    def after_step(self):
        current_iter = self.trainer.iter
        if current_iter % self.list_period == 0:
            self.trainer.dataloader = self.trainer.build_train_loader(self.trainer.args)
            self.trainer._data_loader_iter_obj = iter(self.trainer.dataloader)
            self.trainer.curr_idx += 1
            # print(self.trainer.curr_idx)
            # print('switch to another list...')

class tqdmHook(HookBase):
    def __init__(self, curr_iter, max_iter):
        self.curr_iter = curr_iter
        self.max_iter = max_iter
        self.progress_bar = tqdm.tqdm(total = self.max_iter, initial = self.curr_iter)

    def before_train(self):
        self.progress_bar.clear()
        
    def after_step(self):
        time.sleep(0.1)
        self.progress_bar.update(1)

    def after_train(self):
        self.progress_bar.close()
        
class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.

    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.storage.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer.storage.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)


class BestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                f"Given val metric {self._val_metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)
            self._logger.info(
                f"Saved best model as latest eval score for {self._val_metric} is "
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)
        else:
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
            )

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        # same conditions as `EvalHook`
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()


class LRSchedulerHook(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        self._best_param_group_id = LRSchedulerHook.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)


class BNMomentumHook(HookBase):
    def __init__(self, model, bn_init_decay=0.1, bn_decay_rate=0.5, decay_step=40, clip=0.99):
        self.bn_momentum = BNMomentum(model, bn_init_decay, bn_decay_rate, decay_step, clip)
    
    # def before_train(self):
    #     self.bn_momentum.update(0)

    def after_step(self):
        current_iter = self.trainer.iter
        self.bn_momentum.update(current_iter)
        momentum = self.bn_momentum.get_momentum()[0]
        self.trainer.storage.put_scalar("bn_momentum", momentum, smoothing_hint=False)


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, args, eval_after_train=True):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still evaluate after the last iteration
                if `eval_after_train` is True).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            eval_after_train (bool): whether to evaluate after the last iteration

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self._eval_after_train = eval_after_train
        self._args = args

    def _do_eval(self):
        results = self._func()
        self.trainer.model.train()  # Ensure model returns to training mode @ChatGPT
        # print(results)
        if results:
            flattened_results = flatten_results_dict(results)
            log = f'\n{self.trainer.iter + 1}\n'
            for k, v in flattened_results.items(): 
                log += f'\n{k}\t{v}\n'
            log += '\n'
            with open(os.path.join(self._args.output_dir, 'eval_log.pkl'), 'a') as f: 
                f.write(log)
                # pickle.dump(flattened_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            # self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)
        
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        # print(next_iter)
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()
        self.trainer.model.train()
                

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class EMAclip(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        clip (float): Minimum learning rate. Default 1e-5.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self, optimizer: Optimizer, args, last_epoch=-1, 
        verbose="deprecated"
    ):  
        lr_args = args.lr_scheduler
        self.gamma = lr_args.gamma
        self.decay_step = lr_args.decay_step
        self.clip = lr_args.clip
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#StepLR
        if (self.last_epoch == 0) or (self.last_epoch % self.decay_step != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [max(group["lr"] * self.gamma, self.clip) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * math.pow(self.gamma, self.last_epoch//self.decay_step), self.clip) for base_lr in self.base_lrs]
    
    def step(self, epoch =None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch =  epoch
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr
            
        if self.verbose:
            print(f'Iteration {self.last_epoch}: setting learning rate to {self._last_lr}.')

class BNMomentum:
    def __init__(self, model, bn_init_decay=0.1, bn_decay_rate=0.5, decay_step=40, clip=0.99):
        self.model = model
        self.bn_init_decay = bn_init_decay
        self.bn_decay_rate = bn_decay_rate
        self.decay_step = decay_step
        self.clip = clip

    @torch.no_grad()
    def update(self, current_iter):
        new_momentum = self.bn_init_decay * math.pow(self.bn_decay_rate, current_iter // self.decay_step)
        new_momentum = max(new_momentum, 1 - self.clip)
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.momentum = new_momentum

    def get_momentum(self):
        return [
            module.momentum for module in self.model.modules()
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        ]