# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import json
import logging
import os
import time

import torch
import torch.distributed as dist
import tensorboardX

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.data.datasets import RPCPseudoDataset, ConcatDataset, RPCTestDataset
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from tools.tester import run_test_when_training


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def backward(loss_dict, optimizer, meters):
    losses = sum(loss for loss in loss_dict.values())

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = reduce_loss_dict(loss_dict)

    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    meters.update(loss=losses_reduced, **loss_dict_reduced)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    return loss_dict_reduced


def do_train(
        cfg,
        model,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        distributed,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training!!")
    meters = MetricLogger(delimiter="  ")

    model.train()
    start_training_time = time.time()
    end = time.time()
    total_max_iter = cfg.SOLVER.MAX_ITER
    print(total_max_iter)

    summary_writer = None
    if get_rank() == 0:
        summary_writer = tensorboardX.SummaryWriter(os.path.join(checkpointer.save_dir, 'tf_logs'))

    is_train = True
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    transforms = build_transforms(cfg, is_train)
    dataset_list = cfg.DATASETS.TRAIN
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)
    with open('/data7/lufficc/projects/rpc-detector/outputs_synthesize_v10_masks_density_map_0_45_threshold_cyclegan/inference/rpc_2019_test/pseudo_labeling.json') as fid:
        annotations = json.load(fid)

    total_iteration_index = 0
    for step in range(100):
        source_pseudo_dataset = RPCPseudoDataset(annotations=annotations, density=True, transforms=transforms)
        pseudo_dataset = RPCPseudoDataset(annotations=annotations, density=False, transforms=transforms)
        source_and_target_datasets = datasets + [source_pseudo_dataset]
        source_and_target_datasets = ConcatDataset(source_and_target_datasets)
        max_iter = 5000
        source_and_target_data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=0,
            datasets=[source_and_target_datasets],
            desired_num_iters=max_iter
        )

        target_data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=0,
            datasets=[pseudo_dataset],
            desired_num_iters=max_iter
        )

        for (source_and_target_images, source_and_target_targets, _), (target_images, target_targets, _) \
                in zip(source_and_target_data_loader, target_data_loader):
            data_time = time.time() - end
            total_iteration_index += 1

            arguments["iteration"] = total_iteration_index
            scheduler.step(total_iteration_index)
            # ----------------------------------------------------
            # -------------- source domain training --------------
            # ----------------------------------------------------
            source_and_target_images = source_and_target_images.to(device)
            source_and_target_targets = [target.to(device) for target in source_and_target_targets]
            loss_dict = model(source_and_target_images, source_and_target_targets, train_target=False)
            loss_dict_reduced = backward(loss_dict=loss_dict, optimizer=optimizer, meters=meters)
            # ----------------------------------------------------
            # -------------- target domain training --------------
            # ----------------------------------------------------
            target_images = target_images.to(device)
            target_targets = [target.to(device) for target in target_targets]
            target_loss_dict = model(target_images, target_targets, train_target=True)
            target_loss_dict = {'target_head_' + k: target_loss_dict[k] for k in target_loss_dict}
            target_loss_dict_reduced = backward(loss_dict=target_loss_dict, optimizer=optimizer, meters=meters)
            # ----------------------------------------------------
            # ----------------------------------------------------
            # ----------------------------------------------------

            # merge all loss
            # losses = sum(loss for loss in loss_dict.values())
            #
            # # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = reduce_loss_dict(loss_dict)
            #
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # meters.update(loss=losses_reduced, **loss_dict_reduced)
            #
            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()

            loss_dict_reduced.update(target_loss_dict_reduced)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (total_max_iter - total_iteration_index)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if total_iteration_index % 20 == 0 or total_iteration_index == max_iter:
                if summary_writer:
                    summary_writer.add_scalar('loss/total_loss', losses_reduced, global_step=total_iteration_index)
                    for name, value in loss_dict_reduced.items():
                        summary_writer.add_scalar('loss/%s' % name, value, global_step=total_iteration_index)

                    summary_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step=total_iteration_index)
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=total_iteration_index,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if total_iteration_index % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(total_iteration_index), **arguments)
                if total_iteration_index != total_max_iter:
                    model.module.run_inference = True
                    run_test_when_training(cfg, model, distributed, dataset_names=('rpc_2019_val',))
                    model.module.run_inference = False
                    model.train()  # restore train state

        print('Generating new pseudo labels...')
        test_dataset = RPCTestDataset('/mnt/data', filename='instances_test2019.json', images_dir='test2019', transforms=build_transforms(cfg, False))
        run_test_when_training(cfg, model, distributed, dataset_names=('rpc_2019_test',), datasets=[test_dataset])
        with open('/mnt/rpc-detector/density_map_0_45_threshold_finetune/inference/rpc_2019_test/pseudo_labeling.json') as fid:
            annotations = json.load(fid)
        print(len(annotations))
        model.train()  # restore train state

        if total_iteration_index >= total_max_iter:
            break

    checkpointer.save("model_final", **arguments)
    model.module.run_inference = True
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / total_max_iter
        )
    )
