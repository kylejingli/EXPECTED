#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
import copy
from pycls.core.config import cfg

from pycls.utils import bn_update
#from rich.progress import track

logger = logging.get_logger(__name__)

def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    # torch.manual_seed(cfg.RNG_SEED) !!!!! for selecting different layers
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    #logger.info("Model:\n{}".format(model))
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        # Set complexity function to be module's complexity function
        model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        #test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    #test_meter.log_epoch_stats(cur_epoch)
    # test_meter.reset()
    return test_meter.get_return_results()


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)
def layerwise_update(model, cnt, q, best, update_num):
    rate = 1/32
    sigma = 0.1
    lr = 0.001
    acc = []
    improve = torch.ones(54)
    layer_idx = 0
    no_query_left = False
    test_loader = loader.construct_test_loader()
    for name, p in model.named_parameters():
        if "bn" in name:
            B = int(rate * p.shape[0])  # if dim=64, then B=64*rate=64*0.25=16
            noise_pos = torch.randn(B // 2, p.shape[0])
            noise = torch.cat((noise_pos, -noise_pos), dim=0).cuda()
            del noise_pos
            cnt += B
            if cnt > q:
                cnt -= B
                no_query_left = True
                break
            returned_err = torch.zeros(B)
            for b in range(B):
                model_q = copy.deepcopy(model)
                dict_q = model_q.state_dict()
                dict_q[name] += (sigma * noise[b, :])  # model_q is updated
                test_meter = meters.TestMeter(len(test_loader))
                bn_update(model_q, test_loader)
                top1, top5 = test_epoch(test_loader, model_q, test_meter, cur_epoch=1)
                returned_err[b] = top5
                del model_q, test_meter
            # compute gradient after quries B times
            mean = torch.mean(returned_err)
            std = torch.std(returned_err)
            returned_err = torch.nan_to_num( (returned_err - mean) / std, 1.0)
            returned_err_tiled = torch.reshape(returned_err, (-1, 1)).repeat(1, p.shape[0])
            grad_estimate = torch.mean(returned_err_tiled.cuda() * noise / sigma, 0)
            p.data = p - lr * grad_estimate
            # test for watching
            model_t = copy.deepcopy(model)  # avoid model is updated by func of bn_update
            test_meter = meters.TestMeter(len(test_loader))
            bn_update(model_t, test_loader)
            top1, top5 = test_epoch(test_loader, model_t, test_meter, cur_epoch=1)
            acc.append(top5)
            improve[layer_idx] = (best -top5)/B # minimize problem
            best = top5
            update_num += 1
            print(f'update step={update_num}, target error={top5}, query number {cnt}')
            with open("Gaussian_top5_tuning_monitor.txt", "a") as writer:
                writer.write("step=%s   top1 error=%s   top5 error=%s   query_numer=%s\n" % (str(update_num), str(top1), str(top5), str(cnt)))
            acc.append(top5)
            del model_t, test_meter
            layer_idx+=1
    improve = torch.maximum(improve, torch.zeros_like(improve))
    return  model, cnt, improve, acc, no_query_left, update_num

def additional_layer_update(model, alpha, cnt, q, update_num):
    his = np.zeros_like(alpha)
    softmax = torch.nn.Softmax(dim=0)
    probability = softmax(alpha)
    #import pdb;pdb.set_trace()
    p_cum = torch.cumsum(probability, dim=0)
    additional_budget = 400
    cal = 0
    break_tag = False
    rate = 1 / 32
    sigma = 0.1
    lr = 0.001
    acc = []
    no_query_left = False
    test_loader = loader.construct_test_loader()
    while not break_tag:
        rand = torch.rand(1)
        for idx in range(54):
            if p_cum[idx] > rand:
               break
        layer_idx = 0
        for name, p in model.named_parameters():
            if "bn" in name and layer_idx == idx:
                B = int(rate * p.shape[0])  # if dim=64, then B=64*rate=64*0.25=16
                if cal+B > additional_budget: # additional queries are not enough
                    break_tag = True
                    break
                cal+=B
                his[idx]+=1
                noise_pos = torch.randn(B // 2, p.shape[0])
                noise = torch.cat((noise_pos, -noise_pos), dim=0).cuda()
                del noise_pos
                cnt += B
                if cnt > q: # exceeds the total budget
                    cnt -= B  # for reporting the exact number of consumed queries
                    no_query_left = True
                    break_tag = True
                    break
                returned_err = torch.zeros(B)
                for b in range(B):
                    model_q = copy.deepcopy(model)
                    dict_q = model_q.state_dict()
                    dict_q[name] += (sigma * noise[b, :])  # model_q is updated
                    test_meter = meters.TestMeter(len(test_loader))
                    bn_update(model_q, test_loader)
                    top1, top5 = test_epoch(test_loader, model_q, test_meter, cur_epoch=1)
                    returned_err[b] = top5
                    del model_q, test_meter
                # compute gradient after quries B times
                mean = torch.mean(returned_err)
                std = torch.std(returned_err)
                returned_err = torch.nan_to_num((returned_err - mean) / std , 1.0)
                returned_err_tiled = torch.reshape(returned_err, (-1, 1)).repeat(1, p.shape[0])
                grad_estimate = torch.mean(returned_err_tiled.cuda() * noise / sigma, 0)
                p.data = p - lr * grad_estimate
                # test for watching
                model_t = copy.deepcopy(model)  # avoid model is updated by func of bn_update
                test_meter = meters.TestMeter(len(test_loader))
                bn_update(model_t, test_loader)
                top1, top5 = test_epoch(test_loader, model_t, test_meter, cur_epoch=1)
                acc.append(top5)
                update_num += 1
                print(f'update step={update_num}, target error={top1}, query number={cnt}')      
                with open("Gaussian_top5_tuning_monitor.txt", "a") as writer:
                    writer.write("step=%s   top1 error=%s   top5 error=%s   query_numer=%s\n" % (str(update_num), str(top1), str(top5), str(cnt)))
                del model_t, test_meter
                break # because we only update one layer every time
            else:
                layer_idx+=1

    return model, cnt, acc, no_query_left, top5, update_num, probability, his

def layerwise_coordinate_parameter_search(corruptions, levels):
    all_results = []
    for corruption_level in levels:
        lvl_results = []
        for corruption_type in corruptions:
            #if corruption_type != 'impulse_noise': continue
            cfg.TEST.CORRUPTION = corruption_type
            cfg.TEST.LEVEL = corruption_level
            setup_env()
            model = setup_model()
            # load pre-trained model
            checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
            # params for optimizing
            query_budget = 2000
            layer_num = 54
            update_num = 0
            # test data preparation
            test_loader = loader.construct_test_loader()
            # performance of the load model
            model_0 = copy.deepcopy(model)
            test_meter = meters.TestMeter(len(test_loader))
            bn_update(model_0, test_loader)
            top1, top5 = test_epoch(test_loader, model_0, test_meter, cur_epoch=1)
            print(f'iter={update_num}, target error={top5}')
            last_best = top5
            del (model_0, test_meter)
            cnt = 0  # record query times
            with open("Gaussian_top5_tuning_monitor.txt", "a") as writer:
                writer.write("step=%s   top1 error=%s   top5 error=%s   query_numer=%s\n" % (str(update_num), str(top1), str(top5), str(cnt)))
            beta = 100
            ACC = []
            alpha = torch.zeros(layer_num)
            no_query_left = False
            pro = np.ones(layer_num)/layer_num
            histogram = np.zeros(layer_num)
            while True:
                # conduct base update for every layer.
                model, cnt, improve, acc, no_query_left, update_num = layerwise_update(model, cnt, query_budget, last_best, update_num)
                ACC = np.concatenate((ACC,acc), axis=None)
                if no_query_left: break
                alpha+=(beta*improve)
                # additional update
                model, cnt, acc, no_query_left, last_best, update_num, probability, his_iter = additional_layer_update(model, alpha,cnt, query_budget, update_num)
                pro = np.vstack((pro, probability.cpu().numpy()))
                histogram = np.vstack((histogram, his_iter))
                ACC = np.concatenate((ACC,acc), axis=None)
                if no_query_left: break
            #np.save('ACC_top1_Gauss_layerwise.npy', ACC)
            #np.save("layer_pro.npy", pro)
            #np.save("histogram.npy", histogram)
            lvl_results.append(top5)
            os.exit(0)
        all_results.append(lvl_results)
    return ERR

def fine_tuning(corruptions, levels):
    """Use feed back to fine-tune some part of the model. (with all kind of corruptions)"""
    all_results = []
    for corruption_level in levels:
        lvl_results = []
        for corruption_type in corruptions:
            cfg.TRAIN.CORRUPTION = corruption_type
            cfg.TRAIN.LEVEL = corruption_level
            cfg.TEST.CORRUPTION = corruption_type
            cfg.TEST.LEVEL = corruption_level

            # Setup training/testing environment
            setup_env()
            # Construct the model, loss_fun, and optimizer
            model = setup_model()
            loss_fun = builders.build_loss_fun().cuda()
            optimizer = optim.construct_optimizer(model)
            # Load checkpoint or initial weights
            start_epoch = 0
            checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model, strict=cfg.TRAIN.LOAD_STRICT)
            logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
            # Create data loaders and meters
            train_loader = loader.construct_train_loader()
            test_loader = loader.construct_test_loader()
            train_meter = meters.TrainMeter(len(train_loader))
            test_meter = meters.TestMeter(len(test_loader))
            # Compute model and loader timings
            if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
                benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
            
            # Perform the training loop
            logger.info("Start epoch: {}".format(start_epoch + 1))
            for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
                if cfg.TRAIN.ADAPTATION != 'test_only':
                    if cfg.TRAIN.ADAPTATION == 'update_bn':
                        bn_update(model, train_loader)
                    elif cfg.TRAIN.ADAPTATION == 'min_entropy':
                        # Train for one epoch
                        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
                        bn_update(model, train_loader)

                    # Save a checkpoint
                    if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                        checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
                        logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
                        
                # Evaluate the model
                next_epoch = cur_epoch + 1
                if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
                    top1 = test_epoch(test_loader, model, test_meter, cur_epoch)
            lvl_results.append(top1)
        all_results.append(lvl_results)
    
    for lvl_idx in range(len(all_results)):
        logger.info("corruption level: {}".format(levels[lvl_idx]))
        logger.info("corruption types: {}".format(corruptions))
        logger.info(all_results[lvl_idx])

    return all_results

    
def test_ftta_model(corruptions, levels):
    """Use feed back to fine-tune some part of the model. (with all kind of corruptions)"""
    all_results = []
    for corruption_level in levels:
        lvl_results = []
        for corruption_type in corruptions:
            cfg.TRAIN.CORRUPTION = corruption_type
            cfg.TRAIN.LEVEL = corruption_level
            cfg.TEST.CORRUPTION = corruption_type
            cfg.TEST.LEVEL = corruption_level

            # Setup training/testing environment
            setup_env()
            # Construct the model, loss_fun, and optimizer
            model = setup_model()
            loss_fun = builders.build_loss_fun().cuda()
            optimizer = optim.construct_optimizer(model)
            # Load checkpoint or initial weights
            start_epoch = 0
            checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model, strict=cfg.TRAIN.LOAD_STRICT)
            logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
            # Create data loaders and meters
            train_loader = loader.construct_train_loader()
            test_loader = loader.construct_test_loader()
            train_meter = meters.TrainMeter(len(train_loader))
            test_meter = meters.TestMeter(len(test_loader))
            # Compute model and loader timings
            if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
                benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
            
            # Perform the training loop
            logger.info("Start epoch: {}".format(start_epoch + 1))
            for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
                if cfg.TRAIN.ADAPTATION != 'test_only':
                    if cfg.TRAIN.ADAPTATION == 'update_bn':
                        bn_update(model, train_loader)
                    elif cfg.TRAIN.ADAPTATION == 'min_entropy':
                        # Train for one epoch
                        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
                        bn_update(model, train_loader)

                    # Save a checkpoint
                    if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                        checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
                        logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
                        
                # Evaluate the model
                next_epoch = cur_epoch + 1
                if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
                    top1 = test_epoch(test_loader, model, test_meter, cur_epoch)
            lvl_results.append(top1)
        all_results.append(lvl_results)
    
    for lvl_idx in range(len(all_results)):
        logger.info("corruption level: {}".format(levels[lvl_idx]))
        logger.info("corruption types: {}".format(corruptions))
        logger.info(all_results[lvl_idx])

    return all_results


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
