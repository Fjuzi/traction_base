#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
from numpy.lib.stride_tricks import as_strided
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
import torchvision.models as models
from torch.nn import functional as F

logger = logging.get_logger(__name__)

def build_scene_model(architecture_path):
    #These are for the original CNNs
    '''arch = "resnet18"
    model_file = architecture_path
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)'''
    
    #New architectures
    arch = "densenet161"
    model = models.__dict__[arch](num_classes=365) 
    state_dict = torch.load("/data/Peter/Final/densenet161_places365.pt")
    model.load_state_dict(state_dict)
    cur_device = torch.cuda.current_device()
    model = model.cuda(cur_device)
    model.eval()

    return model

@torch.no_grad()
def perform_test(test_loader, model, model_scene, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        model_scene (model): the pretrained scene recognition model
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    #f = open("log_test_results.txt", "w")

    f = open("./results/results_all_2/dumb.txt","a")
    f2 = open("./results/results_all_2/dumb_scene.txt","a")
    _prev_unique_video_idx = ""
    _prev_unique_video_idx_scene = ""
    for cur_iter, (inputs, labels, video_idx, meta, _unique_video_idx, _temporal_sample_index, _spatial_sample_index, _video_name) in enumerate(test_loader):
        
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            save_video_idx = video_idx
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()
        

        scene_inputs = inputs[1].clone().detach()
        # Perform the forward pass.
        preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        video_idx_numpy = video_idx.numpy()
        preds_numpy = preds.numpy()
        labels_numpy = labels.numpy()
        unique_video_idx = _unique_video_idx.numpy()
        temporal_sample_index = _temporal_sample_index.numpy()
        spatial_sample_index = _spatial_sample_index.numpy()
        video_name = _video_name
       
           
        index_counter = np.zeros(preds_numpy[0,:].shape)
        for i in range(video_idx_numpy.shape[0]):
            if str(labels_numpy[i]) != _prev_unique_video_idx:
                _prev_unique_video_idx = str(labels_numpy[i])
                f.close()
                filenamesm = "./results/results_all_2/" + _prev_unique_video_idx +".txt"
                if os.path.exists(filenamesm):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not
                    #print(labels_numpy[i])
                f = open(filenamesm, append_write)
            f.write(str(video_name[i]))
            f.write(";")
            f.write(str(unique_video_idx[i]))
            f.write(";")
            f.write(str(temporal_sample_index[i]))
            f.write(";")
            f.write(str(spatial_sample_index[i]))
            f.write(";")
            f.write(str(labels_numpy[i]))
            f.write(";")

            #idx_of_results = preds_numpy[i, :]
            #idx_of_results = idx_of_results.argsort()[-5:][::-1]
            #f.write( np.array2string( idx_of_results, separator=',', formatter = {'float_kind':lambda x: "%.8f"%x} ) )
            #f.write( np.array2string( preds_numpy[i, :], separator=',', formatter = {'float_kind':lambda x: "%.8f"%x} ) )
            #f.write(";")
            tmp = preds_numpy[i, :]
            tmp2 = list(tmp)
            for element in tmp2:
                f.write("%1.8f,"%element)
            #tmp = np.squeeze(tmp.reshape(1,600))
            #f.write( np.array2string( preds_numpy[i, :], separator=',', formatter = {'float_kind':lambda x: "%.8f"%x} ) )
            
            f.write("\n")

        
        for i in range(video_idx_numpy.shape[0]):
            if str(labels_numpy[i]) != _prev_unique_video_idx_scene:
                _prev_unique_video_idx_scene = str(labels_numpy[i])
                f2.close()
                filenamesm = "./results/results_all_2/" + _prev_unique_video_idx_scene +"_scene.txt"
                if os.path.exists(filenamesm):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not
                    #print(labels_numpy[i])
                f2 = open(filenamesm, append_write)
            f2.write(str(video_name[i]))
            f2.write(";")
            f2.write(str(unique_video_idx[i]))
            f2.write(";")
            f2.write(str(temporal_sample_index[i]))
            f2.write(";")
            f2.write(str(spatial_sample_index[i]))
            f2.write(";")
            f2.write(str(labels_numpy[i]))
            f2.write(";")
            logit = model_scene.forward(scene_inputs[i,:,32,:,:].unsqueeze(0))
            h_x = F.softmax(logit, 1).data.squeeze()
            #probs, idx = h_x.sort(0, True)
            tmp2 = h_x.tolist()
            for element in tmp2:
                f2.write("%1.8f,"%element)
            f2.write("\n")  
        

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    f2.close()
    f.close()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model_action = build_model(cfg)

    # Build the scene model
    model_scene = build_scene_model("/data/Peter/Final/resnet18_places365.pth.tar")

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model_action, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model_action)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model_action, model_scene, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
