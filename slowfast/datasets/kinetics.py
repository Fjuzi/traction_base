#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from slowfast.datasets import transform
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

from torchvision.utils import save_image

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        #self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS*cfg.TEST.NUM_SPATIAL_CROPS
        #this is always 4 as we use 4 tiles of the 360 degree video
        self._number_of_spatial_crops = 4
        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS*4
        

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = "/data/Peter/Data/Smells/"
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._unique_video_idx = []


        classes = list()
        #with open(r"/data/Peter/Final/configs/kinetics_smell_true.csv", "r") as class_file:
        with open(r"/data/Peter/Final/configs/labels.csv", "r") as class_file:
            for line in class_file:
                class_details = {
                    "classname": line.strip().split(',')[0],
                    "class_value": int(line.strip().split(',')[1]),
                    "smellname": line.strip().split(',')[2],
                    "smellvalue": int(line.strip().split(',')[3]),
                    "type": line.strip().split(',')[4]
                }
                classes.append(class_details)
        classes = tuple(classes)

        
        class_names = os.listdir(path_to_file)
        unique_video_counter = 0
        for class_id, one_class in enumerate(class_names):
            class_name_paths = os.path.join(path_to_file, one_class)

            class_dict, idx = next( ((item, i) for i, item in enumerate(classes) if item["smellname"]== one_class), (None,None) )
            if class_dict is None:   
                class_id = 0
            else:
                class_id = class_dict["smellvalue"]

            vido_ids = os.listdir(class_name_paths)
            for video_id in vido_ids:
                video_path = os.path.join(class_name_paths, video_id) 
                for idx in range(self._num_clips):
                    self._spatial_temporal_idx.append(idx)
                    self._path_to_videos.append(video_path)
                    self._labels.append(class_id)
                    self._unique_video_idx.append(unique_video_counter)
                unique_video_counter += 1
       
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {})".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index


        temporal_sample_index = self._spatial_temporal_idx[index]//self._number_of_spatial_crops
        
        spatial_sample_index = self._spatial_temporal_idx[index]%self._number_of_spatial_crops

        min_scale, max_scale, crop_size = (
            [self.cfg.DATA.TEST_CROP_SIZE] * 3
            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
            + [self.cfg.DATA.TEST_CROP_SIZE]
        )
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1

        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
                index = random.randint(0, len(self._path_to_videos) - 1)

            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=None,
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            #tmp = frames[32,:,:,:].cpu().numpy()
            #from PIL import Image
            #im = Image.fromarray(tmp)
            #im.save("fram.png")
            #asd

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                '''if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)'''
                print('try a new one')
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            #frames = frames.permute(0, 3, 1, 2)
            # Perform data augmentation.
            '''frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )'''

            frames = transform.crop_EAC_image(frames, spatial_sample_index)
            #tmp = frames[:,32,:,:].cpu().permute(1,2,0).numpy() * 255
            #import numpy as np
            #tmp2 = tmp.astype(np.uint8)
            #im = Image.fromarray(tmp2)
            #im.save("fram_crop_" + str(spatial_sample_index)+".png")
            #asd
            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}, self._unique_video_idx[index], temporal_sample_index, spatial_sample_index, self._path_to_videos[index]
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
