#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np

import decoder as decoder
#from . import utils as utils
import video_container as container
import transform as transform
from slowfast.datasets import utils as utils


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

    def __init__(self, preprocess, path ,num_retries=10):
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

        self._video_meta = {}
        self._num_retries = num_retries
        NUM_SPATIAL_CROPS = 4
        NUM_ENSEMBLE_VIEWS = 5
        self._preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) 
        self._number_of_spatial_crops = NUM_SPATIAL_CROPS
        self._num_clips = NUM_SPATIAL_CROPS * NUM_ENSEMBLE_VIEWS

        self._path = path
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        path_to_file = self._path
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._unique_video_idx = []
        
        classes = list()
        with open(r"/data/Peter/Final/configs/ALL_smell_true.csv", "r") as class_file:
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
        sampling_rate = 2

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    False,
                    "pyav",
                )
            except Exception as e:
                index = random.randint(0, len(self._path_to_videos) - 1)

            # Select a random video if the current video was not able to access.
            if video_container is None:
                if index + 1 < len(self._path_to_videos):
                    index = index + 1
                else:
                    index = index-1
                continue
            NUM_ENSEMBLE_VIEWS = 10
            NUM_FRAMES =  64
            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                container = video_container,
                sampling_rate = sampling_rate,
                num_frames = NUM_FRAMES,
                clip_idx = temporal_sample_index,
                num_clips = NUM_ENSEMBLE_VIEWS,
                video_meta=None,
                target_fps=30,
                backend="pyav",
                max_spatial_scale=256,
            )
            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                print('try a new one')
                if index + 1 < len(self._path_to_videos):
                    index = index + 1
                else:
                    index = index-1
                continue


            datamean = [0.45, 0.45, 0.45]
            datastd = [0.225, 0.225, 0.225]

            #used by CLIP:
            datamean = [0.48145466, 0.4578275, 0.40821073]
            datastd = [0.26862954, 0.26130258, 0.27577711]

            frames = utils.tensor_normalize(
                frames, datamean, datastd
            )
            frames = frames.permute(3, 0, 1, 2)
            frames = transform.crop_EAC_image(frames, spatial_sample_index)
            label = self._labels[index]

            # Perform color normalization.
            '''if frames.dtype == torch.uint8:
                frames = frames.float()
                frames = frames / 255.0
            if type(datamean) == list:
                mean = torch.tensor(datamean)
            if type(datastd) == list:
                std = torch.tensor(datastd)
            frames = frames - mean
            frames = frames / std
            
            frames = frames.permute(0, 3, 1, 2)
            frames = transform.rescale(frames)
            frames = transform.uniform_crop(frames[0], 244)'''
            
            '''print(frames.size())
            a = frames.numpy()
            a = (a/a)*255
            a = a.astype(np.uint8)
            im = Image.fromarray(a[0,:,:,:])
            im.save("00.jpeg")'''
            '''im = Image.fromarray(a[1,:,:,:])
            im.save("01.jpeg")
            im = Image.fromarray(a[2,:,:,:])
            im.save("02.jpeg")
            im = Image.fromarray(a[3,:,:,:])
            im.save("03.jpeg")
            im = Image.fromarray(a[4,:,:,:])
            im.save("04.jpeg")'''
            #frames = frames.permute(0, 2, 3, 1)
            # T H W C -> C T H W. ????
            #frames = frames.permute(3, 0, 1, 2)
            #label = self._labels[index]
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
