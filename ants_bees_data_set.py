#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : ants_bees_data_set.py
# Create date : 2019-01-30 13:51
# Modified date : 2019-02-01 21:48
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import torch
from torchvision import datasets, models, transforms

def _get_image_dataset(data_path, kind, data_transforms):
    return datasets.ImageFolder(os.path.join(data_path, kind), data_transforms[kind])

def _get_a_dataloader(image_datasets, kind, config):
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    return torch.utils.data.DataLoader(image_datasets[kind], batch_size=batch_size, shuffle=True, num_workers=num_workers)

def _get_dataloaders(image_datasets,config):
    dataloaders = {}
    dataloaders["train"] = _get_a_dataloader(image_datasets, "train", config)
    dataloaders["val"] = _get_a_dataloader(image_datasets,"val", config)
    return dataloaders

def _get_data_transforms(config):
    img_size = config["image_size"]
    resize = config["resize"]
    mean = config["mean"]
    std = config["std"]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms

def _get_image_datasets(config):
    data_path = config["data_path"]
    data_transforms = _get_data_transforms(config)
    image_datasets = {}
    image_datasets["train"] = _get_image_dataset(data_path,"train", data_transforms)
    image_datasets["val"] = _get_image_dataset(data_path,"val",data_transforms)
    return image_datasets

def _get_dataset_sizes(datasets):
    dataset_sizes = {}
    dataset_sizes["train"] = len(datasets["train"])
    dataset_sizes["val"] = len(datasets["val"])
    return dataset_sizes

def _get_data_dict(dataloaders, dataset_sizes, class_names):
    data_dict = {}
    data_dict["dataloaders"] = dataloaders
    data_dict["dataset_sizes"] = dataset_sizes
    data_dict["class_names"] = class_names
    return data_dict

def get_dataset_info_dict(config):
    image_datasets = _get_image_datasets(config)

    dataloaders = _get_dataloaders(image_datasets,config)
    dataset_sizes = _get_dataset_sizes(image_datasets)
    class_names = image_datasets['train'].classes

    data_dict = _get_data_dict(dataloaders, dataset_sizes, class_names)
    return data_dict
