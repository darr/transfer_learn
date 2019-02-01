#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc.py
# Create date : 2019-01-30 15:17
# Modified date : 2019-02-01 18:27
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch

config = {}

config["dataset"] = "hymenoptera_data"
config["data_path"] = "./data/%s" % config["dataset"]

config["epochs"] = 25
config["batch_size"] = 8
config["num_workers"] = 4
config["image_size"] = 224
config["resize"] = 256

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config["finetune"] = True
config["learn_rate"] = 0.001
config["momentum"] = 0.9
config["step_size"] = 7
config["gamma"] = 0.1

config["mean"] = [0.485, 0.456, 0.406]
config["std"] = [0.229, 0.224, 0.225]

config["train_load_check_point_file"] = True
