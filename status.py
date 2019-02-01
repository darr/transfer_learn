#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : status.py
# Create date : 2019-02-01 13:41
# Modified date : 2019-02-01 21:51
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import copy

def get_status_dict():
    status_dict = {}
    status_dict["best_acc"] = 0.0
    status_dict["best_model_wts"] = None
    status_dict["elapsed_time"] = 0.0
    status_dict["epoch"] = 0
    status_dict["train_epoch_loss"] = 0.0
    status_dict["train_epoch_acc"] = 0.0
    status_dict["val_epoch_loss"] = 0.0
    status_dict["val_epoch_acc"] = 0.0
    status_dict["best_epoch"] = 0
    status_dict["best_acc"] = 0.0
    status_dict["epoch_eplapsed_time"] = 0.0
    status_dict["so_far_elapsed_time"] = 0.0
    return status_dict

def val_epoch_update_status_dict(val_epoch_loss, val_epoch_acc, epoch, model, status_dict):
    status_dict["val_epoch_loss"] = val_epoch_loss
    status_dict["val_epoch_acc"] = val_epoch_acc

    if val_epoch_acc > status_dict["best_acc"]:
        status_dict["best_epoch"] = epoch
        status_dict["best_acc"] = val_epoch_acc
        status_dict["best_model_wts"] = copy.deepcopy(model.state_dict())

def train_epoch_update_status_dict(train_epoch_loss, train_epoch_acc, status_dict):
    status_dict["train_epoch_loss"] = train_epoch_loss
    status_dict["train_epoch_acc"] = train_epoch_acc

def update_eplapsed_time(start, end, status_dict):
    status_dict["epoch_eplapsed_time"] = end - start
    status_dict["so_far_elapsed_time"] += status_dict["epoch_eplapsed_time"]

def update_epoch(epoch, status_dict):
    status_dict["epoch"] = epoch
