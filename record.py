#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : record.py
# Create date : 2019-01-30 21:37
# Modified date : 2019-02-01 21:49
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os

def _get_param_str(config):
    # pylint: disable=bad-continuation
    param_str = "%s_%s_%s_%s_%s" % (
                                config["dataset"],
                                config["image_size"],
                                config["batch_size"],
                                config["learn_rate"],
                                config["finetune"],
                                )
    # pylint: enable=bad-continuation
    return param_str

def get_check_point_path(config):
    param_str = _get_param_str(config)
    directory = "%s/save/%s/" % (config["data_path"], param_str)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_check_point_file_full_path(config):
    path = get_check_point_path(config)
    param_str = _get_param_str(config)
    file_full_path = "%s%scheckpoint.tar" % (path, param_str)
    return file_full_path

def _write_output(config, con):
    save_path = get_check_point_path(config)
    file_full_path = "%s/output" % save_path
    f = open(file_full_path, "a")
    f.write("%s\n" %  con)
    f.close()

def record_dict(config, dic):
    save_content(config, "config:")
    for key in dic:
        dic_str = "%s : %s" % (key, dic[key])
        save_content(config, dic_str)

def save_content(config, con):
    print(con)
    _write_output(config, con)

def save_epoch_status(status_dict, config):
    num_epochs = config["epochs"]
    epoch = status_dict["epoch"]
    train_epoch_loss = status_dict["train_epoch_loss"]
    train_epoch_acc = status_dict["train_epoch_acc"]
    val_epoch_loss = status_dict["val_epoch_loss"]
    val_epoch_acc = status_dict["val_epoch_acc"]
    best_epoch = status_dict["best_epoch"]
    best_acc = status_dict["best_acc"]
    epoch_elapsed_time = status_dict["epoch_eplapsed_time"]
    so_far_elapsed_time = status_dict["so_far_elapsed_time"]

    # pylint: disable=bad-continuation
    save_str = '[%s/%s] [Train Loss:%.4f Acc:%.4f] [Val Loss:%.4f Acc:%.4f] [Best Epoch:%s Acc:%.4f] [%.4fs %.4fs]' % (
                            epoch,
                            num_epochs - 1,
                            train_epoch_loss,
                            train_epoch_acc,
                            val_epoch_loss,
                            val_epoch_acc,
                            best_epoch,
                            best_acc,
                            epoch_elapsed_time,
                            so_far_elapsed_time
                            )

    # pylint: enable=bad-continuation
    save_content(config, save_str)
