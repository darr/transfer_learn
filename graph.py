#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2019-01-30 14:25
# Modified date : 2019-02-01 18:19
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

import record
import status

def _add_last_layer(model, config):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(config["device"])
    return model

def _get_model(config):
    finetune = config["finetune"]
    model = torchvision.models.resnet18(pretrained=True)
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    return _add_last_layer(model, config)

def _get_optimizer(config, model):
    learn_rate = config["learn_rate"]
    momentum = config["momentum"]
    optimizer_ft = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
    return optimizer_ft

def _get_scheduler(optimizer, config):
    step_size = config["step_size"]
    gamma = config["gamma"]
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

class TransferLearnGraph(object):
    def __init__(self, data_dict, config):
        super(TransferLearnGraph, self).__init__()
        self.config = config
        self.data_dict = data_dict
        self.graph_dict = self._init_graph_dict(config)
        self.status_dict = status.get_status_dict()
        self._load_train_model()

    def _save_trained_model(self):
        model_dict = self._get_model_dict()
        file_full_path = record.get_check_point_file_full_path(self.config)
        torch.save(model_dict, file_full_path)

    def _init_graph_dict(self, config):
        graph_dict = {}
        graph_dict["model"] = _get_model(config)
        graph_dict["criterion"] = nn.CrossEntropyLoss()
        graph_dict["optimizer"] = _get_optimizer(config, graph_dict["model"])
        graph_dict["scheduler"] = _get_scheduler(graph_dict["optimizer"], config)
        return graph_dict

    def _get_model_dict(self):
        model_dict = {}
        model_dict["model"] = self.graph_dict["model"].state_dict()
        model_dict["criterion"] = self.graph_dict["criterion"].state_dict()
        model_dict["optimizer"] = self.graph_dict["optimizer"].state_dict()
        model_dict["scheduler"] = self.graph_dict["scheduler"].state_dict()

        model_dict["status_dict"] = self.status_dict
        model_dict["config"] = self.config
        return model_dict

    def _load_model_dict(self, checkpoint):
        self.graph_dict["model"].load_state_dict(checkpoint["model"])
        self.graph_dict["criterion"].load_state_dict(checkpoint["criterion"])
        self.graph_dict["optimizer"].load_state_dict(checkpoint["optimizer"])
        self.graph_dict["scheduler"].load_state_dict(checkpoint["scheduler"])

        self.status_dict = checkpoint["status_dict"]
        self.config = checkpoint["config"]

    def _load_train_model(self):
        file_full_path = record.get_check_point_file_full_path(self.config)
        if os.path.exists(file_full_path) and self.config["train_load_check_point_file"]:
            checkpoint = torch.load(file_full_path)
            self._load_model_dict(checkpoint)
