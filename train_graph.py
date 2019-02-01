#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : train_graph.py
# Create date : 2019-02-01 17:22
# Modified date : 2019-02-01 21:52
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import copy
import time
import torch
from graph import TransferLearnGraph
import status
import record

class TrainTransferLearnGraph(TransferLearnGraph):
    def __init__(self, data_dict, config):
        super(TrainTransferLearnGraph, self).__init__(data_dict, config)

    def _run_a_epoch(self, epoch):
        status.update_epoch(epoch, self.status_dict)
        start = time.time()
        self._train_a_epoch()
        self._eval_a_epoch()
        end = time.time()

        status.update_eplapsed_time(start, end, self.status_dict)
        record.save_epoch_status(self.status_dict, self.config)
        self._save_trained_model()

    def _train_a_step(self, inputs, labels,):
        model = self.graph_dict["model"]
        criterion = self.graph_dict["criterion"]
        optimizer = self.graph_dict["optimizer"]
        device = self.config["device"]

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        l = loss.item()*inputs.size(0)
        corrects = torch.sum(preds == labels.data)
        return l, corrects

    def _train_a_epoch(self):
        dataloaders = self.data_dict["dataloaders"]
        dataset_sizes = self.data_dict["dataset_sizes"]
        model = self.graph_dict["model"]
        scheduler = self.graph_dict["scheduler"]
        running_loss = 0.0
        running_corrects = 0

        scheduler.step()
        model.train()
        for inputs, labels in dataloaders["train"]:
            loss, corrects = self._train_a_step(inputs, labels)
            running_loss += loss
            running_corrects += corrects

        train_epoch_loss = running_loss / dataset_sizes["train"]
        train_epoch_acc = running_corrects.double() / dataset_sizes["train"]
        status.train_epoch_update_status_dict(train_epoch_loss, train_epoch_acc, self.status_dict)

    def _eval_a_step(self, inputs, labels):
        model = self.graph_dict["model"]
        criterion = self.graph_dict["criterion"]
        device = self.config["device"]
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        l = loss.item() * inputs.size(0)
        corrects = torch.sum(preds == labels.data)
        return l, corrects

    def _eval_a_epoch(self):
        epoch = self.status_dict["epoch"]
        dataloaders = self.data_dict["dataloaders"]
        dataset_sizes = self.data_dict["dataset_sizes"]
        model = self.graph_dict["model"]

        dataloader = dataloaders["val"]
        dataset_size = dataset_sizes["val"]
        running_loss = 0.0
        running_corrects = 0

        model.eval()
        for inputs, labels in dataloader:
            loss, corrects = self._eval_a_step(inputs, labels)
            running_loss += loss
            running_corrects += corrects

        val_epoch_loss = running_loss / dataset_size
        val_epoch_acc = running_corrects.double() / dataset_size

        status.val_epoch_update_status_dict(val_epoch_loss, val_epoch_acc, epoch, model, self.status_dict)

    def train_the_model(self):
        model = self.graph_dict["model"]
        record.save_content(self.config, model)
        num_epochs = self.config["epochs"]

        self.status_dict["best_model_wts"] = copy.deepcopy(model.state_dict())
        epoch_start = self.status_dict["epoch"]

        for epoch in range(epoch_start + 1, num_epochs):
            self._run_a_epoch(epoch)
