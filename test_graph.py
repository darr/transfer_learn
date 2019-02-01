#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : test_graph.py
# Create date : 2019-02-01 17:21
# Modified date : 2019-02-01 21:49
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from graph import TransferLearnGraph
import torch
import record

def imshow(inp, config, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(config["mean"])
    std = np.array(config["std"])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def _test_model(dataloader, model, config):
    device = config["device"]
    with torch.no_grad():
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        return inputs, preds

def _show_test_images(inputs, class_names, preds, config):
    images_so_far = 0
    num_images = inputs.size()[0]
    for j in range(num_images):
        images_so_far += 1
        ax = plt.subplot(num_images//2, 2, images_so_far)
        ax.axis('off')
        ax.set_title('predicted: {}'.format(class_names[preds[j]]))
        imshow(inputs.cpu().data[j], config)

    save_path = record.get_check_point_path(config)
    name = "test_images.jpg"
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
#    plt.show()

def run_test(dataloader, model, class_names, config):
    model.eval()
    inputs, preds = _test_model(dataloader, model, config)
    _show_test_images(inputs, class_names, preds, config)

class TestTransferLearnGraph(TransferLearnGraph):
    def __init__(self, data_dict, config):
        super(TestTransferLearnGraph, self).__init__(data_dict, config)

    def test_the_model(self):
        dataloader = self.data_dict["dataloaders"]['val']

        model = self.graph_dict["model"]
        model.load_state_dict(self.status_dict["best_model_wts"])

        class_names = self.data_dict["class_names"]
        run_test(dataloader, model, class_names, self.config)
