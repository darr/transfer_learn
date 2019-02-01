#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-01-30 13:35
# Modified date : 2019-02-01 18:33
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import ants_bees_data_set
from train_graph import TrainTransferLearnGraph
from test_graph import TestTransferLearnGraph
from etc import config
import record

def with_finetune():
    print_str = "run the model with finetune=True"
    config["finetune"] = True
    record.save_content(config, print_str)
    record.record_dict(config, config)
    data_dict = ants_bees_data_set.get_dataset_info_dict(config)

    g = TrainTransferLearnGraph(data_dict, config)
    g.train_the_model()

    test_g = TestTransferLearnGraph(data_dict, config)
    test_g.test_the_model()

def without_finetune():
    print_str = "run the model with finetune=False"
    config["finetune"] = False
    record.save_content(config, print_str)
    record.record_dict(config, config)
    data_dict = ants_bees_data_set.get_dataset_info_dict(config)

    g = TrainTransferLearnGraph(data_dict, config)
    g.train_the_model()

    test_g = TestTransferLearnGraph(data_dict, config)
    test_g.test_the_model()

def run():
    with_finetune()
    without_finetune()

run()
