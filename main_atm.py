#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:41:37 2018

@author: traffic203
"""

from __future__ import absolute_import, division, print_function

import os
import sys

import torch

from EStraffic import ES
from train_atm import train_loop, from_a_to_mlv, run_baseline
import numpy as np
import scipy.io as scio
from torch.autograd import Variable
from networks0 import rm_vsl_co
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "/home/tianyu/code/sumo/bin/sumo"
sumoConfig = "rl_vsl.sumo.cfg"

# chkpt_dir = 'checkpoints/tianyu_feb_13_net1_lc6/'
chkpt_dir = 'checkpoints/tianyu_feb_19_net0_60/latest3500.pth'

if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)
        
synced_model =  ES(12, 5, 80)

for param in synced_model.parameters():
    param.requires_grad = False

train_loop(100000, 0.1, 10, 0.2, 1, synced_model, chkpt_dir)
# state_dict = torch.load(chkpt_dir)  ##train from the previous chkpt?
# synced_model.load_state_dict(state_dict)
horizon=60
net = rm_vsl_co(control_horizon=horizon)
print('control horizon is:',horizon)
net.writenewtrips()
print('baseline result')
total_flow = run_baseline(net)
fname = 'output_sumo.xml'
with open(fname, 'r') as f:  # 打开文件
    lines = f.readlines()  # 读取所有行
    last_line = lines[-2]  # 取最后一行
traveltime = 'meanTravelTime='
h = last_line.index(traveltime)
aat_tempo = float(last_line[h+16:h+21])
print( 'NoVSL: Average Travel Time: %.4f' % aat_tempo, 'Total flow: %.4f' % total_flow )
# net.close()
print('our model result')
this_model_return  = 0
#net_index += 1 ##what this mean?
v = 29.06*np.ones(5,)
net.start_new_simulation(write_newtrips = False)
s, simulationSteps, r = net.run_step_es(v)
s = torch.from_numpy(s)
this_model_return  += r
dvsl = []
while simulationSteps < 18000:
    s = s.float()
    s = s.view(1, synced_model.num_inputs)
    with torch.no_grad():
        action = synced_model(Variable(s))
    action = action[0].data.numpy()
    v = from_a_to_mlv(action)
    dvsl.append(v)
    s_, simulationSteps, r = net.run_step_es(v)
    this_model_return  += r
    s = torch.from_numpy(s_)
fname = 'output_sumo.xml'
with open(fname, 'r') as f:  # 打开文件
    lines = f.readlines()  # 读取所有行
    last_line = lines[-2]  # 取最后一行
traveltime = 'meanTravelTime='
h = last_line.index(traveltime)
aat_tempo = float(last_line[h+16:h+21])
print( 'ESDVSL: Average Travel Time: %.4f' % aat_tempo, 'Total flow: %.4f' % this_model_return )
net.close()

## to do : test different horizon