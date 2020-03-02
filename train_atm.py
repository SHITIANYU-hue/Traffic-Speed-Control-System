#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:20:47 2018

@author: traffic203
"""

from __future__ import absolute_import, division, print_function

import os
import math
import numpy as np

import torch
# import torch.legacy.optim as legacyOptim

import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable

from EStraffic import ES
import os
import sys

THISDIR = os.path.dirname('~/sumo/exercise/traci_pedestrian_crossing')

try:
    # tutorial in tests
    sys.path.append(os.path.join(THISDIR, '..', '..', '..', '..', "tools"))
    # tutorial in docs
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        THISDIR, "..", "..", "..")), "tools"))  
    import traci
    #import xml2csv
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
sumoBinary = checkBinary('sumo')
#sumoBinary = "/home/tianyushi/sumo_binaries/bin/sumo"
sumoConfig = "rl_vsl.sumo.cfg"
import matplotlib.pyplot as plt
from networks0 import rm_vsl_co

optimConfig = []
averageReward = []
maxReward = []
minReward = []
episodeCounter = []

def from_a_to_mlv(a):
    return 4.4704 + 2.2352*np.round(14.49*a)

def do_rollouts(models, random_seeds, return_queue, particular_tnet, are_negative, see_results):
    all_returns = []
    all_num_frames = []
    #net_index = 0
    for model in models:
        this_model_return  = 0
        this_model_num_frames = 0
        net = particular_tnet
        #net_index += 1
        v = 29.06*np.ones(5,)
        net.start_new_simulation(write_newtrips = False)
        s, simulationSteps, r = net.run_step_es(v)
        s = torch.from_numpy(s)
        this_model_return  += r
        while simulationSteps < 18000:
            s = s.float()
            s = s.view(1, model.num_inputs)
            with torch.no_grad():
                action = model(Variable(s))
            action = action[0].data.numpy()
            v = from_a_to_mlv(action)
            s_, simulationSteps, r = net.run_step_es(v)
            this_model_return  += r
            this_model_num_frames += 1
            s = torch.from_numpy(s_)
#        improvement = this_model_return/baseline
        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
        if see_results == True:
#            fname = 'output_sumo.xml'
#            with open(fname, 'r') as f:  # 打开文件
#                lines = f.readlines()  # 读取所有行
#                last_line = lines[-2]  # 取最后一行
#            traveltime = 'meanTravelTime='
#            h = last_line.index(traveltime)
#            aat_tempo = float(last_line[h+16:h+21])
            print('Total traffic flow: %f\n' % (this_model_return))
    net.close()        
    return_queue.put((random_seeds, all_returns, all_num_frames, are_negative))  ##multi processing
    
def run_baseline(tnet):
    v = 29.06*np.ones(5,)
    this_model_return  = 0
    tnet.start_new_simulation(write_newtrips = False)
    s, simulationSteps, r = tnet.run_step_es(v)
    this_model_return  += r
    while simulationSteps < 18000:
        s_, simulationSteps, r = tnet.run_step_es(v)
        this_model_return  += r
    return this_model_return

                
def gradient_update(argsn, sigma, lr, lr_decay, synced_model, returns, random_seeds, neg_list,
                    num_eps, num_frames, chkpt_dir, unperturbed_results, w_save, step):
    '''
    lr learning rate
    lr learning rate decay
    sigma deviation
    argsn number of seed
    '''
    def fitness_shaping(returns):
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([max(0, math.log(lamb/2 + 1, 2) -
                         math.log(sorted_returns_backwards.index(r) + 1, 2))
                     for r in returns])
        for r in returns:
            num = max(0, math.log(lamb/2 + 1, 2) -
                      math.log(sorted_returns_backwards.index(r) + 1, 2))
            shaped_returns.append(num/denom + 1/lamb)
        return shaped_returns
    
    def unperturbed_rank(returns, unperturbed_results):
        nth_place = 1
        for r in returns:
            if r > unperturbed_results:
                nth_place += 1
        rank_diag = ('%d out of %d (1 means gradient '
                     'is uninformative)' % (nth_place,
                                             len(returns) + 1))
        return rank_diag, nth_place
    
    batch_size = len(returns)
    assert batch_size == argsn
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping(returns)
    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
    print('Episode num: %d\n'
              'Average reward: %f\n'
              'Variance in rewards: %f\n'
              'Max reward: %f\n'
              'Min reward: %f\n'
              'Batch size: %d\n'
              'Sigma: %f\n'
              'Learning rate: %f\n'
              'Total num frames seen: %d\n'
              'Unperturbed reward: %f\n'
              'Unperturbed rank: %s\n'  %
              (num_eps, np.mean(returns), np.var(returns), max(returns),
               min(returns), batch_size, sigma, lr, num_frames,
               unperturbed_results, rank_diag))
    
    averageReward.append(np.mean(returns))
    episodeCounter.append(num_eps)
    maxReward.append(max(returns))
    minReward.append(min(returns))
    pltAvg, = plt.plot(episodeCounter, averageReward, label='average')
    pltMax, = plt.plot(episodeCounter, maxReward,  label='max')
    pltMin, = plt.plot(episodeCounter, minReward,  label='min')

    plt.ylabel('rewards')
    plt.xlabel('episode num')
    plt.legend(handles=[pltAvg, pltMax,pltMin])

    fig1 = plt.gcf()

    plt.draw()
    fig1.savefig('graph60.png', dpi=100)
    
    for i in range(argsn):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]
        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(lr/(argsn*sigma) *
                                      (reward*multiplier*eps)).float()
        lr *= lr_decay
    if w_save == True:
        savename = 'latest' + str(step) + '.pth'
        torch.save(synced_model.state_dict(),
               os.path.join(chkpt_dir, savename))
    return synced_model




def perturb_model(model, random_seed, sigma):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the negative perturbation, and returns both perturbed
    models.
    """
    new_model = ES(model.num_inputs,
                   model.num_outputs, model.hidden_neuron)
    anti_model = ES(model.num_inputs,
                   model.num_outputs, model.hidden_neuron)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v), (anti_k, anti_v) in zip(new_model.es_params(),
                                        anti_model.es_params()):
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(sigma*eps).float()
        anti_v += torch.from_numpy(sigma*-eps).float()
    return [new_model, anti_model]


def generate_seeds_and_models(sigma, synced_model):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2**30)
    two_models = perturb_model(synced_model, random_seed, sigma)
    return random_seed, two_models

def train_loop(max_gradient_updates, sigma, argsn, lr, lr_decay, synced_model, chkpt_dir):
    '''
    max_gradient_updates max iteration steps
    sigma deviation
    argsn number of seed
    lr learning rate
    lr_decay learning rate decay
    '''
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return [item for sublist in notflat_results for item in sublist]
    print("Num params in network %d" % synced_model.count_parameters())
    num_eps = 0
    total_num_frames = 0
    for _ in range(max_gradient_updates):
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models = [], []
        networks = []
        network_1 = rm_vsl_co()
        network_1.writenewtrips()
#        baseline_return = run_baseline(network_1)
        for j in range(int(argsn)):
            networks.append(rm_vsl_co())
        for j in range(int(argsn/2)):
            random_seed, two_models = generate_seeds_and_models(sigma,
                                                                synced_model)
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models += two_models
        assert len(all_seeds) == len(all_models)
        is_negative = True
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            particular_tnet = networks.pop()
            p = mp.Process(target=do_rollouts, args=([perturbed_model],
                                                     [seed],
                                                     return_queue,
                                                     particular_tnet, 
                                                     [is_negative],
                                                     False))
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(all_seeds) == 0
        p = mp.Process(target=do_rollouts, args=([synced_model],
                                                 ['dummy_seed'],
                                                 return_queue, network_1, 
                                                 ['dummy_neg'],
                                                 True))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames, neg_list = [flatten(raw_results, index)
                                                for index in [0, 1, 2, 3]]
        _ = unperturbed_index = seeds.index('dummy_seed')
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        _ = num_frames.pop(unperturbed_index)
        _ = neg_list.pop(unperturbed_index)
        total_num_frames += sum(num_frames)
        num_eps += len(results)
        w_save = False
        if num_eps  % 100 == 0:
            w_save = True
        synced_model = gradient_update(argsn, sigma, lr, lr_decay, synced_model, results, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       chkpt_dir, unperturbed_results, w_save,  num_eps)
        