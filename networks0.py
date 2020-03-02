#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:24:19 2018

@author: wuyuankai
"""

from __future__ import division
import os
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "/home/tianyushi/sumo_binaries/bin/sumo"
sumoConfig = "floating_car.sumo.cfg"
import traci

class rm_vsl_co(object):
    '''
    this is a transportation network for training multi-lane variable speed limit and ramp metering control agents
    the simulation is running on sumo
    '''
    def __init__(self, test = False, visualization = False, control_horizon = 60, incident_time = 0, incident_len = 0):
        
        '''
        OD Parameters
        '''
        self.m1flow = np.round(np.array([359+640,6007+1229,5349+2080,5563+1139,5299+1107])) ##inflow mainlane
        self.r3flow = np.round(np.array([480,1153,1129,1176,1095])) ##ramp flow
        self.m1a = [0.75,0.25]
        self.v_ratio = [0.1,0.1,0.4,0.4]
        
        
        '''
        Network Parameters
        '''
        self.edges = ['m3 m4 m5 m6 m7 m8 m9',\
                 'm3 m4 m5 m6 m7 m8 rout1',\
                 'rlight1 rin3 m7 m8 m9']
        self.control_section = 'm6'
        self.state_detector = ['m5_0loop','m5_1loop','m5_2loop','m5_3loop','m5_4loop','m5_5loop',\
                               'm7_0loop','m7_1loop','m7_2loop','m7_3loop','m7_4loop','m7_5loop']
        self.VSLlist = ['m6_0','m6_1','m6_2','m6_3','m6_4']
        self.inID = ['m3_0loop','m3_1loop','m3_2loop','m3_3loop','m3_4loop','rlight1_0loop']
        self.outID = ['m9_0loop','m9_1loop','m9_2loop','m9_3loop','m9_4loop','rout1_0loop']
        self.targetlink = 'm7'
        self.incident_v = ''

        '''
        Simulation Parameters
        '''
        self.simulation_hour = 5  #hours
        self.simulation_step = 0
        self.control_horizon = control_horizon  #seoncs
        self.test = test
        self.visualization = visualization
        self.incident_time = incident_time
        self.incident_len = incident_len
        if self.visualization == False:
            self.sumoBinary = "/home/tianyushi/sumo_binaries/bin/sumo"
        else:
            self.sumoBinary = "/home/tianyushi/sumo_binaries/bin/sumo-gui"

### to do change type
    def writenewtrips(self): 
        with open('fcd.rou.xml', 'w') as routes:
            routes.write("""<?xml version="1.0"?>""" + '\n' + '\n')
            routes.write("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">""" + '\n')
            routes.write('\n')
            routes.write("""<vType id="type0" color="255,105,180" length = "8.0"/>""" + '\n')
            routes.write("""<vType id="type1" color="255,190,180" length = "8.0" carFollowModel = "IDM"/>""" + '\n')
            routes.write("""<vType id="type2" color="22,255,255" length = "3.5"/>""" + '\n')
            routes.write("""<vType id="type3" color="22,55,255" length = "3.5" carFollowModel = "IDM"/>""" + '\n')
            routes.write('\n')
            for i in range(len(self.edges)):
                routes.write("""<route id=\"""" + str(i) + """\"""" + """ edges=\"""" + self.edges[i] + """\"/> """ + '\n')
            temp = 0
            for hours in range(len(self.m1flow)):
                m_in = np.random.poisson(lam = int(self.m1flow[hours,]))
                r3_in = np.random.poisson(lam = int(self.r3flow[hours,]))
                vNum = m_in + r3_in
                dtime = np.random.uniform(0+3600*hours,3600+3600*hours,size=(int(vNum),))            
                dtime.sort()
                for veh in range(int(vNum)):
                    typev = np.random.choice([0,1,2,3], p = self.v_ratio)
                    vType = 'type' + str(typev)
                    route = np.random.choice([0,1,2], p =[m_in*self.m1a[0]/vNum, m_in*self.m1a[1]/vNum, r3_in/vNum])
                    if route == 2:
                        lane = 0
                    else:
                        lane = np.random.choice([0,1,2,3,4], p =[0.2,0.2,0.2,0.2,0.2])
                    routes.write("""<vehicle id=\"""" + str(temp+veh) + """\" depart=\"""" + str(round(dtime[veh],2)) + """\" type=\"""" + str(vType) + """\" route=\"""" + str(route) + """\" departLane=\""""+ str(lane) + """\"/>""" + '\n')        
                    routes.write('\n')
                temp+=vNum
            routes.write("""</routes>""")
            
    #####################  obtain state  #################### 
    def get_step_state(self):
        state_occu = []
        for detector in self.state_detector:
            occup = traci.inductionloop.getLastStepOccupancy(detector)
            if occup < 0:
                occup = -1
            state_occu.append(occup)
        return np.array(state_occu)
    
    #####################  set speed limit  #################### 
    def set_vsl(self, v):
        number_of_lane = len(self.VSLlist)
        for j in range(number_of_lane):
            traci.lane.setMaxSpeed(self.VSLlist[j], v[j])
            
    #####################  the out flow ####################         
    def calc_outflow(self):
        state = []
        statef = []
        for detector in self.outID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            state.append(veh_num)
        for detector in self.inID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            statef.append(veh_num)
        return np.sum(np.array(state)) - np.sum(np.array(statef)), np.sum(np.array(state))
    
    ##################### number of emergecncy braking ##############
    def calc_number_eb(self, vidolist, v):
        vidlist = traci.edge.getLastStepVehicleIDs(self.control_section)
        vn = []
        for vid in vidlist:
            vn.append(traci.vehicle.getSpeed(vid))
        vno = []
        for vido in vidolist:
            vno.append(traci.vehicle.getSpeed(vido))
        da = np.array(vno) - np.array(v)
        eb = len(da[da<=-4.5])
        return vidlist,vn,eb
    
    #####################  the bottleneck speed #####################
    def calc_bottleneck_speed(self):
        return traci.edge.getLastStepMeanSpeed(self.targetlink)
    
    #####################  the CO, NOx, HC, PMx emission  #################### 
    def calc_emission(self):
        vidlist = traci.edge.getIDList()
        co = []
        hc = []
        nox = []
        pmx = []
        for vid in vidlist:
            co.append(traci.edge.getCOEmission(vid))
            hc.append(traci.edge.getHCEmission(vid))
            nox.append(traci.edge.getNOxEmission(vid))
            pmx.append(traci.edge.getPMxEmission(vid))
        return np.sum(np.array(co)),np.sum(np.array(hc)),np.sum(np.array(nox)),np.sum(np.array(pmx))
    
    #####################  a new round simulation  #################### 
    def start_new_simulation(self, write_newtrips = True, incident_time = 0, incident_len = 0):
        self.simulation_step = 0
        if write_newtrips == True:
            self.writenewtrips()
        sumoCmd = [self.sumoBinary, "-c", sumoConfig, "--start", "--no-warnings",  "--no-step-log"]
        traci.start(sumoCmd)
        self.incident_time = incident_time
        self.incident_len = incident_len
        
    #####################  run one step: the process is for evolutionaray strategy ##########
    def run_step_es(self, v):
        state_overall = 0
        outflow_s = 0
        self.set_vsl(v)
        for i in range(self.control_horizon):
            traci.simulationStep()
            state_overall = state_overall + self.get_step_state()
            inout, outflow = self.calc_outflow()
            outflow_s += outflow
            self.simulation_step += 1
            if self.simulation_step == self.incident_time:
                vehid = traci.vehicle.getIDList()
                r_tempo = np.random.randint(0, len(vehid) - 1)
                self.inci_veh = vehid[r_tempo]
            if self.simulation_step > self.incident_time and self.simulation_step < self.incident_time + self.incident_len and self.inci_veh in traci.vehicle.getIDList():
                traci.vehicle.setSpeed(self.inci_veh, 0)
        return state_overall/100/self.control_horizon, self.simulation_step, outflow_s
      
    #####################  run one step: reward is outflow  #################### 
    def run_step(self, v, vidold, vehicle_speed_old):
        state_overall = 0
        rewardio = 0
        rewardsp = 0
        rewardsa = 0
        rewardem = 0
        co = 0
        hc = 0
        nox = 0
        pmx = 0
        self.set_vsl(v)
        for i in range(self.control_horizon):
            traci.simulationStep()
            state_overall = state_overall + self.get_step_state()
            inout, outflow = self.calc_outflow()
            rewardio = rewardio + inout # the reward is defined as the outflow 
            rewardsp = rewardsp + self.calc_bottleneck_speed()
            co_temp, hc_temp, nox_temp, pmx_temp = self.calc_emission()
            co = co + co_temp/1000 # g
            hc = hc + hc_temp/1000 # g
            nox = nox + nox_temp/1000 #g
            pmx = pmx + pmx_temp/1000
            vidold, vehicle_speed_old,eb = self.calc_number_eb(vidold, vehicle_speed_old)
            rewardsa = rewardsa - eb
            rewardem = rewardem - (co/1.5 + hc/0.13 + nox/0.04 + pmx/0.01)/1000
            self.simulation_step += 1
            if self.simulation_step == self.incident_time:
                vehid = traci.vehicle.getIDList()
                r_tempo = np.random.randint(0, len(vehid) - 1)
                self.inci_veh = vehid[r_tempo]
            if self.simulation_step > self.incident_time and self.simulation_step < self.incident_time + self.incident_len and self.inci_veh in traci.vehicle.getIDList():
                traci.vehicle.setSpeed(self.inci_veh, 0)
        return state_overall/12, rewardio/70*0.15 + rewardsp/1800*0.7 + rewardsa/30*0 + rewardem/80*0.15, self.simulation_step, co, hc, nox, pmx, vidold, vehicle_speed_old #s 60 #e 30
    
    def run_step_test(self, v, vidold, vehicle_speed_old):
        state_overall = 0
        rewardio = 0
        rewardsp = 0
        rewardsa = 0
        rewardem = 0
        outflow_s = 0
        co = 0
        hc = 0
        nox = 0
        pmx = 0
        self.set_vsl(v)
        for i in range(self.control_horizon):
            traci.simulationStep()
            state_overall = state_overall + self.get_step_state()
            inout, outflow = self.calc_outflow()
            outflow_s += outflow
            rewardio = rewardio + inout # the reward is defined as the outflow 
            rewardsp = rewardsp + self.calc_bottleneck_speed()
            co_temp, hc_temp, nox_temp, pmx_temp = self.calc_emission()
            co = co + co_temp/1000 # g
            hc = hc + hc_temp/1000 # g
            nox = nox + nox_temp/1000 #g
            pmx = pmx + pmx_temp/1000
            vidold, vehicle_speed_old,eb = self.calc_number_eb(vidold, vehicle_speed_old)
            rewardsa = rewardsa - eb
            rewardem = rewardem - (co/1.5 + hc/0.13 + nox/0.04 + pmx/0.01)/1000    
            self.simulation_step += 1
        return state_overall/12, rewardio, rewardsp, rewardsa, rewardem, self.simulation_step, co, hc, nox, pmx, vidold, vehicle_speed_old, outflow_s
    
    def close(self):
        traci.close()