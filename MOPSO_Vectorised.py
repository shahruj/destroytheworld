# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:58:03 2020
@author: Shahruj Rashid
"""
from random import random
from random import uniform
import math
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import numpy as np
import pickle
import random

model2 = pickle.load(open("model.dat", "rb"))

def evaluate_position(temp,online,limit):
    if not online:
	    temp = np.asarray(temp)
	    CNNout=model2.predict(temp.reshape(1,8))[0]
	    CNNout = (CNNout*(limit[1]-limit[0]))+limit[0]
	    return CNNout
    else:
        out = Invoke_Model(temp)[0]
        out = (out*(limit[1]-limit[0]))+limit[0]
        return out 
        

def evaluate_fitness(arr):
    ideal = [100,0]
    fitness = 0
    for i in range(0,len(arr)):
        fitness = fitness + (ideal[i]-arr[i])**2
    return math.sqrt(fitness)    

class Particle:
    def __init__(self,dim,exclude,bounds,x0,online):
        #parameters of the particl    
        self.position_i=np.zeros((1,dim))
        self.velocity_i=np.zeros((1,dim))      
        self.param_1=-1
        self.param_2=-1 
        
        self.bounds = bounds[0:8]
        self.pos_best_i=np.zeros((1,dim)) #pos_best_i contains the individual best position of the particle       
        self.bestparam_1=-1
        self.bestparam_2=-1
        self.exclude = exclude
        self.fitness = math.inf
        self.dimension = dim 
        self.x0 = x0
        self.limit = bounds[7]
        self.is_online = online
        #defines the dimension of the particle where dim = dimension of the search space
        
        for i in range(0,self.dimension): 
            #instantiates the particle according to the search space dimension
            if i not in self.exclude:
                self.velocity_i[0,i]=uniform(-1,1) #assigns a random velocity
                self.position_i[0,i]=random.randint(int(self.bounds[i][0]), int(self.bounds[i][1])) #assigns a random position according to the bounds
            else:
                self.velocity_i[0,i]=0
                self.position_i[0,i]=self.x0[i]                
    def evaluate(self,price): # this function evaluates the particle 
        temp=np.divide(np.subtract(self.position_i,self.bounds[:,0]),np.subtract(self.bounds[:,1],self.bounds[:,0]))
        self.param_2 = np.dot(self.position_i,price.transpose())[0,0]
        self.param_1 = evaluate_position(temp,self.is_online,self.limit) #updates the value(UCS) of the position returned by the model according to input position      
        if ((self.param_1>self.bestparam_1 and self.param_2<self.bestparam_2)): #checks if the current cost and value dominates the personal best (cost and value)
            self.pos_best_i = self.position_i #updates personal best if it dominates 
            self.bestparam_1= self.param_1
            self.bestparam_2 = self.param_2
        elif (self.param_1==self.bestparam_1 and self.param_2==self.bestparam_2): #if the current values don't dominate the personal best, doesn't do anything 
            toss = random.uniform(0, 1) 
            if(toss>=0.5):
                self.pos_best_i = self.position_i
                self.bestparam_1= self.param_1
                self.bestparam_2 = self.param_2
        self.fitness = evaluate_fitness([self.param_1,self.param_2])
        
        
    def instantiate(self,price):
        temp=np.divide(np.subtract(self.position_i,self.bounds[:,0]),np.subtract(self.bounds[:,1],self.bounds[:,0]))
        self.param_2 = np.dot(self.position_i,price.transpose())[0,0]
        self.param_1 = evaluate_position(temp,self.is_online,self.limit) #updates the value(UCS) of the position returned by the model according to input position      
        self.bestparam_1 = self.param_1
        self.bestparam_2 = self.param_2
        self.pos_best_i = self.position_i
        self.fitness = evaluate_fitness([self.param_1,self.param_2])
        
        
        
    def update_velocity(self,paretofront,pareto_index,best_fitness, k): #this function decides the velocity for the next iteration
        inertia = 0.4
        c1 = 3
        c2 = 1.5
        h = 0
        toss=random.uniform(0,1) #does a toss
        if(toss<0.8): #with a 0.8 probability, it chooses from the fittest values in the repository 
            h=random.randint(0, len(best_fitness)-1)
            h=best_fitness[h]
        else: #else it chooses a random value from the repository
            h=random.randint(0, len(pareto_index)-1)
            h=pareto_index[h]
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        v2per_best=c1*r1*np.subtract(self.pos_best_i,self.position_i) #finds vector to the personal best
        v2glo_best=c2*r2*np.subtract(paretofront[h,0:self.dimension],self.position_i) #finds vector to one of the points on the repository decided by the toss                
        for i in self.exclude:
            v2glo_best[:,i]=0
            v2per_best[:,i]=0
        self.velocity_i=inertia*np.add(np.add(self.velocity_i,v2per_best),v2glo_best)
            
    def update_postion(self, k): #updates position according to calculated velocity 
        for i in range(0,self.dimension): 
            self.position_i[0,i]=self.position_i[0,i]+self.velocity_i[0,i]
            #print("position[i]"+str(self.position_i[i])+">="+str(bounds[i][1])+"upp bounds")
            if self.position_i[0,i]>=self.bounds[i,1]: #prevents search space from exceeding the bound (upper limit)
                self.position_i[0,i]=self.bounds[i,1]
            #print("position[i]"+str(self.position_i[i])+">="+str(bounds[i][0])+"low bounds")           
            if self.position_i[0,i]<=self.bounds[i,0]: #prevents search space from exeeding the bound (lower limit)
                self.position_i[0,i]=self.bounds[i,0]
        
np.set_printoptions(suppress=True)
        
def mopso(x0,bounds,n_particles,max_iter,dimension,exclude,cost,is_online):
    paretonum=20
    #last 3 index is param1, param2, and fitness accordingly
    allindex = list(range(paretonum))
    paretofront=np.zeros((paretonum,dimension+3))
    paretofront[:,dimension+2]=math.inf
    best_fitness=[]
    no_of_fitness = 5
    pareto_index=[]
    empty_index=[]
    particle_arr = [] #instantiates the array of particles
    for i in range(0,n_particles): #appends particles accroding to n_particles
        particle_arr.append(Particle(dimension,exclude,bounds,x0,is_online))
        particle_arr[i].instantiate(cost)
        paretofront[i,0:dimension]=particle_arr[i].position_i
        paretofront[i,dimension]=particle_arr[i].param_1
        paretofront[i,dimension+1]=particle_arr[i].param_2
        paretofront[i,dimension+2]=particle_arr[i].fitness
        pareto_index.append(i)
    empty_index=list(set(allindex)-set(pareto_index))
    
    iterate = 0

    while(iterate<=max_iter):
        #plt.scatter(paretofront[pareto_index,8], paretofront[pareto_index,9])
        print("iteration: "+str(iterate))
        # best_fitness = pareto_index
        if(len(pareto_index)>no_of_fitness):
            best_fitness = np.argpartition(paretofront[pareto_index,dimension+2], no_of_fitness)[0:no_of_fitness].tolist()            
        else:
            best_fitness =  pareto_index
        for k in range(0,n_particles): #for loop for each particle 
            particle_arr[k].evaluate(cost) #evaluates the particle
            pop=[]
            Dominated = False
            for i in pareto_index: #checks if the particle dominates any of the point currently on the pareto front 
                if((particle_arr[k].param_1>paretofront[i,8] and particle_arr[k].param_2<paretofront[i,9])):
                    pop.append(i) #keeps track of the dominated indexes on the paretofront
                    Dominated = True            
            empty_index = empty_index + pop
            if(len(empty_index)!=0 and Dominated):
                paretofront[empty_index[0],0:dimension]=particle_arr[k].position_i
                paretofront[empty_index[0],dimension]=particle_arr[k].param_1
                paretofront[empty_index[0],dimension+1]=particle_arr[k].param_2
                paretofront[empty_index[0],dimension+2]=particle_arr[k].fitness  
                pareto_index.append(empty_index[0])
                empty_index.pop(0)  
            particle_arr[k].update_velocity(paretofront,pareto_index,best_fitness, k)
            particle_arr[k].update_postion(k)                                
            pareto_index = [x for x in pareto_index if x not in pop]
        plt.scatter(paretofront[:,8], paretofront[:,9])
        plt.pause(0.05)
        iterate=iterate+1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_title("compressive strength vs cost",fontsize=14)
    ax.set_xlabel("UCS",fontsize=12)
    ax.set_ylabel("cost",fontsize=12)
    ax.grid(True,linestyle='-',color='0.75')
    ax.scatter(paretofront[pareto_index,dimension], paretofront[pareto_index,dimension+1])
    #print(paretofront)
    #plt.show()
    return fig , paretofront
    
    
#x0 is the initial input value

x0 =[527, 243, 101, 166, 29, 948, 870, 28]
#exclude contains the index not to optimise eg 7, means don't optimise index 7 in x0 which stays as 28 throughout 

exclude =[7]
#cost of each of the materials 

cost = np.array([[0.110,0.060,0.055,0.00024,2.940,0.010,0.006,0]])
#the counds of each of the parameters 
bounds = np.array([[102,540],[0,359],[0,200],[121,247],[0,32],[801,1145],[594,992],[1,365]])
#perform mopso wirh model, x0 as input, bounds, 13 paricles, 100 iteration and 8 dimensions
mopso(x0,bounds,10,100,8,exclude,cost,False)   
