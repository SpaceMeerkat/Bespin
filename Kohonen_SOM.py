#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:23:21 2018

@author: SpaceMeerkat
"""

import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.close('all')

class SOM:
        
        """
        The SOM class takes input data of particular shape:
                [N,F]
        Where N denotes the number of data objects you have and F denotes the number of features per object.
        Please note that F should remain constant throughout all data objects but can take any finite size.
        
        Running the algorithm in it's simplest form will look as follows:
        SOM.generate_SOM(som_maker,x_size=50,y_size=75,your_data=train_data,initial_radius=100,number_of_iterations=100,initial_learning_rate=0.1)
        
        Where each variable has the following property:
                x_size: length of SOM x dimension 
                y_size: length of SOM y dimension 
                your_data: training data which you present to the algorithm with shape [N,F]
                initial_radius: radius of node infulence at epoch 1 of the training phase
                number_of_iterations: number of epochs during the training phase
                initial_learning_rate: hyperparameter for tuning the intensity of node alteration during training
        """

        def Kohonen_Layer(self,x_size,y_size,your_data):
                k_layer = np.random.uniform(your_data.min(),your_data.max(),(x_size,y_size,your_data.shape[-1]))
                return k_layer
        
        def BMU_node(self,k_layer,data_array):
                flat_w = np.vstack(np.copy(k_layer))
                BMU_val = np.linalg.norm(data_array-flat_w,axis=1).argmin()
                BMU = [int(np.floor(BMU_val/float(k_layer.shape[1]))),int(BMU_val-(np.floor((BMU_val/float(k_layer.shape[1])))*k_layer.shape[1]))]
                return BMU
        
        def lambda_function(self,initial_radius,number_of_iterations):
                return number_of_iterations/np.log(initial_radius)
                
        def radial_decay_function(self,time_step,lambda_value,initial_radius):
                return initial_radius*np.exp(-time_step/lambda_value)
        
        def learning_rate(self,initial_learning_rate,time_step,lambda_value):
                return initial_learning_rate*np.exp(-time_step/lambda_value)
        
        def node_influence(self,node_distances,radius):
                return np.exp((-(node_distances**2))/(2*(radius**2)))
        
        def weight_update(self,k_layer,theta,L,data,weights,weight_rows,weight_cols):
                k_layer[weight_rows,weight_cols] += (theta*L*(data-weights).T).T
                return k_layer
               
        def generate_SOM(self,x_size,y_size,your_data,initial_radius,number_of_iterations,initial_learning_rate): 
                
                self.x_size = x_size
                self.y_size = y_size
                self.your_data = your_data
                self.initial_radius = initial_radius
                self.number_of_iterations = number_of_iterations
                self.initial_learning_rate = initial_learning_rate
                
                k_layer = self.Kohonen_Layer(x_size,y_size,your_data)
                data_length = np.arange(0,your_data.shape[0]-1,1)
                bmu_list = []
                cols_list = []
                rows_list = []
                for i in tqdm(range(int(number_of_iterations))):
                        random.shuffle(data_length)
                        for j in range(len(data_length)):
                                data_array = your_data[data_length[j]]
                                lambda_value = self.lambda_function(initial_radius,number_of_iterations)
                                radius = self.radial_decay_function(i,lambda_value,initial_radius)                
                                BMU = self.BMU_node(k_layer,data_array)
                                bmu_list.append(BMU)
                                indices = np.indices((k_layer.shape[0],k_layer.shape[1])).reshape(2, -1).T
                                nodes_in_r = indices[np.where(np.linalg.norm(BMU-indices,axis=1)<radius)[0]]
                                node_dists = np.linalg.norm(nodes_in_r-BMU,axis=1)
                                rows = nodes_in_r[:,0]
                                cols = nodes_in_r[:,1]
                                cols_list.append(cols)
                                rows_list.append(rows)
                                node_values = k_layer[rows,cols]
                                theta = self.node_influence(node_dists,radius)
                                learn_rate = self.learning_rate(initial_learning_rate,i,lambda_value)
                                k_layer = self.weight_update(k_layer,theta,learn_rate,data_array,node_values,rows,cols)
                return k_layer
        
        def evaluate(self, test_data,k_layer):
                
                self.k_layer = k_layer
                self.test_data = test_data
                
                cols = []
                rows = []
                for i in range(test_data.shape[0]):
                        bmu = self.BMU_node(k_layer,test_data[i])
                        cols.append(bmu[0])
                        rows.append(bmu[1])
                return np.vstack([np.hstack(np.array(cols)), np.hstack(np.array(rows))])
