#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:23:21 2018

@author: SpaceMeerkat
"""

from IPython.display import clear_output
import ipywidgets as widgets
import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.close('all')

class SOM(object):

    def __init__(self, x_size, y_size, data):
        self.x_size = x_size
        self.y_size = y_size
        self.data = data

    def _create_front_widgets(self):
        
        self.initial_radius = widgets.IntSlider(min=0, 
                                                max=max(self.x_size,self.y_size), 
                                                value=max(self.x_size,self.y_size), 
                                                description=r'$\sigma$(t)')
        self.initial_learning_rate = widgets.FloatSlider(min=0.0, 
                                                         max=1.0, 
                                                         value=0.1, 
                                                         description=r'a(t$_{0}$):')
        self.iterations = widgets.IntSlider(min=0, 
                                            max=200, 
                                            value=100, 
                                            description='n:')
        self.train_button = widgets.Button(
            description='Make map',
            disabled=False,
            button_style='success',
            tooltip='Begins the SOM map creation process',
            icon='')
        self.train_button.on_click(self._on_front_click)
        
    def _create_end_widgets(self):
        self.plot_button = widgets.Button(
            description='Show map',
            disabled=False,
            button_style='success',
            tooltip='Shows the trained map',
            icon='')
        self.plot_button.on_click(self._on_end_click)
        
    def Kohonen_Layer(self,x_size,y_size,data):
        k_layer = np.random.uniform(data.min(),data.max(),(x_size,y_size,data.shape[-1]))
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
           
    def generate_SOM(self,initial_radius,number_of_iterations,initial_learning_rate): 
            k_layer = self.Kohonen_Layer(self.x_size,self.y_size,self.data)
            data_length = np.arange(0,self.data.shape[0]-1,1)
            bmu_list = []
            cols_list = []
            rows_list = []
            for i in range(int(number_of_iterations)):
                    random.shuffle(data_length)
                    for j in range(len(data_length)):
                            data_array = self.data[data_length[j]]
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
        
    def evaluate_map(self, test_data, k_layer):
        cols = []
        rows = []
        for i in range(test_data.shape[0]):
                bmu = self.BMU_node(k_layer,test_data[i])
                cols.append(bmu[0])
                rows.append(bmu[1])
        tested_data =  np.vstack([np.hstack(np.array(cols)), np.hstack(np.array(rows))])
        
        clear_output()
        plt.figure(figsize=(3,3))
        for i in range(k_layer.shape[1]):
                for  j in range(k_layer.shape[0]):
                        plt.plot(j,i,'s',color=k_layer[j,i],markersize=15.0,zorder=0)
        plt.plot(tested_data[0],tested_data[1],'k*',zorder=1)

    def _on_front_click(self, change):
        self.out.clear_output()
        with self.out:
            self.k_layer = self.generate_SOM(self.initial_radius.value, 
                                             self.iterations.value, 
                                             self.initial_learning_rate.value)
        self.end_display()
    
    def _on_end_click(self, change):
        self.out.clear_output()
        with self.out:
            self.map = self.evaluate_map(self.data,
                                         self.k_layer)

    def front_display(self):
        self._create_front_widgets()
        self.out = widgets.Output() 
        left = widgets.VBox([self.iterations,
                            self.initial_learning_rate, 
                            self.initial_radius])
        right = widgets.VBox([self.train_button])
        display(widgets.HBox([left,
                              right,
                              self.out]))
        
    def end_display(self):
        self._create_end_widgets()
        self.out = widgets.Output() 
        display(widgets.HBox([self.plot_button]))
    