#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:23:21 2018

@author: SpaceMeerkat
"""
# =============================================================================
# Import relevant packages
# =============================================================================

from IPython.display import clear_output
import ipywidgets as widgets
import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Setup SOM class
# =============================================================================

class SOM(object):
    
    """
    Trains a Kohonen self organising map
    
    :class KinMSError:
        Instantiates the class 'SOM', for training self organising maps given
        specified input data.
    """

    def __init__(self, x_size, y_size, data):
        """
        Instantiates the SOM with x and y dimensional sizes and training data.
        
        :param x_size:  
            (int) x-axis size for the resultant map
        :param y_size:  
            (int) y-axis size for the resultant map
        :param data:  
            (array) N-dimensional vectors for training the map 

        """
        
        self.x_size = x_size
        self.y_size = y_size
        self.data = data

    def _create_front_widgets(self):
        """
        Creates the front end UI widgets landing.

        :return: 
            (class attributes) UI landing widgets

        """
        
        self.initial_radius = widgets.IntSlider(min=0, 
                                                max=max(self.x_size,self.y_size), 
                                                value=max(self.x_size,self.y_size), 
                                                description=r'$\sigma(t_{0})$:')
        self.initial_learning_rate = widgets.FloatSlider(min=0.0, 
                                                         max=1.0, 
                                                         value=0.1, 
                                                         description=r'$a(t_{0})$:')
        self.iterations = widgets.IntSlider(min=0, 
                                            max=200, 
                                            value=100, 
                                            description='$N$:')
        self.train_button = widgets.Button(
            description='Make map',
            disabled=False,
            button_style='success',
            tooltip='Begins the SOM map creation process',
            icon='')
        self.train_button.on_click(self._on_front_click)
        
    def _create_end_widgets(self):
        """
        Creates the UI widgets landing having trained a map.

        :return: 
            (class attributes) UI landing widgets

        """
        
        self.plot_button = widgets.Button(
            description='Show map',
            disabled=False,
            button_style='success',
            tooltip='Shows the trained map',
            icon='')
        self.plot_button.on_click(self._on_end_click)
        
    def Kohonen_Layer(self,x_size,y_size,data):
        """
        Instantiates a 2-dimensional map with uniformly sampled vectors in the
        interval [-1,1].
        
        :param x_size:  
            (int) x-axis size for the resultant map
        :param y_size:  
            (int) y-axis size for the resultant map
        :param data:  
            (array) N vectors of any length for training the map

        :return: 
            (array) 2-dimensional kohonen layer

        """
        
        k_layer = np.random.uniform(data.min(),data.max(),(x_size,y_size,data.shape[-1]))
        return k_layer
        
    def BMU_node(self,k_layer,data):
        """
        Calculates the xy positional indices of an input vectors Best Matching
        Unit (BMU).
        
        :param k_layer:  
            (array) 2-dimensional kohonen layer 
        :param data:  
            (array) A single N-dimensional input vector 

        :return: 
            (list) list of length=2, giving the x and y positions of the BMU

        """
        
        flat_w = np.vstack(np.copy(k_layer))
        BMU_val = np.linalg.norm(data-flat_w,axis=1).argmin()
        BMU = [int(np.floor(BMU_val/float(k_layer.shape[1]))),int(BMU_val-(np.floor((BMU_val/float(k_layer.shape[1])))*k_layer.shape[1]))]
        return BMU
    
    def lambda_function(self,initial_radius,number_of_iterations):
        """
        Calculates lambda, for use in the radial and learning rate decays as a
        function of the timestep.
        
        :param initial_radius:  
            (int or float) initial radius around the BMU within which, nodes 
            are updated based on the input vector and proximity to the BMU
        :param number_of_iterations:  
            (int) The number of epochs with which to train the map 

        :return: 
            (float) lambda, the time-dependent scaling factor for weight decay

        """
        
        return number_of_iterations/np.log(initial_radius)
            
    def radial_decay_function(self,time_step,lambda_value,initial_radius):
        """
        Decays the size of the zone of influence around the BMU given a 
        timestep. 
        
        :param time_step:  
            (int) The current time step in the interval [0,T]
        :param initial_radius:  
            (int or float) initial radius around the BMU within which, nodes 
            are updated based on the input vector and proximity to the BMU
        :param lambda_value:  
            (float) The time-dependent scaling factor for weight decay 

        :return: 
            (float) radial distance about the BMU for weight updates

        """
        
        return initial_radius*np.exp(-time_step/lambda_value)
    
    def learning_rate(self,initial_learning_rate,time_step,lambda_value):
        """
        Decays the *learning rate*, a hyperparameter for decaying the weight
        update influence on BMU neighbouring nodes with time.
        
        :param initial_learning_rate:  
            (float) initial value for the hyperparameter of weight update 
            influence on BMU neighbouring nodes
        :param time_step:  
            (int) The current time step in the interval [0,T]
        :param lambda_value:  
            (float) The time-dependent scaling factor for weight decay 

        :return: 
            (float) learning rate at current timestep

        """
        
        return initial_learning_rate*np.exp(-time_step/lambda_value)
    
    def node_influence(self,node_distances,radius):
        """
        Calculates the influence of nodes, given their proximity to the BMU, as
        weights.
        
        :param node_distances:  
            (array) Euclidean distances of neighbouring nodes to the BMU
        :param radius:  
            (float) The distance from the BMU dictating node influence given a 
            radial decay function

        :return: 
            (array) neighbourhood influence weights

        """
        
        return np.exp((-(node_distances**2))/(2*(radius**2)))
    
    def weight_update(self,k_layer,theta,learning_rate,data,weights,weight_rows,weight_cols):
        """
        Updates the weights neighbouring the BMU, given input vectors.
        
        :param k_layer:  
            (array) 2-dimensional kohonen layer
        :param learning_rate:  
            (float) value for the hyperparameter of weight update influence on 
            BMU neighbouring nodes
        :param data:  
            (array) A single N-dimensional input vector 
        :param weights:  
            (array) vectors of the BMU neighbouring nodes selected for weight 
            update
        :param weight_rows:  
            (list) row indices of BMU neighbouring nodes selected for weight 
            update
        :param weight_cols:  
            (list) column indices of BMU neighbouring nodes selected for weight 
            update

        :return: 
            (array) 2-dimensional kohonen layer

        """
        
        k_layer[weight_rows,weight_cols] += (theta*learning_rate*(data-weights).T).T
        return k_layer
           
    def generate_SOM(self,initial_radius,number_of_iterations,initial_learning_rate): 
        """
        A forward process which loops over training data and a specified number
        of timesteps, updating weights and training the map.
        
        :param initial_radius:  
            (int or float) initial radius around the BMU within which, nodes 
            are updated based on the input vector and proximity to the BMU
        :param number_of_iterations:  
            (int) The number of timesteps over which to train the map 
        :param initial_learning_rate:  
            (float) initial value for the hyperparameter of weight update 
            influence on BMU neighbouring nodes

        :return: 
            (array) 2-dimensional kohonen layer

        """
        
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
                        k_layer = self.weight_update(k_layer,
                                                     theta,
                                                     learn_rate,
                                                     data_array,
                                                     node_values,
                                                     rows,
                                                     cols)

        return k_layer
        
    def evaluate_map(self, test_data, k_layer):
        """
        Given a trained map, finds the BMUs for input data and plots them on
        the trained map.
        
        :param test_data:  
            (array) N-dimensional vectors for evaluating the map 
        :param k_layer:  
            (array) 2-dimensional kohonen layer

        :return: 
            None

        """
        
        cols = []
        rows = []
        for i in range(test_data.shape[0]):
                bmu = self.BMU_node(k_layer,test_data[i])
                cols.append(bmu[0])
                rows.append(bmu[1])
        tested_data =  np.vstack([np.hstack(np.array(cols)), np.hstack(np.array(rows))])
        
        clear_output()
        plt.figure(figsize=(5,5))
        for i in range(k_layer.shape[1]):
                for  j in range(k_layer.shape[0]):
                        plt.plot(j,i,'s',color=k_layer[j,i],markersize=15.0,zorder=0)
        plt.plot(tested_data[0],tested_data[1],'r*',zorder=1)
        plt.axis('off')
        plt.xlim(-0.5,self.x_size-0.5); plt.ylim(-0.5,self.y_size-0.5)
        return 

    def _on_front_click(self, change):
        """
        Controls the map generation through landing widget button clicks.

        :return: 
            (class attributes) UI post-training landing widgets

        """
        
        self.out.clear_output()
        with self.out:
            self.k_layer = self.generate_SOM(self.initial_radius.value, 
                                             self.iterations.value, 
                                             self.initial_learning_rate.value)
        self.end_display()
        return
    
    def _on_end_click(self, change):
        """
        Controls the map evaluation through landing widget button clicks.

        :return: 
            None

        """
        
        self.out.clear_output()
        with self.out:
            self.map = self.evaluate_map(self.data,
                                         self.k_layer)
            return

    def front_display(self):
        """
        Structures the UI landing widgets prior to map training.

        :return: 
            (ipywidgets object) Pre-training landing UI

        """
        
        self._create_front_widgets()
        self.out = widgets.Output() 
        left = widgets.VBox([self.iterations,
                            self.initial_learning_rate, 
                            self.initial_radius])
        right = widgets.VBox([self.train_button])
        display(widgets.HBox([left,
                              right,
                              self.out]))
        return
        
    def end_display(self):
        """
        Structures the UI landing widgets post map training.

        :return: 
            (ipywidgets object) Post-training landing UI

        """
        
        self._create_end_widgets()
        self.out = widgets.Output() 
        display(widgets.HBox([self.plot_button]))
        return

# =============================================================================
# End of class
# =============================================================================

# =============================================================================
# End of script
# =============================================================================
