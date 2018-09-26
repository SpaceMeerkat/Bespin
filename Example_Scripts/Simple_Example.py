#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 16:54:55 2018

@author: SpaceMeerkat
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from Kohonen_SOM import SOM

#Generate data
train_data = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,1],[0,0,0],[1,1,0],[1,0,1],[0,0,1],[0,0,0.8],[0,0,0.91],[0,0,0.95],[0,0,0.949]]) 
          
#Instantiate the SOM classs
som_maker = SOM() 

#Train the SOM
som = SOM.generate_SOM(som_maker,x_size=100,y_size=100,your_data=train_data,initial_radius=100,number_of_iterations=100,initial_learning_rate=0.1)       

#Evaluate data on the trained map
tested_data = SOM.evaluate(som_maker,train_data,som)

               
#Plot the evaluated data 
plt.figure()
for i in tqdm(range(som.shape[1])):
        for  j in range(som.shape[0]):
                plt.plot(j,i,'s',color=som[j,i],markersize=10.0,zorder=0)
plt.plot(tested_data[0],tested_data[1],'k*',zorder=1)
