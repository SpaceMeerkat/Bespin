# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:16:00 2021

@author: SpaceMeerkat
"""

import ipywidgets as widgets


def choose_params():
    radius = widgets.IntSlider(min=0, max=20, value=10, description='Radius:')
    learning_rate = widgets.FloatSlider(min=0.0, max=1.0, value=0.1, description='Learning rate:')
    iterations = widgets.IntSlider(min=0, max=200, value=100, description='Iterations:')
    
    button = widgets.Button(
    description='Set parameters?',
    disabled=False,
    button_style='success',
    tooltip='Begins the SOM map creation process',
    icon='')
    
    d = widgets.HBox([iterations, learning_rate, radius, button])
    return d