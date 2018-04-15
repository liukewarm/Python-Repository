# -*- coding: utf-8 -*-
"""
Lesson 3: Data manipulation with Pandas

@author: Liukewarm
"""

import numpy as np
import pandas as pd

#The Pandas Series Object: is a one-dimensional array of indexed data created from a list or array 

data = pd.Series([0.25,0.5,0.75,1.0])
data

data.values #The values are a simply numpy array
data.index #the index is an array-like object of type pd.index

data[1] #Data can be access by the associated index via the familiar Python square-bracket notation:
data[1:3]

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index = ['a','b','c','d']) #Unlike NumPy integers, they do not need to be an integer
data['b'] #Item access can work as expected as well

data = pd.Series([0.25,0.5,0.75,1.0],
                 index = [2,5,3,7])
data[5] #We can even use noncontinguous or nonsequential indicies. Here 5 is not the element index but the index tied to the series

    #Series as a specialized dictionary
    
population_dict = {'California':3334232,
                   'Texas': 43242323,
                   'New York': 43214321,
                   'Florida':432233,
                   'Illinois':443223} #Creating a dictionary object with states as keys and ints as values

population = pd.Series(population_dict) #Passing dictionary into the pandas series
population
population['California'] #Dict-style item access
