# -*- coding: utf-8 -*-
"""
Lesson 2: Working with Numpy

this chapter focuses on working with Numpy and we'll use Starlink as an example
to create a ficticious dataset to work with and use in creating some of our
dashboards 
"""

# =============================================================================
# Chapter 1: Basic Numpy functions



# =============================================================================

import numpy as np

#Knowing that I will have some categorical data, I want to generate 
category_array = np.random.choice(["Category 1","Category 2","Category 3"], size = 100, p = [.2,.7,.1])

#I want 10,000 records and 5 fields, so I will reshape this afterwards
data = np.random.normal(scale = 1.0, size = (10000,5))
data.shape #Checking to see if this is the shape we wanted 

#Other example of numpy arrays 

np.zeros(10, dtype = int) #10 zeros 
np.ones(10, dtype = float) #10 ones with floating decimal
np.full((3,5), 3.14) #Creating a 3x5 (3 rows, 5 columns) Array filled with 3.14
np.arange(0,20,2) #Creating an array filled from 0 to 20, step by 2
np.linspace(0,1,5) #Create an array of five values, evenly spaced between 0 and 1
np.random.random((3,3)) #Create a 3x3 array of uniformly distributed values between 0 and 1
np.random.normal(0,1 (3,3)) #Create a 3x3 array of normally distributed random values, with mean of 0 and STD of 1 
np.random.randint(0,10,(3,3)) #Create a 3x3 array of random integers in the interval of 0 - 10 
np.eye(3) #Create a 3x3 identity matrix

# =============================================================================
# Chapter 2: Basics of NumPy Array
# Attributes of arrays: Determining the size, shape, memory consumption and data types of arrays
# Indexing of arrays: Getting and setting the value of individual array elements
# Slicing of arrays: Getting and setting smaller subarrays within a larger array
# Reshaping of arrays: Changing the shape of a given array
# Joining and splitting of arrays: Combining multiple arrays into one, and splitting one array into many 
# =============================================================================

np.random.seed(0) #seed for reproducibility and to ensure random results are the same

x1 = np.random.randint(10, size = 6) #One-dimensional array
x2 = np.random.randint(10, size = (3,4)) #Two-dimensional array
x3 = np.random.randint(10, size = (3,4,5)) #Three-dimensional array

print("x3 dim", x3.ndim)
print("x3 shape", x3.shape)
print("x3 size", x3.size)
print("x3 dtype", x3.dtype)
print("item size", x3.itemsize, "bytes") # Size of each array element
print("nbytes:", x3.nbytes, "bytes") #total size of the whole array (itemsize times size)

#Indexing from the array (1-dimension)

x1
x1[0]
x1[4] #one dimension is similar to a list
x1[-1] #starts from the end of the array

#Indexing from the array (2-dimension)

x2 #Access using a comma-separated typle of indices
x2[0,0] #references the first array row and first array element
x2[0,-1] #references first array row and last element

#Slicing: Accessing sub-arrays x[start:stop:step]

x = np.arange(10)
x[:5] #first five elements
x[5:] #elements after the 5 index
x[4:7] #middle subarray
x[::2] #Every other element
x[1::2] #Every other element starting at index 1

x[::-1] #all elements are reversed
x[5::-2] #reversed every other element from index 5

    #Multi-dimensional subarray slices

x2
x2[:2,:3] #first argument in tuple is slice of rows and second argumen is slice of columns
x2[:3,::2] #All rows and evey other column
print(x2[:,0]) #First column of array
print(x2[0,:]) #First row of x2
print(x2[0]) #Same result as above

    # Sub-arrays as no-copy views: slices return views no copies of array data
    
print(x2)
x2_sub = x2[:2,:2] #Grabbing a 2x2 sub-array
print(x2_sub)
x2_sub[0,0] = 99 #Changing the first row array element to 99
x2 #If we look at the original variable "x2" we can see that it has changed!!!
x2_sub_copy = x2[:2,:2].copy() #To get around this we need to explicitly copy it to store it into a new object
x2_sub_copy[0,0] = 42 #Changing the first row array element to 42
x2_sub_copy #we can see that it's changed
x2 #our original object, however, remains the same

#Reshaping of arrays

grid = np.arange(1,10).reshape(3,3) #changed a 1-dimension array into 3x3 two-dimension array
x[:,np.newaxis] #new turns a one-dimensional array into a column vector, from a row vector


#Joining of arrays:

    #Concatenation of arrays
    
x = np.array([1,2,3])
y = np.array([3,2,1])
z = np.full(3,99)
np.concatenate([x,y]) #concatenating two object arrays
np.concatenate([x,y,z]) #concatenating three object arrays

grid = np.array([[1,2,3],
                 [4,5,6]])

np.concatenate([grid,grid]) #can also be done for two-dimensional arrays
np.concatenate([grid,grid], axis =1) # concatenate along the second axis

    #Concatenation with mixed dimensions is best done using vstack and hstack
    
x = np.array([1,2,3])
grid = np.array([[9,8,7],
                 [6,5,4]])

np.vstack([x,grid]) #appends by row (sort of like union), must have same number of cols

x = np.array([[99],
              [99]])

np.hstack([x,grid]) #appends as cols

#Splitting of arrays: pass a list of indicies giving the points of splits

x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3,5]) #splits into 3 objects and first 0 - 2, 3-4 and 5 to the end
print(x1,x2, x3) #N split points will always lead to N+1 sub arrays

    #vsplit and hsplit

grid = np.arange(16).reshape((4,4))
grid

upper, lower = np.vsplit(grid, [2]) #Splits into 2 groups with 2 records/rows
upper
lower

left, right = np.hsplit(grid, [2])#Splits into two groups by column 

# =============================================================================
# Chapter 3: Vectorized functions
# Vectorized functions or UFUNCs in Numpy make calculations faster and execute repeated
# Operations on values of NumpPy Array
# =============================================================================

#How slow are non-vectorized? Problem is with loops

import numpy as np

np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1,10, size = 5)
compute_reciprocals(values)

big_array = np.random.randint(1,100, size = 1000000)    
%timeit compute_reciprocals(big_array) #Checking the time to complete and it ain't good
%timeit (1.0 / big_array) #Checking the time of the vectorized function, and it is way faster

    #Multi-Dimensional UFuncs
    
x = np.arange(9).reshape((3,3)) #Creating a 3x3 two-dim matrix
2 ** x #Sqaure each array element


    #Array Arithmetics: standard addition, subtraction, multiplication and division

x = np.arange(4)

print("x   =", x)
print("x + 5 =",x + 5)
print("x - 5 =",x - 5)
print("x * 2 =",x * 2)
print("x / 2 =",x / 2)
print("x // 2 =", x // 2)

print("-x =", -x) #negation
print("x ** 2 =", x**2) #exponentiation
print("x % 2 =", xx % 2) #Modulus (remainder, 1 if true)
-(0.5*x + 1)**2 #We can also string these together

np.add(x,2) #operations are just convenient wrappers for numpy functions
np.subtract(x,-2)
np.power(x,2)

x = np.array([-2,-1,0,1,2])
abs(x) #also understands python built-in functions as well 
np.absolute(x) #This is the corresponding NumPy unfunc
np.abs(x)#same as above

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 +1j])
np.abs(x) #returns the magnitude

theta = np.linspace(0, np.pi, 3) #Create an array of three elements evenly spaced between 0 and 3.14

    #Trigonometric functions 
    
print("theta =", theta)
print("sin(theta) =", np.sin(theta))
print("cos(theta) =", np.cos(theta))
print("tan(theta) =", np.tan(theta))

    #Exponents and logarithms
    
x = np.arange(1,4)
print("x =", x)
print("e^x =", np.exp(x)) #This is the base value (e) to the power of each element in x
print("2^x =", np.exp2(x)) #This is the base value of 2 to the power of each element in x
print("3^x =", np.power(3,x)) #This takes a base value provided as arg to the power of each element in x

x = [1,2,4,10]
print("x =", x)
print("ln(x) =", np.log(x)) #the base of e (2.71...) needs to be raised to a power of (i) to get each of the array element values
print("log2(x) =", np.log2(x)) #the base of 2 which needs to be raised to a power of (i) to equal each array element value
print("log10(x) =", np.log10(x)) #the base of 10 which needs to be raised to a power of (i) to equal each array element value

#Advanced Ufuncs and best practices: for large calculations it is best to be able to specify where the results will be stored

x = np.arange(5)
y = np.empty(5) #this is a placeholder location in memory to store results
np.multiply(x,10, out = y)
y

y = np.zeros(10)
np.power(2, x, out =y[::2])
y # demonstrating how this can be used for array views as well

# =============================================================================
# Chapter 4: Aggregations 
# =============================================================================

x = np.arange(1,6)
np.add.reduce(x) #Reduce function repeatedly applies a given operation to the elements of an array (recursive)
np.multiply.reduce(x) #we can recursively apply reduce to any ufunc from numpy, awesome!

np.add.accumulate(x) #This keeps a cumulative total
np.multiply.accumulate(x)

np.prod(x) #This is the same as using a np function and reduce method
np.cumprod(x)#This is the same as using the np function and accumulate
np.sum(x)
np.cumsum(x)

big_array = np.random.rand(1000000)
%timeit sum(big_array) #Using built in functions is much slower
%timeit np.sum(big_array) 

%timeit np.min(big_array) #Min and max numpy functions, again faster with np
%timeit min(big_array)
%timeit np.max(big_array)
%timeit max(big_array)

big_array.min() #You could also use the method of an array itself

M = np.random.random((3,4))
M
M.sum() #aggregation occurs for all columns and rows
M.min(axis=0) #aggregation occurs for the columns
M.min(axis=1) #aggregation occurs for the rows

%pwd #Checking current director
%cd #let's go back to our beginning directory
cwd = 'pythonscripts/Lessons/book-py-data-science-handbook/PythonDataScienceHandbook-master/notebooks/data/president_heights.csv'    

import pandas as pd
data = pd.read_csv(cwd)
heights = np.array(data['height(cm)']) #selecting only heights from the dataset
print(heights)

    
print("Mean height:", heights.mean())
print("Standard Deviation:", heights.std())
print("Minimum Height:", heights.min())
print("Maximum height:", heights.max())

print("25th Percentile:",np.percentile(heights,25)) #array doesn't have this method so we call np
print("Median:", np.median(heights))
print("75th Percentile:",np.percentile(heights,75))

    #Visualization of the aggregate data

%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn; seaborn.set() #set plot style

plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('Height (cm)')
plt.ylabel('number');


# =============================================================================
# Chapter 5: Computing on Arrays
# Broadcasting: simply a set of rules for applying ufuncs (+,-,*,etc) on arrays of different sizes
# =============================================================================

    #Broadcasting: simple example
import numpy as np    
    
a = np.array([0,1,2])
b = np.array([5,5,5])
a + b #Elements of the same size are performed on an element-by-element basis 

a + 5 #using a scalar to get the same result
      #Think of it as duplicating 5 into the array [5,5,5] and adds the results. Duplication actually does not take place but useful mental model for conceptualization
      
    #Broadcasting: one-dimension to two-dimension 
    
M = np.ones((3,3))
M
M + a #here the one-dimensional array is stretched or broadcast across the second dimension in order to match shape of M
      #Since they both have 3 columns but M has 3 rows and a has 1, a "broadcasts" row 1 to rows 2 & 3
      
a = np.arange(3)
b = np.arange(3)[:,np.newaxis]
a
b
a + b #a needs to create two more rows to match dimensionality of b (duplicating the first), and b needs to create two more columns (duplicating first col)

    #Broadcasting - Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is stretched to match other shape

M = np.ones((2,3))
a = np.arange(3)
print(M.shape)
print(a.shape) #This is missing a dimension cause it is a 1-dimensional array so we need to add one a.shape(1,3)

#a.Shape -> (1,3) equals 1 row with 3 columns 
#M.shape -> (2,3) equals 2 rows with 3 columns, the dimensional structure matches but still disagrees cause M has one more row 

# a.shapre -> (2,3) equals 2 rows with 3 columns by duplicating first row to match number in that dimension of M
# M.shape -> (2,3) 

M + a #Okay this worked nicely! 

    #Broadcasting example 2: Both arrays need to be broadcast
    
a = np.arange(3).reshape((3,1)) 
b = np.arange(3)
a.shape #Don't need to add dimensions but need to stretch to 3 columns
b.shape #need to add an extra dimension and make stretch 1 row into 3

a + b

    #Broadcasting example 3: incompatiple arrays
    
M = np.ones((3,2))
a = np.arange(3)

M + a #Throws an exception

a[:, np.newaxis].shape






