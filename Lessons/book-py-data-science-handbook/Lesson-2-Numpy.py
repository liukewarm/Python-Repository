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
path = 'pythonscripts/Lessons/book-py-data-science-handbook/PythonDataScienceHandbook-master/notebooks/data/president_heights.csv'    

import pandas as pd
data = pd.read_csv(path)
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

a = a[:, np.newaxis].shape #taking a 1x3 array and making it a 3x1 array to force the broadcasting
np.logaddexp(M, a) #Using another ufunc to demonstrate that it works 

    #Broadcasting in practice: Centering an array
    
X = np.random.random((10,3))
Xmean = X.mean(0) #calculate the mean for each feature/column, 0 is for axis

X_centered = X - Xmean #taking the dataset (10,3) and subtracting from column means (1,3) to calculate z-scores. In this case, we will see it stretch to a 10x3 with same values each row 
X_centered.mean(0) #Check to see if we have a near-zero mean

    #Broadcasting in practice: Plotting two-dimension function: 
    #we defint a function z = f(x,y) and broadcast to compute across the grid
    
x = np.linspace(0,5,50)
y = np.linspace(0,5,50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos( 10 + y * x) * np.cos(x)

%matplotlib inline
import matplotlib.pyplot as plt

plt.imshow(z, origin = "lower", extent = [0,5,0,5],
           cmap ="viridis")
plt.colorbar()

# =============================================================================
# Chapter 6: Comparison, Masks, and Boolean Logic
# Masking comes up when you want to extract, modify, count, or otherwise manipulate values in an array based on some criterion
# =============================================================================

    #Example: Counting rainy days. Imagine you have a series of data the represents
    #the amount of percipitation each day for a year in a given city. For example, 
    # here we'll load the daily rainfall statistics for the city of Seattle in 2014
    
import numpy as np
import pandas as pd

path = 'PythonDataScienceHandbook-master/notebooks/data/Seattle2014.csv'    
rainfall = pd.read_csv(path)['PRCP'].values
inches = rainfall / 254 #1/10mm -> inches
inches.shape
    
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot styles

plt.hist(inches, 40) #Doesn't tell us much about rainy days though, how do we filter

x = np.array([1,2,3,4,5])
x < 3 #check each element and return a boolean array
x > 3
x <= 3
x >= 3
x != 3
x == 3
(2 * x) == (x ** 2) #It is also possible to do an element-by-element comparison of two arrays and to include compoound expressions 

rng = np.random.RandomState(0) #Random generation always produces the same result
x = rng.randint(10, size =(3,4))
x
x < 6 #Example from a two-dimensional array

    #Working with Boolean Arrays

np.count_nonzero( x < 6) #count the number of true entires
np.sum( x < 6) #Same result but different method since false = 0
np.sum( x < 6, axis = 1) #how many values less than 6 in each row

np.any(x > 8) #true if any value is greater than 8
np.any(x < 0) #true if any value is less than 0
np.all(x < 10) #true if each element is less and 10
np.all(x == 6) #true if all elements equal 6
np.all(x < 8, axis = 1) #true if all values in each row is less than 8

    #Boolean Operators: filtering on two or more conditions
    
np.sum((inches > 0.5) & (inches < 1)) #29 days rainfall between 0.5 and 1.0 inches
np.sum(~( (inches <= 0.5) | (inches >= 1) )) #Tilde is NOT operator

print("Number of days without rain: ", np.sum(inches ==0))
print("Number days with rain: ", np.sum(inches != 0))
print("Days with more than 0.5 inches ", np.sum(inches > 0.5))
print("Rainy days with < 0.1 inches ", np.sum((inches > 0) &
                                              (inches < 0.2)))

    #Boolean Arrays as Masks
    
x < 5 #this is our boolean array 
x[x < 5] #we mask the boolean array to index and select elements. one-dimension array returned

rainy = (inches > 0) #Consturcting a mask of all rainy days
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0) #construct a mask of summer days (June 21st is the 172nd day)

print("Median precip on rainy days in 2014 (inches)", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches)", np.median(inches[summer]))
print("Maximum percip on summer days in 2014 (inches)", np.max(inches[summer]))
print("Median percip on non-summer rainy days (inches)", np.median(inches[rainy & ~summer]))

    #Fancy Indexing: passing an array of indicies to access multiple array elements at once 
    
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100,size=10)
x
    
[x[3], x[7], x[2]]
ind = [3,7,4]
x[ind] #Passing a single list or array of indicies to obtain the same result

ind = np.array([[3,7],
                [4,5]])
x[ind] #Shape of the result reflects the shape of the index arrays rather than the shape of the array being indexed

X = np.arange(12).reshape((3,4))
row = np.array([0,1,2])
col = np.array([2,1,3])
X[row,col] #first index refers to the row and second column. it's 1-dim array cause first value in result is X[0,2] second is x[1, 1]. Follows broadcasting rules

X[row[:,np.newaxis], col] #If we combine the column vector and a row vector within the indicies, we get a two-dimensional result
row[:, np.newaxis] * col #this is how each row value is matched with each column vector

    #Combining Indexing
    
X[2, [2,0,1]] #We can combine fancy and simple indices
x[1:,[2,0,1]] #We can also combine fancy indexing with slicing

mask = np.array([1,0,1,0], dtype = bool)
X[row[:,np.newaxis], mask] #Combining fancy indexing with masking


    #Example: selecting Random Points
    
mean =[0,0]
cov = [[1,2],
       [2,5]]
X = rand.multivariate_normal(mean, cov, 100) #One commun use of fancy indexing is the selection of subset of rows from a matrix
X.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #for plotting style

plt.scatter(X[:,0], X[:,1]); #Let's examine our multivariate_normal

indicies = np.random.choice(X.shape[0], 20, replace=False) #get number of rows from X and select 20 random values without replacement
indicies
selection = X[indicies]
selection

plt.scatter(X[:,0], X[:,1], alpha=0.3)
plt.scatter(selection[:,0], selection[:, 1],
            facecolors="None", edgecolors='r', s=200); #Plot results with subset of data select by random
            
    #Modifying Values with Fancy Indexing 

x = np.arange(10)
i = np.array([2,1,8,4])
x[i] = 99
x # everything in fancy index is now 99

x[i] -= 10 #Fancy assignment which takes the each array element of the index and subtracts 10

x = np.zeros(10)
x[[0,0]] = [4,6]
x #x[0] is first assigned 4 then followed and replaced by 6

i = [2,3,3,4,4,4]
x[i] += 1
x #Really means x[i] + 1, i refers the indices not values. and so when we +1 to the same index value (3,4) it just replaces and reassigns

x = np.zeros(10)
np.add.at(x,i,1) #This adds the value 1 add each index i, instead of replacing it like the one above
x

np.random.seed(42)
x = np.random.randn(100)

bins = np.linspace(-5,5, 20) #-5 to 5 with 20 increments of same value
counts = np.zeros_like(bins) #Create an empty array of same size as bins


i = np.searchsorted(bins, x) #finds the appropriate bin and returns and array with index for insertions
np.add.at(counts, i, 1) #using the empty array same size as bins, we +1 to the index(i) array

plt.plot(bins, counts, linestyle='steps'); #plotting the histrogram
np.histogram(x, bins) #this achieve the same thing

# =============================================================================
# Chapter 7: Sorting Arrays
# soritng the values in a list or array
# =============================================================================

import numpy as np

def selection_sort(x):
    for i in range(len(x)):  #object of sequence of into from 0 to 100-1      
        swap = i + np.argmin(x[i:]) #storing a swap index if any of the array values after is smaller
        (x[i], x[swap]) = (x[swap], x[i]) 
        
x = np.array([2,1,4,3,5]) #numpy sort
x.sort() #can also use the array method to sort 
x

x = np.array([2,1,4,3,5]) #arg sort provides the idicies of the sorted elements
i = np.argsort(x)
print(i) #first element gives the index of smallest element, second is index of second smallest element
x[i] #fancy indexing to get our sorted result, too

rand = np.random.RandomState(42)
x=rand.randint(0,10,(4,6)) 
np.sort(x, axis=0) # sorting by column 
np.sort(x, axis=1) #sorting by row


    #Partial Sorts: Partitioning 
    #Takes an array and number K: the result is a new array with smalled K values to the left and remaining to right in arbitrary order
    
x = np.array([7,2,3,1,6,5,4])
np.partition(x,3) #three smalles values to the left, but not in order

X = np.random.randint(0,10,(4,6))
np.partition(X, 2, axis = 0) #partition by column, first two smalles value in the array
np.pariition(X, 2, axis = 1) #partition by row, first 2 smallest values 

    #Example: K-Nearest Neighbours
    #Using argsort function along with multiole axes to find the nearest neighbour for each point in a set

X = np.random.rand(10,2)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #Plot Styling

plt.scatter(X[:,0], X[:,1], s=100);

dist_sq = np.sum( (X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis = 1) #We can compute the matrix of square distances in a single code


    #Decompose into steps: K-nearest neighbours

difference = X[:,np.newaxis,:].shape - X[np.newaxis,:,:].shape #for each pair of points, compute differences in their coordinates
difference.shape # we can see we broadcasting works across rows and columns
sq_differences = difference **2 #Square distances
dist_sq = sq_differences.sum(-1) #Sum the coordinate differences to get the squared distance

nearest = np.argsort(dist_sq, axis=1)
nearest

K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis = 1)




#draw lines from each point to its two nearest neighbours
plt.scatter(X[:, 0], X[:, 1], s=100)

K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        # plot a line from X[i]
        plt.plot(*zip(X[j],X[i]), color='black') #The asterisk just displays the zip object
        
# =============================================================================
# Chapter 8: Structured Arrays
# Storage of compound and hetergenous data
# =============================================================================

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4 ,dtype = int)
data = np.zeros(4, dtype = {'names': ('name', 'age', 'weight'), #name for the different data types and formats using dictionary method
                            'formats':('U10', 'i4', 'f8')})
print(data.dtype)
    
    

data['name'] = name #We can fill out empty array with a list of values
data['age'] = age 
data['weight'] = weight
print(data)   

data['name'] #Now we can refer to arrays by their name
data[0]
data[-1]['name'] #last array element and the name array element

    #Filtering on names
    
data[data['age'] < 30]]['name'] #Get names where age is under 30, returns boolean array, then we fancy index the ones which are true and only grab the 'name' array elements
    
    #Creating Structured Arrays
    
np.dtype({'names': ('name', 'age', 'weight'),
          'formats': ((np.str_,10), int, np.float32)}) #Structured Arrays can be made using NumPy dtypes
    
np.dtype([('name','S10'),('age', '<i4'), ('weight', 'f8')]) #compound type can also be passed as a list of tuples
np.dtype('S10,i4,f8') #IF names don't matter you can pass as comma separated stting 

    #More advanced compound types
    
tp = np.dtype([('id','i8'),('mat', 'f8', (3,3))])
X = np.zeros(1, dtype = tp)
print(X[0])
print(X['mat'][0]) #Each element in the X array consists of an id and a 3x3 matrix

    #RecordArrays: Fields can be accessed as attributes rather than as dicitonary keys
    
data_rec = data.view(np.recarray)
data_rec.age #acessing the record array using attributes instead rather than dicitonary keys, it's a bit slower though 

%timeit data['age'] #it's a bit slower though when done this way...
%timeit data_rec['age']
%timeit data_rec.age

