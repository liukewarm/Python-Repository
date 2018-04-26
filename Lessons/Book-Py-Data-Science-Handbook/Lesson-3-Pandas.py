# -*- coding: utf-8 -*-
"""
Lesson 3: Data manipulation with Pandas

@author: Liukewarm
"""

import numpy as np
import pandas as pd

# =============================================================================
# Chapter 1: The Pandas Series Object
# A one-dimensional array of indexed data created from a list or array  
# =============================================================================

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
population['California':'Illinois'] #array-style slicing but with series index

    #Constructing series objects: We've already seen a few ways of constructing Pandas Series from scratch
    
pd.Series([1,2,3]) #data can be a list and index defaults to an integer sequence
pd.Series(5, index=[100, 200, 300]) #data can be a scalar, which is repeated to fill the index
pd.Series({2:'a', 1:'b', 3:'c'}) #dictionary keys are the default values
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3,2]) #The Series is populated only with the explicitly identified keys

# =============================================================================
# Chapter 2: Pandas DataFrame Object
# Dataframe as sequence of aligned Series objects. They share the same index 
# =============================================================================

area_dict = {'California': 1234, 'New York': 32434, 'Texas': 9384,
             'Florida': 75783, 'Illinois': 32134}
area = pd.Series(area_dict)

states = pd.DataFrame({'Population': population,
                       'area': area}) #Keys are the column titles and values are the PD series. values are joined by index not position
states.index #Index object holding the column labels
states.columns #Index object holding the column labels, Like a two-dimensional array where both rows and columns have index for accessing data 

states['area'] #attribute area returns a series object

    #Constructing DataFrame objects
    
pd.DataFrame(population, columns=['population']) #single dataframe can be constructed from series

data = [{'a': i, 'b': 2 * i} for i in range(3)] #List of dicitonary can be made into a DataFrame, this is a fancy one using list comprehension
pd.DataFrame(data)

pd.DataFrame([{'a':1, 'b':2}, {'b':3, 'c': 4}]) #Nan if some keys/indexes are missing

pd.DataFrame(np.random.rand(3,2),
             columns=['foo','bar'],
             index=['a','b','c']) #Given a two-dimensional array of data, we can create a DataFrame with indexes for columns, rows

A = np.zeros(3, dtype=[('A','i8'), ('B', 'f8')]) #three tuples with two elements, one 8byte integer and 8byte float
pd.DataFrame(A)

# =============================================================================
# Chapter 3: Pandas Index Object
# =============================================================================

ind = pd.Index([2,3,5,7,11]) #Creating an index object
ind
ind[1] #Acts like an array and we can use standard Python indexing notation to get values
ind[::2] #grab every other element from the Array
print(ind.size, ind.shape, ind.ndim, ind.dtype) #Also has many of the same attributes as an array

ind[1] = 0 #They are, however, immutable and cannot be changed

    #Index as ordered set and set theory
    
indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])

indA & indB #Set intersection
indA | indB #Set Union, is it in indA or indB, the complete set of unique indexes
indA ^ indB #symmetric difference

indA.intersection(indB) #Can also be accessed through index methods

    #Data Selection in Series
    
import pandas as pd
data = pd.Series([0.25,0.5,0.75,1.0],
                 index=['a','b','c','d'])
data
data['b']

'a' in data #Checks to see if the sting a is in the index array
data.keys() #like dictionary it provides a list of index in an index object
list(data.items()) #Like dictionary provides the key-value pairs 

data['e'] = 1.25 #Just like assigning to a dictionary by indexing new key with value, works the same way by assigning to new index
data

    #Series as a one-dimensional array: Provides some mechanism for selecting like numpy
    
data['a':'c'] #Slicing by explicit index. Also the final index 'c' is included
data[0:2] #Slicing by implicit integer index, final index is excluded
data[(data > 0.3) & (data < 0.8)] #Masking
data[['a','e']] #Fancy indexing

    #Indexers: loc, iloc, and ix. Because it can be confusing to use implicit and explicit indexing. For Example, data[1:3] do we include the last index???
    
data = pd.Series(['a','b','c'], index=[1,3,5])
data    

data[1] #explicit style
data[1:3] #implicit index when slicing

data.loc[1] 
data.loc[1:3] #This always uses the explicit indexing style

data.iloc[1]
data.iloc[1:3] #iloc allows indexing and slicing that always references the implicit python-style index

    #Data selection in DataFrame: In many ways like a two-dimensional or structured array
    #   also like a dictionary of Series structures sharing same index
    
    #Dataframe like a dictionary
    
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995}) #Key-Value of index and values for area

pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135}) #Key-Value of index and values for population
data = pd.DataFrame({'area':area, 'pop':pop}) #dictionary of series created above with keys representing column names
data

data['area'] #Individual series of the dataframe can be accessed via dictionary styled indexing of column name
data.area #you can also use attribute-style column access

data.pop is data['pop'] #returns false because pop is a method, sometime dot notation for attribute conflicts so be careful!. Avoid this for column assignment

data['density'] = data['pop'] / data['area'] #dictionary-style syntax to modify the object, or add new column

    #Dataframe as two-dimensional array
    
data.values #values are presented as a two-dimensional numpy array
data.T #Transpose to swap rows and columns
data.values[0] #allows us to access a row

data.iloc[:3, :2] #rows 0 to 2 and 0 to 1 to include
data.loc[:'Illinois', :'pop'] #this is explicit so last index value is included in range
data.ix[:3, :'pop'] #ix allows for a hybird of both methods (iloc and loc)

data.loc[data.density > 100, ['pop', 'density']] #Any of the familiar Num-Py style data access patterns can be used. this combines masking and fancy indexing 

data.iloc[0,2] = 90 #Any of these indexing conventions may also be used to set or modify values 
data #first row [0] and 3 column we are changing value to 90

    #Additional indexing conventions
    
data['Florida':'Illinois'] #Slicing rows from Florida to Illinois
data[data.density > 100]  #masking rows which meet this condition
data[1:3] #Second and third element in records

# =============================================================================
# Chapter 4: Operating on Data in Pandas
# =============================================================================

#Ufuncs: Index preservations
#   Because Pandas is designed to work with NumPy and NumPy ufunc will work on Pandas
#   Series and DataFrame objects. Let's start by defining a simple series and DataFrame on
#   which to demonstrate 

import pandas as pd 
import numpy as np

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser

df = pd.DataFrame(rng.randint(0, 10, (3,4)),
                  columns=['A','B','C','D'])
df

np.exp(ser) #If we apple NumpPy ufunc on either of objects, the result will be another Pandas object with indicies preserved 
np.sin(df * np.pi / 4)

    #Ufuncs: Index ALignment
    #   Pandas will align indicies in the process of performing operation
    
area = pd.Series({'Alaska': 133232, 'Texas' : 4423423,
                  'California': 45334}, name = 'area')

population = pd.Series({'California' : 43253253, 'Texas': 43432,
                       'New York': 1432423}, name = 'population')

population / area #any item which one does not have the other in the index it is marked with NaN (Not a number)

A = pd.Series([2,3,4], index = [0,1,2])
B = pd.Series([1,3,5], index = [1,2,3])
A + B

A.add(B, fill_value=0) #If there isn't a matching index on one of the Series, it is given a value of 0 and then the func is performed

A = pd.DataFrame(rng.randint(0, 20, (2,2)),
                 columns = list('AB'))
A

B = pd.DataFrame(rng.randint(0, 10, (3,3)),
                 columns=list('BAC'))
A + B #Index alignment takes place when performing operation on DataFrames instead of Series. Results are also sorted

fill = A.stack().mean() #Compute the mean for the dataframe
A.add(B, fill_value=fill) #fill with the mean of all values in A 

    #Ufuncs: Operating Between DF and Series
    #   Operations between DF and Series is like two-dim and one-dim NumPy array

A = rng.randint(10,size=(3,4))
A

A - A[0] #Subtraction occurs row wise

df = pd.DataFrame(A, columns = list('QRST'))
df - df.iloc[0] 
df.subtract(df['R'], axis=0) #Specify axis if you prefer column wise subtractions

halfrow = df.iloc[0,::2]
df - halfrow #preservation and alignment of columns mens this operation DF will maintain data context, in this case column names

# =============================================================================
# Chapter 5: Handling Missing Data 
    #   General considerations for missing data. Null, NaN or NA
# =============================================================================

import numpy as np
import pandas as pd

vals1 = np.array([1, None, 3, 4])
vals1 #None is python object it and can only be included in arrays with data type 'Object', and overhead is usally high

for dtype in ['object','int']: #Creating a loop to create an array using the different data types
    print("dtype = ", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum() #Object is considerably higher
    
vals1.sum() #Some objects are not supported because of the 'None' object

#NaN: Missing numerical data
#   Not a number uses a special floating-point value recognized by all systems that use standard IEEE floating-point representation

vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype #Notice that the object is floating-point type means that it supports fast operations pushed into compile code

vals2.sum(), vals2.min(), vals2.max() #It infects other objects and will always return NaN as long as one exists in the array
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2) #NP special aggregations which help deal with nan values

    #Nan and None in Pandas
    
pd.Series([1,np.nan,2,None]) #Can handle both interchangably, converting between them where appropriate

x = pd.Series(range(2), dtype=int)
x
x[0] = None #type casts when NA values are present, casts from int to floating-point and changes None to NaN

    #Detecting null values
    
data = pd.Series([1 , np.nan, 'hello', None])
data.isnull() #isnull() method returns a boolean mask

data[data.notnull()] #Using the notnull() method and boolean masks ot remove na records
data.dropna() #The dropna value produces the same result 

    #Dropping NA values
    
df = pd.DataFrame([[1, np.nan, 2],
                  [2, 3, 5],
                  [np.nan, 4, 6]])
df

df.dropna() #Will drop all rows in which any null value is present
df.dropna(1) #Will drop all columns containing a null value

df[3] = np.nan 
df.dropna(axis='columns', how='all') #WIll only drop columns where all rows are NaN
df.dropna(axis='rows', thresh=3) #Rows 0 and 2 are dropped because they contain two NaN values. non NaN for a row must be >= Thresh value

    #Filling NA values
    
data = pd.Series([1, np.nan, 2, None, 2], index = list('abcde'))
data
data.fillna(0) #fills na values with 0
data.fillna(method='ffill') #Will propogate the previous value forward
data.fillna(method='bfill') #Will specify a back-fill to propogate the next values backward

df
df.fillna(method='ffill', axis = 1) #DF is different because we can specify an axis to fill values; however, when using fill if previous value is not available NaN is still returned

# =============================================================================
# Chapter 6: Hierarchical Indexing
#   Multi-level indexing at mutiple levels within a single index
# =============================================================================

import pandas as pd
import numpy as np

    #The bad way: Using tuples as keys
    
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]

populations = [42341232,441432123,
               4324231423,43421324,
               43214212,243214231]
pop = pd.Series(populations, index=index)
pop

pop[('California', 2010):('Texas', 2000)] #Slice series based on multiple index
pop[[i for i in pop.index if i[1] == 2010]] #Selecting all values from 2010 requires mess munging 

    #The better way: Pandas Multiindex
    
index = pd.MultiIndex.from_tuples(index) #Takes each tuple element and splits them into two levels for indexing
pop = pop.reindex(index) #Method to reassign an index
pop #We've now have two level multi-index representaiton

pop[:, 2010] #Result is a single indexed array with just the keys we are interested in 

pop_df = pop.unstack() #convert multi indexed series into a conventionally indexed DataFrame
pop_df.stack() #provides the opposite operation to the one above

pop_df = pd.DataFrame({'total': pop,
                       'under 18': [343242,432432,
                                    432342,432432,
                                    432432,432232]}) #Adding another column, <18 years, into the DF is as simple as adding to df
pop_df

f_u18 = pop_df['under 18'] / pop_df['total']
f_u18.unstack() #ufuncs work with hierarchical indexing, in this case % of pop under 18

    #Methods of MultiIndex Creation 
    #   Most straight forward way is to pass two or more index arrays into contructor
    
df = pd.DataFrame(np.random.rand(4,2),
                  index = [['a','b','a','b'], [1,2,1,2]], #pass list of two or more index arrays 
                  columns = ['data1','data2'])
df

data = {('California', 2000): 33871648,
                    ('California', 2010): 37253956,
                    ('Texas', 2000): 20851820,
                    ('Texas', 2010): 25145561,
                    ('New York', 2000): 18976457,
                    ('New York', 2010): 19378102}
pd.Series(data) #You can also pass a tuple as keys and python will multi index for you

    #Explicit Multiindex constructors
    #   flexibility in how the index is constructed using class method constructors
    
pd.MultiIndex.from_arrays([['a','a','b','b'], [1,2,1,2]])
pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)]) #from an array of tuples

    #MultIndex level names 
    
pop.index.names = ['state','year'] #Useful to pass name attribute of the index aftewards
pop

    #Multiindex for columns
    
index = pd.MultiIndex.from_product([[2013, 2014], [1,2]],
                                   names=['year', 'visit']) #hierarchical indices


columns = pd.MultiIndex.from_product([['Bob','Guido','Sue'],['HR','Temp']],
                                     names = ['subject', 'type']) #hierarchical columns

data = np.round(np.random.randn(4, 6), 1) 
data[:,::2] *=10
data += 37 #Making some random data

health_data = pd.DataFrame(data, index=index, columns=columns) #Creating the DF
health_data

    #Indexing and Slicing MultiIndex: Series
    
pop
pop['California', 2000] #Single Element
pop['California'] #PArtial indexing returns series with the lower-level indices maintained
pop['California':'New York'] #PArtial slicing
pop[:,2000] #We can perform partial indexing on lower levels with empty slice
pop[pop > 2000000] #boolean masks still work as well
pop[['California', 'Texas']] #Selection based on fancy indexing

    #Multiply Indexed Data Frames
    
health_data
health_data['Guido','HR'] #Columns are primary to DF and we can recover Guido's heart rate as such
health_data.iloc[:2, :2] #We can also use explicit, implicit and combination IX
health_data.loc[:, ('Bob','HR')] #Tuple can be used to pass multiple indices

idx = pd.IndexSlice
health_data.loc[idx[:,1], idx[:, 'HR']] #first argument is higher level and second is lower level. In this case for each year take only visit 1 and look at all patients and only heart rate

#Rearranging Muti-indices
#   Many of the multiIndex slicing operations will fail if the index is not sorted

index = pd.MultiIndex.from_product([['a','c','b'],[1,2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data

try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e) #Partial slices and other similar operations require the levels in MultiIndex be sorted (lexographical) order

data = data.sort_index() #Correcy with sort_index method
data
data['a':'b'] #with this partial slicing works as expected

pop
pop.unstack(level=0) #Unstack from highest level into columns for series
pop.unstack(level=1) #Unstack from the lower level into columns for series

pop
pop_flat = pop.reset_index(name='population')
pop_flat #We can turn index labels into columns and give our column a name
pop_flat.set_index(['state','year']) #often we will get data like this and it's best to set a multi index

    # Data Aggregations on Multi-Indices
        # We can pass a level param for hierarchical indexed data

data_mean = health_data.mean(level = 'year')
data_mean #we can get the average for each year using the index

data_mean.mean(axis=1, level='type') #We can get the mean along the levels within the columns 

# =============================================================================
# Chapter 7: Combining Datasets - Concat and Append 
# =============================================================================

import pandas as pd
import numpy as np

def make_df(cols, ind):
    """ Quickly Makes a Data Frame"""
    data = {c:[str(c) + str(i) for i in ind]
    for c in cols}
    return pd.DataFrame(data, ind) #quick function to make a dataframe

make_df('ABC', range(3))

    #Simple Concatentation with pd.concat
    
ser1 = pd.Series(['A','B','C'], index=[1,2,3])
ser2 = pd.Series(['D','E','F'], index=[4,5,6])
pd.concat([ser1, ser2]) 

df1 = make_df('AB', [1,2])
df2 = make_df('AB', [3,4]) 

print(df1); print(df2); print(pd.concat([df1,df2])) #concatentation takes place row wise
print(df1); print(df2); print(pd.concat([df1,df2],axis = 1)) #you can also specify the axis for concatentation

    #Repeating Indexes
    
x = make_df('AB', [0,1])
y = make_df('AB', [2,3]) 

y.index = x.index #make duplicate indices
print(x); print(y); print(pd.concat([x,y]))
    
try:
    pd.concat([x,y], verify_integrity = True)
except ValueError as e:
    print("ValueError:", e) #with verify integrity we get a value error and DF doesn't produce
    
print(x); print(y); print(pd.concat([x, y], ignore_index = True)) #This will reindex the records 
print(x); print(y); print(pd.concat([x, y], keys =['x','y'])) #Another alternative is to use keys options to specify a label and index for data sources (MultiIndex Keys)

    #Concatenation with joins

df5 = make_df('ABC', [1,2])
df6 = make_df('BCD', [3,4])
print(df5); print(df6); print(pd.concat([df5,df6])) #Since D doesn't exist in df5 and A doesn't exist in df6, it returns NaN

print(df5); print(df6); print(pd.concat([df5, df6], join = 'inner')) #Inner only concats fields that appear in both sets
print(df5); print(df6); print(pd.concat([df5, df6], join_axes = [df5.columns])) #Here we explicitly provide a list of fields to concat on, this is based off of df5

print(df1); print(df2); print(df1.append(df2)) #append method does the same thing, but not efficient better to use concat 

#Combining Datasets: Merge and Join (pd.merge)

    #One-to-one joins
    
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'Group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee' : ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)

df3 = pd.merge(df1, df2) #combines both into a single dataframe, index is usually discarded 
df3

    #Many-to-one joins
    
df4 = pd.DataFrame({'Group': ['Accounting', 'Engineering', 'HR'],
                    'Supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4); print(pd.merge(df3, df4)) # Result is one additional column with 'Supervisor' information, where the information is repeated in one or more locations 

    #Many-to-many 
    
df5 = pd.DataFrame({'Group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math','spreadsheets','coding','linux','spreadsheets','organization']})
print(df1); print(df5); print(pd.merge(df1, df5)) # If key in both left and right array contains duplicates, then the result is many-to-many merge 

#Specification of the Merge Key
#   without arguments it looks for one or more matching column names between two inputs
#   When names don't match the are options to handle this

    #The on keyword
    
print(df1); print(df2); print(pd.merge(df1, df2, on='employee')) #This works when both have the same column name

    #Left_on and right_on keywords
    
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3);
print(pd.merge(df1, df3, left_on='employee', right_on='name')) #Joining on a column with different names on each data frame 
pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis=1) #When joining based on a column with diff names in the dataframes, the join column will be repeated and redundant so use drop method

    #Left index and right index keywords
    
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a); print(df2a)
print(pd.merge(df1a, df2a, left_index=True, right_index=True)) #Joining on the index instead of columns

print(df1a); print(df2a); print(df1a.join(df2a)) #using the join method on one of the dataframe makes this task easier

print(df1a); print(df3);
print(pd.merge(df1a, df3, left_index=True, right_on='name')) #You can mix both using index join on one dataframe and a left_on condition on the other data frame

#Specifying Set Arithmetic for Joins
#    When value appears in one column but not the others

    #Inner Join

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish','beans','bread']},
                    columns=['name','food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine','beer']},
                    columns=['name','drink'])

print(df6); print(df7); print(pd.merge(df6, df7)) #Only have a single name in common between dataframes so we perform only a inner join

    #Outer join: return a jin over the union of input columns and fills in all missing values with NAs
    
print(df6); print(df7); print(pd.merge(df6, df7, how='outer'))

    #Left and Right joins: return join over hte left entries and right entries, spectively
    
print(df6); print(df7); print(pd.merge(df6, df7, how='left'))
print(df6); print(df7); print(pd.merge(df6, df7, how='right'))

    #Overlapping Column Names: The suffixes Keyword:  Autmatically suffixes _x and _y when output columns are not unique 
    
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa','Sue'],
                    'rank': [1,2,3,4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa','Sue'],
                    'rank': [3,1,4,2]})
print(df8); print(df9); print(pd.merge(df8, df9, on='name')) #conflicting output names requires the suffix _X and _y automatically
print(df8); print(df9); print(pd.merge(df8, df9, on='name', suffixes=['_L','_R'])) #suffixes argument take list argument for conflicting column names

    #Example: US States Data
    
path = 'PythonDataScienceHandbook-master/notebooks/data/' 
pop = pd.read_csv(path + 'state-population.csv')
areas = pd.read_csv(path + 'state-areas.csv')
abbrevs = pd.read_csv(path + 'state-abbrevs.csv') #Load all required datasets

print(pop.head()); print(areas.head()); print(abbrevs.head())

merged = pd.merge(pop, abbrevs, how = 'outer',
                  left_on='state/region', right_on='abbreviation') #Want to keep all fields in POP df
merged = merged.drop('abbreviation',1) #Drop the column abberviation and keep only one join field
merged.head()

merged.isnull().any() #Looks for rows with nulls to see if there are any mismatches
merged[merged['population'].isnull()].head() #Let's do some masking to figureout which ones these are
merged.loc[merged['state'].isnull(),'state/region'].unique() #issue is that Puerto Rico and Uniter states don't have a state abbreviation key

merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any() #As you can see based on the assignment above, we have fixed the missiong information

final = pd.merge(merged, areas, on='state', how='left') #Left join to include all values from 'merged' dataframe 
final.head() 

final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique() #Clever indexing to filter out which states do not have area calculations

final.dropna(inplace=True)#Removing the NaN values and inplace argurment overwrites the existing object
final.head()

data2010 = final.query('year == 2010 & ages == "total"') #Query tool makes filtering the data easier
data2010.set_index('state', inplace=True) #Reset the index by state
density = data2010['population'] / data2010['area (sq. mi)'] #Calculate density
density.sort_values(ascending=False, inplace=True)
density.head() #Most dense states
density.tail() #least dense states

# =============================================================================
# Chapter 8: Aggregation and Grouping
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape
planets.head()

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser

ser.sum() #Same as one-dimensional array in numpy, aggregates return a single value
ser.mean()

df = pd.DataFrame({'A': rng.rand(5),
              'B': rng.rand(5)})
df
df.mean() #aggregate functions will be calculate for each column automatically
df.mean(axis='columns') #Aggregate within each row instead of column

planets.dropna().describe().T #Describe method computes several common aggregates for each column 

#GroupBy: Split, Apply, Combine
    #   Split: involves breaking up and grouping a DataFrame depending on the value of the specified key
    #   apply: involves computing some function, usually an aggregate transformation or filtering within individual groups
    #   Combine: merges the results of these operations into an output array
    
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data': range(6)}, columns=['key','data'])
df
df.groupby('key') #Split
df.groupby('key').sum() #applies and combines

    #Column Indexing: returns a modified groupby object
    
planets.groupby('method')
planets.groupby('method')['orbital_period'] #Indexing selects a particular series group from DF Group

planets.groupby('method')['orbital_period'].median() #No transformation is done until you call an aggregate on the object 

    #Iteration over groups: GroupBy object supports direct iteration over groups, returning each group as a Series or DataFrame
    
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method,group.shape)) #Method is the name of the grouping and the group is the actual DF object
    
    #Dispatch methods: Any method not explicitly implemented by the GroupBy object will be passed through and called in the indivudal groups
    
planets.groupby('method')['year'].describe()

#Combine: aggregate(), filter(), transform(), and apply() methods before combining the grouped data

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0,10,6)},
                    columns = ['key', 'data1', 'data2'])
df

    #Aggregation: can take a string, a function or a list thereof and compute all aggregates at once
    
df.groupby('key').aggregate(['min', np.median, max]) #Calculating aggregated of 3 functions on the groups
df.groupby('key').aggregate({'data1':'min',
                             'data2':'max'}) #pass a dictionary mapping column names to operations

    #Filtering: drop data based on the group properties
    
def filter_func(x):
    return x['data2'].std() >4 #Creating a filter function

print(df); print(df.groupby('key').std()); 
print(df.groupby('key').filter(filter_func)) #Keep all groups in which the standard deviation is larger than some critical value, like HAVING clause

    #Transformation: return some transformed version of the full data to recombine
    
df.groupby('key').transform(lambda x: x - x.mean()) #Centering the variables

    #apply() method: arbitrary function to the group results. flexible unlike transform which returns the whole set
    
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x #normalizes the first column by the sum of the second

print(df); print(df.groupby('key').apply(norm_by_data2))

    #Split keys: can be any series or list with a length mathing that of a dataframe
    
L = [0,1,0,1,2,0]
print(df); print(df.groupby(L).sum())  #Passing a list, L
print(df); print(df.groupby(df['key']).sum())

df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2); print(df2.groupby(mapping).sum()) #Another method is to provide a dictionary that maps index values to group keys

df2.groupby([str.lower, mapping]).mean() #Preceding key choices can be combines to group on a multi-index

    #Grouping example: can put all together and count discovered planets by method and decade

decade = 10 * (planets['year'] // 10)
decade = decade.astype(str)+'s'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0) #count discovered planet by method and decade

# =============================================================================
# Chapter 9: Pivot Tables
#   Groups entries into a two-dimensional table that provides a multidimensional summarization of the data
#   Split-Apply-Combine happens across not a one-dimensional index but two-dimensional grid
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

titanic.head()

titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack() #Survival by sex and class doing it the hard way
titanic.pivot_table('survived', index='sex', columns='class')

    #Multilevel Pivot Tables: Grouping can be specified with multiple levels
    
age = pd.cut(titanic['age'], [0, 18, 80]) #bin the age using pd.cut 
titanic.pivot_table('survived', ['sex', age], 'class') #Adding in multiple index levels sex and age (from out pd.cut)

fare = pd.qcut(titanic['fare'], 2) #Quantiles Median
titanic.pivot_table('survived', ['sex', age], [fare, 'class']) #Adding in multiple columns as well

titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived': sum, 'fare':'mean'}) #Can pass kwargs into the agg function with key being column and value being agg function

titanic.pivot_table('survived', index='sex', columns='class', margins=True) #We get the overall survival rate for class, gender and class & gender

    #Example: Birthrate Data 
    
births = pd.read_csv(path + 'births.csv')
births.head()

births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum') #Male outnumbers female

%matplotlib inline
import matplotlib.pyplot as plt
sns.set() 
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births per year');

    #Further Data Exploration: Removing outliers via robust sigma-clipping operation
    
quartiles = np.percentile(births['births'],[25,50,75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0]) #.74 comes from interquartile range of gaussian distribution

births = births.query('(births > @mu - 5 * @sig) & births < @mu + 5 * @sig') #Filtering outliers 5 times the IQR from median 

births['day'] = births['day'].astype(int) #Change day to int because it was string cause of null values

births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format = '%Y%m%d') #Create a datetime index from year month and day
births['dayofweek'] = births.index.dayofweek

import matplotlib.pyplot as plt
import matplotlib as mpl

births.pivot_table('births', index='dayofweek',
                   columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'])
plt.ylabel('mean births by day')


