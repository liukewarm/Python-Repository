"""
SCRIPT: JSON to CSV
CREATED: 07-23-2017
VERSION: 1.0
COMMENTS: This script should dynamically flatten highly-nested JSON data or files into CSV formats. Should
    only require the input JSON file path and the output path for the CSV file.
"""
import argparse
import collections
import csv
import pandas.io.json as json_normalize
import simplejson as json
import pandas as pd

with open("stocks.json") as json_data:
    d = json.load(json_data)

sample_object = {'Name':'John', 'Location':{'City':'Los Angeles','State':'CA'}, 'hobbies':['Music', 'Running']}

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out



def write_to_csv(path,dict_object):
    with open(path, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_object.items():
            writer.writerow([key, value])

def append_to_dataframe(json_file): #This iterates through each list item first before flattening the JSON object
    flat_json = []
    for item in json_file:
        flat = flatten_json(item)
        flat_json.append(flat)
    return flat_json

flat_json = append_to_dataframe(d)
df_flat = json_normalize.json_normalize(flat_json) #this turns the dictionary into a data frame

#flat = flatten_json(d)
#df_flat = json_normalize.json_normalize(flat)
df_flat.to_csv('test.csv',sep=',',header=True) #this saves the data frame into a csv
