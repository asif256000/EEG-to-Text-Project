import pickle
import numpy as np
import os


loc = './dataset/processed/Task3_TSR_processed.pickle'

with open(loc, 'rb') as f:
    data = pickle.load(f)

subjects = list(data.keys())
print(len(data[subjects[0]][0]['word']))