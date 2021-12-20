import pandas as pd
import numpy as np
from rtree import index
import csv
from consts import sizes

def generate_tree(size):
    path = "./index/rtree_"+str(size)
    df = pd.read_csv("./csv/preprocessed.csv")
    df = df.head(size)
    columns = [str(i) for i in range(1, df.shape[1])]
    x = df.loc[:, columns]
    y = df.loc[:, ["dir"]]
    prop = index.Property()
    prop.dimension = x.shape[1]
    prop.buffering_capacity = 5
    prop.dat_extension = "data"
    prop.idx_extension = "index"
    
    idx = index.Index(path, properties = prop)
    
    for i in range(size):
        temp = x.iloc[i].values
        point = tuple(np.concatenate([temp, temp]))
        image_dir = y.iloc[i].values[0]
        idx.insert(i, point, obj = image_dir)
        
def generate_trees():
    for size in sizes:
        generate_tree(size)
    print("terminado")
    
    
generate_trees()
