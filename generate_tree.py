import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from rtree import index
import csv
from consts import sizes
def preprocess_data():
    df = pd.read_csv("./csv/data.csv")
    columns = [str(i) for i in range(1, 129)]
    x = df.loc[:, columns]
    y = df.loc[:, ["dir"]]
    #Se estandarizan los datos
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    #PCA
    pca = PCA(.95)
    pca.fit(x)
    x = pca.transform(x)
    with open('csv/preprocessed.csv', 'w', newline='') as file:
        header =  [str(i) for i in range(1, len(x[0]))]
        header.append("dir")
        writer = csv.writer(file)
        writer.writerow(header)
        for index, line in enumerate(x):
            row = list(line)
            row.append(y["dir"][index])
            writer.writerow(row)

def generate_tree(size):
    path = "./index/rtree_"+str(size)
    df = pd.read_csv("./csv/preprocessed.csv")
    df = df.head(size)
    columns = [str(i) for i in range(1, df.shape[1])]
    x = df.loc[:, columns]
    y = df.loc[:, ["dir"]]
    prop = index.Property()
    prop.dimension = df.shape[1] - 1
    prop.buffering_capacity = 100
    prop.dat_extension = "data"
    prop.idx_extension = "index"
    
    idx = index.Index(path, properties = prop)
    
    for i in range(size):
        temp = x.iloc[i].values
        point = tuple(np.concatenate([temp, temp]))
        idx.insert(i, point, obj = y.iloc[i].values[0])
        
def generate_trees():
    for size in sizes:
        generate_tree(size)
    print("terminado")
    
    
generate_trees()