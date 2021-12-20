import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from rtree import index
import csv
from consts import sizes
import pickle


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
    #pca = PCA(.95)
    #  sabemos que .95 es 58 columnas
    pca = PCA(n_components=58)
    pca.fit(x)
    x = pca.transform(x)
    with open('csv/preprocessed.csv', 'w', newline='') as file:
        header =  [str(i) for i in range(1, 59)]
        header.append("dir")
        writer = csv.writer(file)
        writer.writerow(header)
        for index, line in enumerate(x):
            row = list(line)
            row.append(y["dir"][index])
            writer.writerow(row)
    file.close()
    pickle.dump(scaler, open("./helpers/scaler.dat", "wb"))
    pickle.dump(pca, open("./helpers/pca.dat", "wb"))

preprocess_data()