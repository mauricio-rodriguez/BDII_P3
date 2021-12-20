import pandas as pd
import face_recognition
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from rtree import index
import pickle
from heapq import *
from generate_tree import global_properties

def preprocess_input(input):
    picture = face_recognition.load_image_file(input)
    known_encoding = face_recognition.face_encodings(picture)

    input = pd.DataFrame(data=known_encoding, columns = [str(i) for i in range(1, 129)])

    scaler = pickle.load(open("./helpers/scaler.dat", "rb"))
    pca = pickle.load(open("helpers/pca.dat", "rb"))
    
    #scaler
    data = scaler.transform(input)  
    #PCA
    data = pca.transform(data)
    
    return data[0]


def knn_rtree(size, input, k = 8):
    path = "./index/rtree_"+str(size)
    prop = index.Property()
    prop.dimension = 58
    prop.buffering_capacity = 5
    
    idx = index.Index(path, properties = prop)
    image = preprocess_input(input)
    point = tuple(np.concatenate([image, image]))
    result = list(idx.nearest(coordinates=point, num_results=k))
    print(result)
    return result
    
            

def knn_sequential(size,input, k = 8):
    h = []
    df = pd.read_csv("./csv/preprocessed.csv")
    df = df.head(size)
    columns = [str(i) for i in range(1, df.shape[1])]
    x = df.loc[:, columns]
    y = df.loc[:, ["dir"]]
    
    image = preprocess_input(input)

    for i in range(size):
        dist = np.linalg.norm(x.iloc[i].values - image) * -1
        
        image_dir = y.iloc[i].values[0]
        if i < k:
            heappush(h, (float(dist), image_dir))
        else:
            heappushpop(h, (float(dist), image_dir))
                    
    result = [0] * k
    for i in range(k):
        result[i] = heappop(h)[1]
    result = list(reversed(result))
    
    return result