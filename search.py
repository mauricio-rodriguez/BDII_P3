import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from rtree import index

def preprocess_input(input):
    picture = face_recognition.load_image_file(input)
    known_encoding = face_recognition.face_encodings(input)
    
    scaler = StandardScaler()
    scaler.fit(known_encoding[0])
    data = scaler.transform(known_encoding[0])
    
    #PCA
    pca = PCA(n_components = 58)
    pca.fit(data)
    data = pca.transform(data)
    
    return data


def knn_rtree(size, input, k = 8):
    path = "./index/rtree_"+str(size)
    prop = index.Property()
    prop.dimension = 58
    prop.buffering_capacity = 5
    
    idx = index.Index(path, properties = prop)
    image = preprocess_input(input)
    point = tuple(np.concatenate([image, image], axis = None))
    result = list(idx.nearest(coordinates=point, num_results=k, objects = str(size)))
    return result
    
            

def knn_sequential(size,input, k = 8):
    h = []
    df = pd.read_csv("./csv/preprocessed.csv")
    df = df.head(size)
    x = df.loc[:, columns]
    y = df.loc[:, ["dir"]]
    
    image = preprocess_input(input)
    
    for i in range(size):
        dist = np.linalg.norm(x.iloc[i].values - input) * -1
        
        image_dir = y.iloc[i].values[0]
        if i < k:
            heappush(h, float(dist), image_dir)
        else:
            heappushpop(h, float(dist), image_dir)
                    
    result = [0] * k
    for i in range(k):
        result[i] = heappop(h)[1]
        
    return result.reverse()