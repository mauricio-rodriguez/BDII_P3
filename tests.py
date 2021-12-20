from search import knn_rtree,knn_sequential
from time import time
from consts import sizes
import matplotlib.pyplot as plt

def test():
    path = "fotos_bd/vizcarra.png"
    
    times_rtree = []
    times_sequential = []
    for size in sizes:
        start = time()
        knn_rtree(size, path)
        finish = time()
        times_rtree.append(finish-start)

        start = time()
        knn_sequential(size, path)
        finish = time()
        times_sequential.append(finish-start)
    
    print("Rtree: ",times_rtree,"Sequential: ", times_sequential)

    plt.plot(sizes, times_rtree, label='KNN RTree')
    plt.plot(sizes, times_sequential, label='KNN Secuencial')
    plt.xlabel("Number of images")
    plt.ylabel("Seconds")
    plt.legend()
    plt.savefig('img')

test()