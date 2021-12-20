from search import knn_rtree,knn_sequential
from time import time
from consts import sizes
import matplotlib.pyplot as plt
from rtree import index
# from generate_tree import indices, generate_trees
def test():
    #generate_trees()
    path = "fotos_bd/vizcarra.png"
    times_rtree = []
    times_sequential = []
    iterador = 0
    for size in sizes:
        prop = index.Property()
        prop.dimension = 58
        prop.buffering_capacity = 5
        tree_dir = "./index/rtree_"+str(size)
        
        idx = index.Index(tree_dir, properties = prop)
        start = time()
        knn_rtree(idx,size, path)
        iterador += 1
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