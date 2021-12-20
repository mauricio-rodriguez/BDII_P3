from search import knn_rtree,knn_sequential
from consts import sizes

def test():
    path = "./fotos_test"
    times_rtree = []
    times_sequential = []
    for size in sizes:
        knn_rtree(size, input)
        knn_sequential(size, input)