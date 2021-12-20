from flask import Flask,render_template, request, Response, redirect,url_for
import json

from flask.helpers import send_from_directory
from search import knn_rtree, knn_sequential
# from generate_tree import generate_tree
import os
from rtree import index

app = Flask(__name__)

allRows = 13170


@app.route("/search", methods = ['GET','POST'])
def search():
    if 'file' not in request.files:
        print("no file part")
    else: 
        file = request.files['file']
        if file.filename == '':
            return "No hay archivo"
        if file:
            filenames = []
            file_dir = os.path.join("input_images/", file.filename)
            file.save(file_dir)
            k = request.form.get("kvalue")
            tipo = request.form.get("tipo")
            if tipo.lower() == "seq":
                closest_neighbors = knn_sequential(13170, file_dir)
            else:
                prop = index.Property()
                prop.dimension = 58
                prop.buffering_capacity = 5
                tree_dir = "./index/rtree_"+str(13170)
                idx = index.Index(tree_dir, properties = prop)
                closest_neighbors = knn_rtree(idx, 13170, file_dir, int(k))
                
            for i in closest_neighbors:
                dirname = os.path.abspath(i)
                #print(dirname)
                filenames.append(dirname)

        #print(filenames)
        return render_template('results.html', results = filenames)

        

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/image/<directory>/<filename>")
def sendimage(filename, directory):
    print(directory)
    d = 'lfw/' + directory
    path = os.path.abspath(d)
    print(path)
    return send_from_directory(path, filename)

if __name__ == '__main__':
    app.secret_key = ".."
    app.run(port=5000, threaded=True, host=('127.0.0.1'))

