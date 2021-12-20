# BDII_P3
## Integrantes
Nombre y Apellido | Codigo | 
--- | --- | 
Mauricio Rodriguez |  201810642 | 
Renzo Tenazoa | 201810251|

## Introduccion y Objetivos del proyecto
El proyecto en cuestión busca devolver al usuario los rostros más similares al rostro que este cargue en la web.
Para ello, se busca utilizar un estructura multidimensional que indexe los vectores característicos de miles de rostros a fin de realizar una búsqueda eficiente.
La estructura multidimensional a la que nos referimos es un rtree, y los rostros se obtienen de la base de datos [lfw](http://vis-www.cs.umass.edu/lfw/). Asimismo, los vectores característicos que se insertan en el rtree nacen de la librería [face recognition](https://github.com/ageitgey/face_recognition) Cabe destacar que este vector lo consiguen mediante el uso de una red neuronal entrenada para identificar las caracteristicas principales de un rostro.

## Obteniendo la data
Lo primero que realizamos en este proyecto fue obtener todos los vectores caracteristicos de las más de 13000 fotos de la base de datos y almacenarlos en un excel que nos permita reutilizar estos vectores. Aparte de guardar los valores, guardamos también la dirección de la ruta a la foto que corresponde al vector. Esto es debido a que por cada rostro hay una carpeta y en cada carpeta pueden haber varias tomas del mismo rostro por lo que es útil guardar este dato también.

```python
def images_to_csv():
    with open('csv/data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for folder in os.listdir('./lfw/'): 
            person_dir = pjoin('./lfw/', folder)
            for i in os.listdir(person_dir):  
                image_dir = pjoin(person_dir, i)
                picture = face_recognition.load_image_file(image_dir)
                known_encoding = face_recognition.face_encodings(picture)
                if (bool(known_encoding)):
                    row = list(known_encoding[0])
                    row.append(image_dir)
                    writer.writerow(row)
```

## Procesando la data

Luego de haber obtenido los vectores característicos de todos los rostros en el dataset, preprocesamos toda la data.
Para ello, primero estandarizamos los datos, de forma que si hay valores muy distintos entre ellos se normalicen y no afecten al resultado. Esto lo hacemos de la forma: 

```python
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
```
Donde x son todas las columnas del csv excepto la última, ya que el dato en la última columna es la dirección de la imagen.

Luego de ello también utilizamos una reducción del tamaño del vector característico mediante analisis de componentes principales (PCA), la cual es una técnica que 
detecta los valores más importantes de un vector y los agrupa de forma que ocupen menos dimensiones y pierdan la menor cantidad de información al hacer esto. El PCA se realiza para evitar la **maldición de la dimensionalidad**. Los vectores característicos de parte de la librería face_recognition tienen 128 elementos, es decir es un vector de 128 dimensiones. El PCA lo realizamos de la siguiente forma:
```python
    pca = PCA(.95)
    pca.fit(x)
    x = pca.transform(x)
```
El parámetro .95 declara que queremos mantener el 95% de información al comprimir la data. Con ello, nos quedamos con un total de 58 columnas, es decir, la cantidad de dimensiones del vector se redujo a menos de la mitad.
Estos datos los guardamos, al igual que en el caso anterior, en un csv para uso posterior.

## Generando los árboles
Luego de preprocesar la data pasamos a ingresar todos esos datos en estructuras que puedan indexarlos, en este caso en los rtree. 
Utilizamos distintos tamaños para realizar pruebas posteriores, pero principalmente convertimos esos datos en un "punto" legible para el rtree y lo insertamos. Tanto el índice como los datos se guardan en una carpeta.

```python
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

      idx = index.Index(path, properties = prop)

      for i in range(size):
          temp = x.iloc[i].values
          point = tuple(np.concatenate([temp, temp]))
          image_dir = y.iloc[i].values[0]
          idx.insert(i, point, obj = image_dir)
```
## Busqueda knn_sequential
  Para esta busqueda no se utiliza el rtree, es una busqueda secuencial que utiliza una priority queue. La priority queue fue obtenida de una libreria de python 
  llamada **heapq**. La búsqueda devuelve los k elementos más cercanos pero utiliza un medio iterativo por lo que no es muy eficiente. 
  
  ```python
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
    print("sequential: ",result)
    return result
    ```
    ## Busqueda knn_rtree
      Esta búsqueda utiliza los datos que se han indexado, por lo que es más eficiente. Al igual que en el caso anterior, devuelve los k valores más cercanos al  input.
```python
  def knn_rtree(idx, size, input, k = 8):
      image = preprocess_input(input)
      point = tuple(np.concatenate([image, image]))
      pre_result = list(idx.nearest(coordinates=point, num_results=k*2, objects = "raw"))
      result = []
      for i in range(k):
          result.append(pre_result[i*2])
      print("rtree: ",result)
      return result
```

## Tests
Realizamos tests para ambos tipos de busqueda con distintos tipos de tamaños, por tamaño nos referimos a cantidad de fotos sobre las que harán la búsqueda los algoritmos.

KNN_sequential | KNN_rtree | Tamaño |
--- | --- | --- |
0.35722804069519043|  0.21970176696777344 | 100
0.3506777286529541| 0.21974420547485352| 200
0.3776669502258301 | 0.21924376487731934| 400
0.3974766731262207 | 0.23142075538635254| 800
0.44744157791137695| 0.2402486801147461| 1600
0.5368270874023438| 0.27453184127807617| 3200
0.7456254959106445 | 0.3316769599914551| 6400
1.1445732116699219 | 0.4600968360900879| 12800

Dados los test, obtuvimos el siguiente grafico comparativo del tiempo de ejecucion de cada búsqueda

![alt text](https://github.com/mauricio-rodriguez/BDII_P3/blob/main/test.png)
