# from sklearn.preprocessing import StandardScaler
# import face_recognition
# from sklearn.decomposition import PCA

import os
from os.path import join as pjoin
import csv
import face_recognition
import pandas as pd
# from consts import raw_dir
header =  [str(i) for i in range(1, 129)]
header.append("dir")

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
                    
images_to_csv()