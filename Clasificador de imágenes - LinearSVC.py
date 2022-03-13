#!/usr/bin/env python
# coding: utf-8

# # Detección de objetos con SVM mediante las características HOG

# Un Histograma de Gradientes Orientados (HOG - Histogram Oriented Gradients) es un descriptor de características utilizado en una variedad de aplicaciones de procesamiento de imágenes y visión por computadora con el fin de detectar objetos.  
#   
# En esta sección, demostraremos cómo se pueden usar las funciones de la biblioteca `python-opencv` para detectar objetos en una imagen usando **HOG-SVM**. El siguiente código muestra cómo calcular los descriptores HOG a partir de una imagen y utilizar los descriptores para alimentar un clasificador SVM previamente entrenado.  
#    
# Para el ejemplo, haremos uso de un dataset que contiene los siguientes 4 escenarios:  
#   
# - **cristian**
# - **manex**
# - **alvaro**
# - **iker**

# **1. Importamos las librerías necesarias y las imágenes a usar:**

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2

from tqdm import tqdm
import random as rn
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

from skimage import feature, color, data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Importamos las imagenes de entrenamiento y de testeo:

# In[3]:


# Especificamos la ruta de train
trn_img_path = "seg_train/"

# Especificamos la ruta de test.
tst_img_path = "seg_test/escenario1/"

# Vamos a crear 2 conjuntos de arrays para los datos de entrenamiento y de prueba. 
# Uno para almacenar los datos de la imagen y otro para los detalles de la etiqueta

X_train =[] # Stores the training HOG: Histogram of Oriented Gradients 
label_train = [] # Stores the training image label

X_test = [] # Stores the testing image hog data
label_test = [] # Stores the testing image label

scene_label=['alvaro','cristian','iker','manex']


# La función **"hog_data_extractor"** obtendrá las características de HOG del archivo jpeg que se le pasa:

# In[4]:


def hog_data_extractor(jpeg_path):
    jpeg_data = cv2.imread(jpeg_path)
    jpeg_data=cv2.resize(jpeg_data,(150,150)) 
    hog_data = feature.hog(jpeg_data)/255.0
    return hog_data


# La función **"jpeg_to_array"** carga las imágenes de la ruta dada y almacena las características de HOG en *X_train* y *X_test* respectivamente:

# In[5]:


def jpeg_to_array (scene_type, img_root_path,data_type):
    scene_path = os.path.join(img_root_path,scene_type.lower())
    print('Loading ' + data_type +' images for scene type '+scene_type)

    
    for img in os.listdir(scene_path):
        img_path = os.path.join(scene_path,img)
        if img_path.endswith('.jpg'):
            if(data_type == 'Training'):
                X_train.append(hog_data_extractor(img_path))
                label_train.append(str(scene_type))
            if(data_type =='Testing'):
                X_test.append(hog_data_extractor(img_path))
                label_test.append(np.array(str(scene_type)))


# A continuación, llamamos a la función **jpeg_to_array** para cargar las imáegenes tanto de entrenamiento como de validación:

# In[6]:


[jpeg_to_array(scene,trn_img_path,'Training')for scene in scene_label]
print("Tamaño de X_train: ", len(X_train))


# In[7]:


[jpeg_to_array(scene,tst_img_path,'Testing')for scene in scene_label]
print("Tamaño de X_test: ", len(X_test))


# Hacemos uso de `LabelEncoder()` para codificar las etiquetas de los escenarios:

# In[8]:


label_train


# In[9]:


label_test


# In[10]:


le = LabelEncoder()
y_train = le.fit_transform(label_train)
y_test = le.fit_transform(label_test)


# In[11]:


y_train


# In[12]:


y_test


# **2. Creamos un modelo lineal SVM y lo entrenamos:**

# In[13]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import LinearSVC


# In[14]:


lsvc = LinearSVC(random_state=0,tol=1e-5)
lsvc.fit(X_train,y_train)


# Aplicamos una validación cruzada:

# In[15]:


import warnings # filter all the warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
lsvc_score = lsvc.score(X_test,y_test)
print('Score', lsvc_score)
kfold = KFold(n_splits=10, random_state=9, shuffle = True)
cv_results = cross_val_score(lsvc , X_train, y_train, cv=kfold, scoring="accuracy")
print(cv_results)


# **3. Finalmente, predecimos la clasificación de escenarios de algunas imágenes aleatorias:**

# In[16]:


def scene_predict(img_path):
    image = cv2.imread(img_path)
    ip_image = Image.open(img_path)
    image = cv2.resize(image,(150,150))
    prd_image_data = hog_data_extractor(img_path)
    scene_predicted = lsvc.predict(prd_image_data.reshape(1, -1))[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(ip_image)
    ax[0].set_title('input image')

    ax[1].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    ax[1].set_title('Scene predicted :'+ scene_label[scene_predicted]);


# In[17]:


#Posar antes de ejecutar
cap = cv2.VideoCapture(0)

leido, frame = cap.read()

if leido == True:
	cv2.imwrite("seg_pred/fotoinstantanea.jpg", frame)
	print("Foto tomada correctamente")
else:
	print("Error al acceder a la cámara")

"""
	Finalmente liberamos o soltamos la cámara
"""
cap.release()


# In[18]:


ip_img_folder = 'seg_pred/'
ip_img_files = ['manex10.jpg','manex11.jpg','manex12.jpg','cristian10.jpg','cristian11.jpg','cristian12.jpg','alvaro10.jpg','alvaro11.jpg','alvaro12.jpg','iker10.jpg','iker11.jpg','iker12.jpg','fotoinstantanea.jpg']
scene_predicted = [scene_predict(os.path.join(ip_img_folder,img_file))for img_file in ip_img_files]


# In[19]:


# Predicciones test
# ==============================================================================
predicciones = lsvc.predict(X_test)
predicciones


# In[20]:


# Accuracy de test del modelo 
# ==============================================================================
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")

# %%
