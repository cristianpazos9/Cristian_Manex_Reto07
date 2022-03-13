#!/usr/bin/env python
# coding: utf-8

# In[1]:


import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
from PIL import Image
from io import BytesIO


# In[2]:


es = Elasticsearch([{'host': '192.168.56.105', 'port': 9200, 'scheme' : 'http'}])


# In[3]:


df_total = pd.read_excel('Excel_Fotos.xlsx',sheet_name='total')
df_total


# # INDICE1

# In[4]:


idd = 1
for m,n,i in zip(df_total.Name_Image,df_total.Num,df_total.index):
    with open('fotos/'+ str(m) + str(n) +'.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        encoded_string=encoded_string.decode("utf-8")

    doc = {
        'Name_Image': df_total.loc[df_total['Name_Image']==i,'Name_Image'],
        'Nombre': df_total.loc[df_total['Nombre']==i,'Nombre'],
        'Mascarilla': df_total.loc[df_total['Mascarilla']==i,'Mascarilla'],
        'image' : encoded_string
        }

    
    res = es.index(index = 'index_fotos', id = idd, document = doc)
    idd += 1


# In[5]:


res = es.get(index="index_fotos", id=36)


# In[6]:


img=res['_source']['image'].encode()
im = Image.open(BytesIO(base64.b64decode(img)))


# In[7]:


imgplot = plt.imshow(im)
plt.show()


# In[8]:


print(es.cat.count(index = 'index_fotos',params={'format':'jpg'}))


# In[9]:


#es.indices.delete(index='index_fotos',ignore=[400,404])


# # INDICE 2

# In[10]:


df_acc = pd.read_excel('Excel_Accuracy.xlsx')
df_acc


# In[25]:


res = es.search(index = 'index_versiones_del_modelo', query = {'match':{'Version_del_modelo':1}})
for hit in res['hits']['hits']:
    print(hit['_source'])
    print('/n')


# In[23]:


idd = 1

for i,a,p,r,t,s,c,y,n,g in zip(df_acc.Version,df_acc.model_accuracy,df_acc.predict_accuracy,df_acc.random_state_model,df_acc.tol,df_acc.n_splits,df_acc.random_state_cv,df_acc.scoring,df_acc.number_img_test,df_acc.number_noisy_img_tets):
    doc = {
        'Version_del_modelo': df_acc.loc[df_acc['Version']==i,'Version'],
        'Accuracy_del_modelo': df_acc.loc[df_acc['model_accuracy']==a,'model_accuracy'],
        'Accuracy_de_las_predicciones': df_acc.loc[df_acc['predict_accuracy']==p,'predict_accuracy'],
        'Random_state_del_modelo': df_acc.loc[df_acc['random_state_model']==r,'random_state_model'],
        'tol': df_acc.loc[df_acc['tol']==t,'tol'],
        'number_splits': df_acc.loc[df_acc['n_splits']==s,'n_splits'],
        'Random_state_del_cross_validation': df_acc.loc[df_acc['random_state_cv']==c,'random_state_cv'],
        'Scoring': df_acc.loc[df_acc['scoring']==y,'scoring'],
        'Number_of_images_test': df_acc.loc[df_acc['number_img_test']==n,'number_img_test'],
        'Number_of_noisy_images_test': df_acc.loc[df_acc['number_noisy_img_tets']==g,'number_noisy_img_tets'],
        }

    
    res = es.index(index = 'index_versiones_del_modelo', id = idd, document = doc)
    idd += 1


# In[22]:


#es.indices.delete(index='index_versiones_del_modelo',ignore=[400,404])


# # INDICE 3

# In[13]:


df_predicciones = pd.read_excel('Predicciones.xlsx')
df_predicciones


# In[15]:


idd = 1
for m,n,a,i,t,p in zip(df_predicciones.Name_Image,df_predicciones.Num,df_predicciones.Nombre,df_predicciones.predict_version1,df_predicciones.predict_version2,df_predicciones.predict_version3):
    with open('fotos_predicion/'+ str(m) + str(n) +'.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        encoded_string=encoded_string.decode("utf-8")

    doc = {
        'Name_Image': df_predicciones.loc[df_predicciones['Name_Image']==m,'Name_Image'],
        'Num': df_predicciones.loc[df_predicciones['Num']==n,'Num'],
        'Nombre': df_predicciones.loc[df_predicciones['Nombre']==a,'Nombre'],
        'predict_version1': df_predicciones.loc[df_predicciones.predict_version1==i,'predict_version1'],
        'predict_version2': df_predicciones.loc[df_predicciones.predict_version2==t,'predict_version2'],
        'predict_version3': df_predicciones.loc[df_predicciones.predict_version3==p,'predict_version3'],
        'image' : encoded_string
        }

    
    res = es.index(index = 'index_predicciones', id = idd, document = doc)
    idd += 1


# In[ ]:


#es.indices.delete(index='index_predicciones',ignore=[400,404])

