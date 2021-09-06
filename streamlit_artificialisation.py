import streamlit as st
import numpy as np
import pandas as pd
import os

from keras_unet.utils import plot_patches

from PIL import Image
image = Image.open('austin4.jpg')
st.image(image, width=500)


st.title('''Artificialisation des sols en France''')
st.write('''*Comment évaluer l'artificialisation des sols?*''')
st.header('''**Définition d'artificialisation**''')
st.write('''"Ce phénomène consiste à transformer un sol naturel, agricole ou forestier, par des opérations d’aménagement pouvant entraîner une imperméabilisation partielle ou totale, afin de les affecter notamment à des fonctions urbaines ou de transport (habitat, activités, commerces, infrastructures, équipements publics…). "
[Ministère de la transition écologique](https://www.ecologie.gouv.fr/artificialisation-des-sols) ''')

## User input
file_bytes = st.file_uploader("Upload a tile", type=("png", "jpg")) 

if file_bytes == None:
    st.warning('No file selected. Please select a file.')
else:
    image2 = Image.open(file_bytes)
    ### Converting image to numpy array
    data = np.asarray(image2)

    ### Getting smaller batches
    
    st.write("x_crops shape: ", str(data.shape))         
    plot_patches(
        img_arr=data_crops, # required - array of cropped out images
        org_img_size=(1000, 1000), # required - original size of the image
        stride=100) # use only if stride is different from patch size

    

## Try model

from keras_unet.models import satellite_unet

model = satellite_unet(input_shape=(256, 256, 3))

# Opens a image and get in it in the correct data form
im = Image.open("austin4.jpg")
im = im.resize((256, 256))
img_list=[]
img_list.append(np.array(im))
img_np = np.asarray(img_list)
x = np.asarray(img_np, dtype=np.float32)/256

# Download the model
model_filename = 'artificialisation_model_25082021.h5'

#Use the model
model.load_weights(model_filename)
y_pred = model.predict(x)

#Print the result
from matplotlib import pyplot as plt
plt.imshow(y_pred[0, :, :, :], cmap='gray')
plt.savefig('saved_figure.png')
image = Image.open('saved_figure.png')
st.image(image, width=500, caption="Raster obtenu avec le modèle Keras Unet pour la photo d'Austin")
