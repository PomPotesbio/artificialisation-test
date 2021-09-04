import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
image = Image.open('austin4.jpg')
st.image(image, width=500)


st.title('''Artificialisation des sols en France''')
st.write('''*Comment évaluer l'artificialisation des sols?*''')
st.header('''**Définition d'artificialisation**''')
st.write('''"Ce phénomène consiste à transformer un sol naturel, agricole ou forestier, par des opérations d’aménagement pouvant entraîner une imperméabilisation partielle ou totale, afin de les affecter notamment à des fonctions urbaines ou de transport (habitat, activités, commerces, infrastructures, équipements publics…). "
[Ministère de la transition écologique](https://www.ecologie.gouv.fr/artificialisation-des-sols) ''')

## User input
import os

filename = st.text_input('Enter a file path:')
try:
    with open(filename) as input:
        st.text(input.read())
except FileNotFoundError:
    st.error('File not found.')



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
