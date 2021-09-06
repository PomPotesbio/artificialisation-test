import streamlit as st
import numpy as np
import pandas as pd
import os

from keras_unet.utils import get_patches
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
    ### Getting smaller batches
    data = np.array(image2)
    
    data_crops = get_patches(
    img_arr=data, # required - array of images to be cropped
    size=500, # default is 256
    stride=500) # default is 256

    st.write("x_crops shape: ", str(data_crops.shape))
    
     
    plot_patches(
        img_arr=data_crops, # required - array of cropped out images
        org_img_size=(5000, 5000), # required - original size of the image
        stride=100) # use only if stride is different from patch size
