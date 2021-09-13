import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import rasterio as rio
import rasterio.plot
import geopandas as gpd
import fiona
import os

from PIL import Image

from keras_unet.utils import get_patches
from keras_unet.utils import plot_patches

st.title('''Artificialisation des sols en France''')
st.write('''*Comment évaluer l'artificialisation des sols?*''')
st.header('''**Définition d'artificialisation**''')
st.write('''"Ce phénomène consiste à transformer un sol naturel, agricole ou forestier, par des opérations d’aménagement pouvant entraîner une imperméabilisation partielle ou totale, afin de les affecter notamment à des fonctions urbaines ou de transport (habitat, activités, commerces, infrastructures, équipements publics…). "
[Ministère de la transition éque](https://www.ecologie.gouv.fr/artificialisation-des-sols) ''')

## User input
user_input = st.text_input("Choisir sa ville")
if user_input == None:
    st.warning('No file selected. Please select a file.')
else:
    if user_input == "Paris" or user_input=="paris":
        im=Image.open("DémoImages/Paris.jpg")
        im_size=im.size 
        st.image(im)
    elif user_input == "Strasbourg" or user_input=="strasbourg":
        im=Image.open("/DémoImages/Strasbourg.jpg")
        im_size=im.size
    elif user_input == "Lectoure" or user_input=="lectoure":
        im=Image.open("/DémoImages/Lectoure.jpg")
        im_size=im.size
    elif user_input == "Montargis" or user_input=="montargis":
        im=Image.open("/DémoImages/Montargis.jpg")
        im_size=im.size
 
    ### Getting smaller batches - type(data) and type(data_crops) are numpy.ndarray
        data = np.array(im)
        data_crops = get_patches(img_arr=data)
    
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plot_patches(img_arr=data_crops, org_img_size=im_size))
    
   
