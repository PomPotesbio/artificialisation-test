import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import os

from PIL import Image

from keras_unet.utils import get_patches
from keras_unet.utils import plot_patches

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
    im = Image.open(file_bytes)
    im_size=im.size
    st.image(im, width=500)
    
    ### Getting smaller batches - type(data) and type(data_crops) are numpy.ndarray
    data = np.array(im)
    data_crops = get_patches(img_arr=data)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_patches(img_arr=data_crops, org_img_size=im_size))
    
    from keras_unet.models import satellite_unet
    model = satellite_unet(input_shape=(256, 256, 3))

    # Assuming data is in data_crops as a numpy array of (x, 256, 256, 3) shape.
    x = np.asarray(data_crops, dtype=np.float32)/256

    # Download the model
    model_filename = 'artificialisation_model_06092021.h5'
    #Use the model
    model.load_weights(model_filename)
    y_pred = model.predict(x)

    # Plot the small patches
    from keras_unet.utils import plot_patches
    plot_patches(
        img_arr=y_pred, # required - array of cropped out images
        org_img_size=image_size) # required - original size of the image
    
    # Reconstruct the picture
    import matplotlib.pyplot as plt
    from keras_unet.utils import reconstruct_from_patches

    im_reconstructed = reconstruct_from_patches(
        img_arr=y_pred, # required - array of cropped out images
        org_img_size=im_size) # required - original size of the image

    plt.figure(figsize=(10,10))
    plt.imshow(im_reconstructed[0])
    plt.show()
