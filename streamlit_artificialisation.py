import streamlit as st
import numpy as np
import cv2
import matplotlib as plt

from PIL import Image

from keras_unet.utils import get_patches
from keras_unet.utils import plot_patches

st.title('''Artificialisation des sols en France''')
st.write('''*Comment évaluer l'artificialisation des sols?*''')
st.header('''**Définition d'artificialisation**''')
st.write('''"Ce phénomène consiste à transformer un sol naturel, agricole ou forestier, par des opérations d’aménagement pouvant entraîner une imperméabilisation partielle ou totale, afin de les affecter notamment à des fonctions urbaines ou de transport (habitat, activités, commerces, infrastructures, équipements publics…). "
[Ministère de la transition éque](https://www.ecologie.gouv.fr/artificialisation-des-sols) ''')

list_images=[Image.open("DémoImages/Paris.jpg"), Image.open("DémoImages/Strasbourg.jpg"), Image.open("DémoImages/Lectoure.jpg"), Image.open("DémoImages/Montargis.jpg")]
list_captions=["Paris","Strasbourg","Lectoure","Montargis"]
st.image(list_images, list_captions, width=256)


## User input
user_input = st.selectbox('Quelle ville souhaitez-vous voir?',('Aucune ville','Lectoure', 'Paris', 'Strasbourg', 'Montargis'))
st.write('Vous avez choisi:', user_input)
 
if user_input == "Aucune ville":
    st.warning("Sélectionnez une ville s'il vous plaît.")
else:
    if user_input == "Paris":
        im=Image.open("DémoImages/Paris.jpg")
        im_size=im.size 
    elif user_input == "Strasbourg":
        im=Image.open("DémoImages/Strasbourg.jpg")
        im_size=im.size
    elif user_input == "Lectoure":
        im=Image.open("DémoImages/Lectoure.jpg")
        im_size=im.size
    elif user_input == "Montargis":
        im=Image.open("DémoImages/Montargis.jpg")
        im_size=im.size
 
    ### Getting smaller batches - type(data) and type(image_crops) are numpy.ndarray
    data = np.array(im)
    image_crops = get_patches(img_arr=data)
    image_crops_normalized = np.asarray(image_crops, dtype=np.float32)/image_crops.max()
    
    ### Import model and use it
    from keras_unet.models import satellite_unet
    model = satellite_unet(input_shape=(256, 256, 3))
    model_filename = 'artificialisation_model_06092021.h5'
    model.load_weights(model_filename)
    masks = model.predict(image_crops_normalized)
    
    ### Get the data ready (reshape, stack arrays)
    list_masks=[]
    for i in masks:
     a = cv2.threshold(i, 0.15, 255, cv2.THRESH_BINARY)
     list_masks.append(a[1])
     
    new_list_masks=[]
    for i in list_masks:
     b=np.stack((i, i, i), axis=2)
     new_list_masks.append(b)
     
    array_masks=np.array(new_list_masks)
    
    ### Print final data
    from keras_unet.utils import reconstruct_from_patches

    st.write("x_crops shape: ", str(image_crops.shape))
    x_reconstructed = reconstruct_from_patches(img_arr=array_masks, org_img_size=im_size)

    st.write("x_reconstructed shape: ", str(x_reconstructed.shape))

    plt.figure(figsize=(10,10))
    plt.imshow(x_reconstructed[0])
    plt.show()
