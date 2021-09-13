# Here is a list of code to add into the streamlit file


# Use smaller batches from user image with the model

from keras_unet.models import satellite_unet
model = satellite_unet(input_shape=(256, 256, 3))
# Assuming data is in data_crops as a numpy array of (x, 256, 256, 3) shape.
x = np.asarray(img_np, dtype=np.float32)/256
# Download the model
model_filename = 'artificialisation_model_25082021.h5'
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


# Use model

from keras_unet.models import satellite_unet
model = satellite_unet(input_shape=(256, 256, 3))
# Assuming data is in data_crops as a numpy array of (x, 256, 256, 3) shape.
x = np.asarray(img_np, dtype=np.float32)/256
# Download the model
model_filename = 'artificialisation_model_25082021.h5'
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

# Obtenir les infos d'une dale
import rasterio as rio
import rasterio.plot

import fiona
import pyproj
image=rio.open("...")
print(f"Nom du fichier image : {image.name}")
print(f"Projection associée : {image.crs}")
print(f"Couverture spatiale : {image.bounds}")
print(f"Nombre de bandes : {image.count}")


  im_name=image.name
    im_proj=image.crs
    im_boundingbox=image.bounds
    im_bands=image.count   
    st.write("L'image s'appelle:", im_name, "La taille de l'image est de:", im_size, "La projection de l'image est:", im_proj, "Ses limites dont définies par", im_boundingbox)
    
       
    image=rio.open(im)
    st.write(type(image))
    

    
    file_bytes = st.file_uploader("Upload a tile", type=("png", "jpg", "tif", "tiff", "jp2", "jpeg")) 

if file_bytes == None:
    st.warning('No file selected. Please select a file.')
else:
    im = Image.open(file_bytes)
    im_size=im.size
