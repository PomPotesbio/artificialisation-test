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

