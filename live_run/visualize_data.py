# Visualize the data

import numpy as np
import matplotlib.pyplot as plt

from tactile import cropper

test_data = np.load('./alpha-ksh/test1.npy')


visualize_data = test_data[233]
plt.imshow(visualize_data)
plt.show()

Cropper = cropper.Foot_cropper(
    32, 32, 25, 25, 15, 15, 100
)

cropped = Cropper.crop(visualize_data, test_data[232])
cropped_image = Cropper.crop_image(visualize_data, cropped)

# make the 25 * 25 yellow box
plt.imshow(visualize_data)


plt.plot([cropped[1], cropped[1] + 25], [cropped[0], cropped[0]], 'y')
plt.plot([cropped[1], cropped[1]], [cropped[0], cropped[0] + 25], 'y')
plt.plot([cropped[1] + 25, cropped[1] + 25], [cropped[0], cropped[0] + 25], 'y')
plt.plot([cropped[1], cropped[1] + 25], [cropped[0] + 25, cropped[0] + 25], 'y')

plt.show()