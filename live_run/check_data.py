import numpy as np
from tactile.tools import data_visualization

image_data = np.load('../data_make/image.npy')
label_data = np.load('../data_make/label.npy')

print(np.min(image_data))
print(np.max(image_data))