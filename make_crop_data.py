from tactile import cropper, data_loader, tools
import numpy as np
from tqdm import tqdm

DataLoader = data_loader.DataLoader(
    './230823_walkingData',
    {
        'cyh': ['1', '2'],
        'ksh': ['1', '2'],
        'sch': ['1', '2']
    }
)
Cropper = cropper.foot_cropper(
    64, 64, 25, 25, 15, 15, 100
)

image = []
label = []
for index, keys in enumerate(DataLoader.keys):
    for frame in tqdm(range(1, DataLoader.get_data_length(keys))):
        now = DataLoader.get_data(keys, frame)
        before = DataLoader.get_data(keys, frame-1)
        loc = Cropper.crop(now, before)
        images = Cropper.crop_image(now, loc)
        for count, img in enumerate(images):
            image.append(img)
            label.append(index)
    print(f"{keys} done")
image = np.array(image)
label = np.array(label)
print(image.shape, label.shape)
np.save('./crops/image.npy', image)
np.save('./crops/label.npy', label)