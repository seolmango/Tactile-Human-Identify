from tactile import cropper, data_loader
import numpy as np
from tqdm import tqdm

DataLoader = data_loader.DataLoader(
    '../live_run',
    {
        'alpha-hym': ['test1'],
        'alpha-ksh': ['test1', 'test2'],
        'alpha-sch': ['test1', 'test2'],
        'alpha-ojj': ['test1'],
        'alpha-pjj': ['test1'],
        'alpha-rhs': ['test1']
    }
)
Cropper = cropper.Foot_cropper(
    32, 32, 25, 25, 15, 15, 100
)

image = []
label = []
for index, keys in enumerate(DataLoader.keys):
    for frame in tqdm(range(1, DataLoader.get_data_length(keys))):
        now = DataLoader.get_data(keys, frame)
        now = np.array(now) / 255
        before = DataLoader.get_data(keys, frame-1)
        before = np.array(before) / 255
        loc = Cropper.crop(now, before)
        images = Cropper.crop_image(now, loc)
        for count, img in enumerate(images):
            image.append(img)
            label.append(index)
    print(f"{keys} done")
image = np.array(image)
label = np.array(label)
print(image.shape, label.shape)
np.save('../data_make/move_image.npy', image)
np.save('../data_make/move_label.npy', label)