from tactile.data_loader import DataLoader
from tactile.classifier import Classifier
from tactile.cropper import Foot_cropper
import numpy as np
from tqdm import tqdm
from tactile.tools import save_data_visualization

DataLoader = DataLoader(
    '230823_walkingData',
    {
        'sch': ['1']
    }
)

Cropper = Foot_cropper(
    64, 64, 25, 25, 15, 15, 100
)

Classifier = Classifier('./data_make/model.pth')

label = ['cyh', 'ksh', 'sch']
wrong_images = []
for index, keys in enumerate(DataLoader.keys):
    for frame in tqdm(range(1, DataLoader.get_data_length(keys))):
        now = DataLoader.get_data(keys, frame)
        before = DataLoader.get_data(keys, frame - 1)
        loc = Cropper.crop(now, before)
        images = Cropper.crop_image(now, loc)
        for count, img in enumerate(images):
            result = Classifier.predict(img)
            if result != 2:
                wrong_images.append((img, result))

print(len(wrong_images))
for i, (img, result) in enumerate(wrong_images):
    save_data_visualization(img, f'{i}_{label[result]}.png', './wrong_images')