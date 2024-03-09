from tactile.data_loader import DataLoader
from tactile.classifier import Classifier
from tactile.cropper import Foot_cropper
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from PIL import Image

DataLoader = DataLoader(
    '230823_walkingData',
    {
        'two': ['1']
    }
)
Cropper = Foot_cropper(
    64, 64, 25, 25, 15, 15, 100
)

Classifier = Classifier('./data_make/model.pth')

label = ['cyh', 'ksh', 'sch']
image = frames = []
for index, keys in enumerate(DataLoader.keys):
    for frame in tqdm(range(1, DataLoader.get_data_length(keys))):
        now = DataLoader.get_data(keys, frame)
        before = DataLoader.get_data(keys, frame - 1)
        loc = Cropper.crop(now, before)
        images = Cropper.crop_image(now, loc)
        temp = []
        for count, img in enumerate(images):
            temp.append((loc[count], Classifier.predict(img)))
        plt.cla()
        plt.figure(figsize=(16, 9))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_title(f'{keys}_frame: {frame}')
        plt.imshow(now, cmap='gray')
        for loc, label in temp:
            ax.text(loc[1], loc[0], label, fontsize=8, color='red')
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image = np.array(Image.fromarray(image).resize((1920, 1080)))
        ax.clear()
        plt.close(fig)
        frames.append(image)

    output_file = f'./data_make/{keys}.mp4'
    writer = imageio.get_writer(output_file, fps=30, macro_block_size=None, format='MP4')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f'{keys} is saved')

