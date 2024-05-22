from tactile.data_loader import DataLoader
from tactile.cropper import Foot_cropper
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from PIL import Image

DataLoader = DataLoader(
    '../live_run',
    {
        'test': ['sch']
    }
)
Cropper = Foot_cropper(
    32, 32, 25, 25, 15, 15, 100
)

image = frames = []
for index, keys in enumerate(DataLoader.keys):
    for frame in tqdm(range(1, DataLoader.get_data_length(keys))):
    # for frame in tqdm(range(50, 150)):
        now = DataLoader.get_data(keys, frame)
        before = DataLoader.get_data(keys, frame - 1)
        loc = Cropper.crop(now, before)
        images = Cropper.crop_image(now, loc)
        temp = []
        for count, img in enumerate(images):
            temp.append((loc[0][count], loc[1][count]))
        plt.cla()
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        plt.gca().set_aspect('equal')
        ax.set_title(f'{keys}_frame: {frame}')
        plt.imshow(now, cmap='gray')
        for loc in temp:
            center_x = loc[0] + (15 - 1) // 2
            center_y = loc[1] + (15 - 1) // 2
            center_x = min(max(center_x, (25 - 1) // 2), 32 - (25 + 1) // 2)
            center_y = min(max(center_y, (25-1) // 2), 32 - (25+1) // 2)
            rect = patches.Rectangle((center_y-12, center_x-12), 25, 25, linewidth=1, edgecolor='r', facecolor='none')
            center_point = patches.Circle((center_y, center_x), radius=1, color='r')
            ax.add_patch(rect)
            ax.add_patch(center_point)
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image = np.array(Image.fromarray(image).resize((1920, 1080)))
        ax.clear()
        plt.close(fig)
        plt.close('all')
        frames.append(image)

    output_file = f'../data_make/{keys}.mp4'
    writer = imageio.get_writer(output_file, fps=12, macro_block_size=None, format='MP4')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f'{keys} is saved')
