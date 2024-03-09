import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 데이터
image_data = np.load('.\data_make\image.npy')
label_data = np.load('.\data_make\label.npy')

print(image_data.shape)