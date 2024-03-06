import numpy as np
import matplotlib.pyplot as plt

def make_ground_0(data):
    """
    :param data: 데이터
    :return: 데이터의 최솟값으로 모두 빼줌
    """
    return data - np.min(data)

def data_visualization(data, title=''):
    """
    :param data:
    :param title:
    :return:
    """
    plt.cla()
    plt.title(f"{title}")
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.show()

def image_watchmap(data):
    data = make_ground_0(data)
    watchmap = np.zeros(data.shape)
    for x in range(1, data.shape[0]-1):
        for y in range(1, data.shape[1]-1):
            if abs(data[x][y]-data[x-1][y]) < 150 and abs(data[x][y]-data[x][y-1]) < 150 and abs(data[x][y]-data[x+1][y]) < 150 and abs(data[x][y]-data[x][y+1]) < 150:
                watchmap[x][y] = 0
            else:
                watchmap[x][y] = 1
    return watchmap

def generate_center_increasing_array(size_x, size_y):
    """
    :param size_x: x 길이
    :param size_y: y 길이
    :return: 중심으로 갈수록 값이 증가하는 배열
    """
    if size_x % 2 == 0 or size_y % 2 == 0:
        raise ValueError("size_x와 size_y는 홀수여야 합니다.")
    center_x = (size_x-1) / 2
    center_y = (size_y-1) / 2
    arr = np.empty((size_x, size_y), dtype=np.float64)
    for i in range(size_x):
        for j in range(size_y):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            arr[i, j] = int(distance)
    arr = np.ones((size_x, size_y), dtype=np.float64) * arr.max() - arr
    return arr * np.abs(arr)

def save_data_visualization(data, title, path='./'):
    """
    :param data:
    :param title:
    :param path:
    """
    plt.cla()
    plt.title(f"{title}")
    plt.imshow(data, cmap='gray')
    plt.savefig(f"{path}/{title}.png")