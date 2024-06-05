import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

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
    # pre-processing
    watchmap = np.zeros(data.shape)
    for x in range(1, data.shape[0]-1):
        for y in range(1, data.shape[1]-1):
            if abs(data[x][y]-data[x-1][y]) < 200 and abs(data[x][y]-data[x][y-1]) < 200 and abs(data[x][y]-data[x+1][y]) < 200 and abs(data[x][y]-data[x][y+1]) < 200:
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


def live_visualize(image, loc=None, result=None, labels=None):
    image = copy.deepcopy(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    size = len(image)
    if loc is None:
        loc = [[], []]
    image = cv2.resize(image, (500, 500))
    temp = []
    for count in range(len(loc[0])):
        temp.append((loc[0][count], loc[1][count]))
    for index, location in enumerate(temp):
        center_x = location[0] + (15 - 1) // 2
        center_y = location[1] + (15 - 1) // 2
        center_x = min(max(center_x, (25 - 1) // 2), size - (25 + 1) // 2)
        center_y = min(max(center_y, (25 - 1) // 2), size - (25 + 1) // 2)
        cv2.rectangle(image, (int((center_y - 12) * (500 / size)), int((center_x - 12) * (500 / size))),
                      (int((center_y + 12) * (500 / size)), int((center_x + 12) * (500 / size))), (255, 0, 0), 2)
        if result is not None and labels is not None:
            cv2.putText(image, labels[result[index]],
                        (int((center_y - 12) * (500 / size)), int((center_x) * (500 / size)) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
    cv2.imshow("Pressure", image)
    if cv2.waitKey(1) & 0xff == 27:
        return False
    else:
        return True
