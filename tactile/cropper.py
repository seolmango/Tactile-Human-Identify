"""
Tactile 센서를 통해 입력되는 압력 데이터에서 발만 잘라내기 위해 모듈입니다.
"""
import numpy as np
from tactile.tools import image_watchmap, generate_center_increasing_array

class Foot_cropper:
    def __init__(self, input_x, input_y, crop_x, crop_y, filter_x, filter_y, threshold):
        """
        :param input_x: 입력 데이터의 x 길이
        :param input_y: 입력 데이터의 y 길이
        :param crop_x: 크롭된 데이터의 x 길이
        :param crop_y: 크롭된 데이터의 y 길이
        :param filter_x: 필터의 x 길이
        :param filter_y: 필터의 y 길이
        :param threshold: 경험적으로 찾는 한계점
        """
        if input_x < crop_x or input_y < crop_y:
            raise ValueError("크롭할 데이터가 입력 데이터보다 큽니다.")
        self.input_x = input_x
        self.input_y = input_y
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.filter_x = filter_x
        self.filter_y = filter_y
        self.threshold = threshold
        self.filter = generate_center_increasing_array(filter_x, filter_y)

    def crop(self, frame, before_frame):
        """
        :param frame: 발의 위치를 크롭할 프레임
        :param before_frame: 그 전 프레임
        :return:
        """
        now_watchmap = image_watchmap(frame)
        before_watchmap = image_watchmap(before_frame)
        watchmap = now_watchmap * before_watchmap
        feature_map = []
        for i in range(0, self.input_x - self.filter_x + 1):
            for j in range(0, self.input_y - self.filter_y + 1):
                temp = (watchmap[i:i+self.filter_x, j:j+self.filter_y] * self.filter).sum()
                if temp > self.threshold:
                    feature_map.append(temp)
                else:
                    feature_map.append(0)
        feature_map = np.array(feature_map)
        feature_map = feature_map.reshape(self.input_x - self.filter_x + 1, self.input_y - self.filter_y + 1)
        for i in range(self.input_x - self.filter_x + 1):
            for j in range(self.input_y - self.filter_y + 1):
                start_x = max(0, i - self.filter_x)
                end_x = min(self.input_x - self.filter_x + 1, i + self.filter_x)
                start_y = max(0, j - self.filter_y)
                end_y = min(self.input_y - self.filter_y + 1, j + self.filter_y)
                if feature_map[i, j] != feature_map[start_x:end_x, start_y:end_y].max():
                    feature_map[i, j] = 0
        return np.where(feature_map != 0)

    def crop_image(self, frame, loc):
        """
        :param frame: 발의 위치를 크롭할 프레임
        :param loc: 발의 위치
        :return: 크롭된 이미지
        """
        x, y = loc
        images = []
        for i in range(0, len(x)):
            center_x = x[i] + (self.filter_x - 1) // 2
            center_y = y[i] + (self.filter_y - 1) // 2
            center_x = min(max(center_x, (self.crop_x-1) // 2), self.input_x - (self.crop_x+1) // 2)
            center_y = min(max(center_y, (self.crop_y-1) // 2), self.input_y - (self.crop_y+1) // 2)
            images.append(frame[center_x - (self.crop_x-1) // 2:center_x + (self.crop_x+1) // 2, center_y - (self.crop_y-1) // 2:center_y + (self.crop_y+1) // 2])
        return np.array(images)