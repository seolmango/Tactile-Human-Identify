"""
데이터 파일을 쉽게 다루기 위한 모듈
"""
import numpy as np
from .tools import make_ground_0, data_visualization

class DataLoader:
    def __init__(self, data_path, data_structure):
        """
        데이터를 쉽게 다룰 수 있게 하는 모듈
        :param data_path:
        :param data_structure:
        """
        self.data_path = data_path
        self.data_structure = data_structure
        self.keys = []
        self.data = {}
        self.load_data()
        return None

    def load_data(self):
        """
        데이터를 불러옵니다.
        """
        for key in self.data_structure:
            self.keys.append(key)
            self.data[key] = []
            for data_id in self.data_structure[key]:
                print(f"loading file > {self.data_path}/{key}/{data_id}.npy")
                data = np.load(f"{self.data_path}/{key}/{data_id}.npy")
                self.data[key].extend(data)
        return None

    def get_data(self, key, frame=None):
        """
        데이터를 가져옵니다.
        :param key: 데이터의 키(사람)
        :param frame: 데이터의 프레임
        :return: 데이터
        """
        if frame is not None:
            return self.data[key][frame]
        return self.data[key]

    def get_data_length(self, key):
        """
        데이터의 길이를 가져옵니다.
        :param key: 데이터의 키(사람)
        :return: 데이터의 길이
        """
        return len(self.data[key])

    def data_visualization(self, key, frame):
        """
        데이터를 시각화합니다.
        :param key: 데이터의 키(사람)
        :param frame: 데이터의 프레임
        """
        data = self.get_data(key, frame)
        data = make_ground_0(data)
        data_visualization(data, f"{key}_{frame}")
        return None