"""
hdf5 파일을 쉽게 다루기 위한 모듈
"""
import h5py
import numpy as np
from .tools import make_ground_0, data_visualization

class DataLoader:
    def __init__(self, data_path, data_structure):
        """
        :param data_path: 데이터 경로
        :param data_structure: 데이터 구조
        """
        self.data_path = data_path
        self.data_structure = data_structure
        self.keys = []
        self.data = {}
        self.load_data()

    def load_data(self):
        """
        데이터를 불러옴
        """
        for key in self.data_structure:
            self.keys.append(key)
            self.data[key] = []
            for data_id in self.data_structure[key]:
                print(f"loading file > {self.data_path}/{key}/{data_id}.hdf5")
                with h5py.File(f"{self.data_path}/{key}/{data_id}.hdf5", 'r') as f:
                    data = f['pressure'][()][:int(f['frame_count'][()])]
                    for frame in data:
                        frame = np.array(frame, dtype=np.intc)
                        self.data[key].append(frame)

    def get_data(self, key, frame=None):
        """
        :param key: 데이터의 키(사람)
        :param frame: 데이터의 프레임
        :return: 데이터
        """
        if frame is not None:
            return self.data[key][frame]
        return self.data[key]

    def get_data_length(self, key):
        """
        :param key: 데이터의 키(사람)
        :return: 데이터의 길이
        """
        return len(self.data[key])

    def data_visualization(self, key, frame):
        """
        :param key: 데이터의 키(사람)
        :param frame: 데이터의 프레임
        """
        data = self.get_data(key, frame)
        data = make_ground_0(data)
        data_visualization(data, f"{key}_{frame}")