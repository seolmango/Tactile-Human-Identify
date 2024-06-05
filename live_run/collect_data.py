from sensors.sensors import SensorEnv
import numpy as np
import cv2
import os
from serial.serialutil import SerialException
from tactile.cropper import Foot_cropper
from tactile.tools import live_visualize
from tqdm import tqdm


cropper = Foot_cropper(
    32,32,25,25,15,15,100
)
def init_sensor():
    print("initializing sensors...")
    sensor = SensorEnv(
        # ports=["COM10", "COM12", 'COM8', 'COM11'],
        ports=["COM7"],
        stack_num= 1, # was 20 / 2022.09.28
        adaptive_calibration= False,
        normalize=False
    )
    print("sensor init finish")
    return sensor


def main():
    try:
        sensor = init_sensor()
    except SerialException:
        print("Sensor not connected.")
        return
    data_label = str(input("Enter the label for the data> "))
    path = "./"
    file_list = os.listdir(path)
    if data_label not in file_list:
        os.mkdir(data_label)
    path = path + data_label + "/"
    data_name = str(input("Enter the name for the data> "))
    max_frame = int(input("Enter the number of frames to collect> "))
    new_data = []
    before = None
    for i in tqdm(range(max_frame)):
        images = sensor.get()
        image = images[-1]
        if before is not None:
            loc = cropper.crop(image, before)
            before = image
        else:
            before = image
            loc = [[], []]
        new_data.append(image)
        live_visualize(image, loc=loc)
    sensor.close()
    np.save(f"{path}{data_name}.npy", new_data)
    print(f"Data saved. file path : {path}{data_name}.npy")


if __name__ == "__main__":
    main()