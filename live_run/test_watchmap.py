from sensors.sensors import SensorEnv
import numpy as np
import cv2
from tactile.tools import image_watchmap, live_visualize

def init_sensor():
    print("initializing sensors...")
    sensor = SensorEnv(
        #ports=["COM6", "COM5", 'COM8', 'COM7'],
        ports=["COM7"],
        stack_num= 1, # was 20 / 2022.09.28
        adaptive_calibration= False,
        normalize=False
    )
    print("sensor init finish")
    return sensor

def test_sensor():
    sensor = init_sensor()
    while True:
        images = sensor.get()
        image = images[-1]
        watchmap = image_watchmap(image)
        live_visualize(watchmap)
    sensor.close()

if __name__ == "__main__":
    test_sensor()