from sensors.sensors import SensorEnv
import numpy as np
import cv2
from tactile import cropper
from tactile.tools import live_visualize


Cropper = cropper.Foot_cropper(
    32, 32, 25, 25, 15, 15, 100
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

def crop_test():
    sensor = init_sensor()
    before = sensor.get()[-1]
    while True:
        images = sensor.get()
        image = images[-1]
        loc = Cropper.crop(image, before)
        live_visualize(image, loc=loc)
        print(f"sensor FPS : {sensor.fps}, images shape : {images.shape}, loc : {loc}, image_min : {image.min()}, image_max : {image.max()}")
        before = image
    sensor.close()

if __name__ == "__main__":
    crop_test()