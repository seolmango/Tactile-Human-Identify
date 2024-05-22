from sensors.sensors import SensorEnv
#from model.VisionModel_isaac import FootDetector as isaac_model
#from model.VisionModel_yunho import FootDetector as yunho_model
#from model.VisionModel_Integrated import FootDetector as integrated_model
#from model.model_sh import CarpetClassifier
from sensors.app.FramerateMonitor import FramerateMonitor
#from unreal.env import DogChasingEnv
import numpy as np
import cv2
import time
import os
import pickle
import datetime
import torch
#import pygame
#import sys


def visualize(image):
    if image.dtype != np.uint8:
        image *= 255
        image[image < 0] = 0
        image = image.astype(np.uint8)
    image = cv2.resize(image, (500, 500))
    cv2.imshow("Pressure", image)
    if cv2.waitKey(1) & 0xff == 27:
        return False
    else:
        return True

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

def test_sensor():
    sensor = init_sensor()
    while True:
        images = sensor.get()
        image = images[-1]
        image = (image - image.min()) / (image.max() - image.min())
        if not visualize(image):
            break
        print(f"sensor FPS : {sensor.fps}, images shape : {images.shape}")
    sensor.close()

if __name__ == "__main__":
    # main(save_log=True, log_dir=".\\logs")
    #main()
    # test_env()
    test_sensor()
    # test_model(save_log=True, log_dir=".\\logs")