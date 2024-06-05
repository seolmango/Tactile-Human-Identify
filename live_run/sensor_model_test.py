from sensors.sensors import SensorEnv
import numpy as np
import cv2
from tactile.cropper import Foot_cropper
from tactile.classifier import Classifier
import numpy as np
from tactile.tools import live_visualize

cropper = Foot_cropper(
    32,32,25,25,15,15,100
)

classifier = Classifier("../data_make/model.pth")
labels = ['sch','ksh','rhs']


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
    before = None
    while True:
        images = sensor.get()
        image = images[-1]
        if before is not None:
            loc = cropper.crop(image, before)
            crop_images = cropper.crop_image(image, loc)
            result = []
            before = image
            for i in range(len(crop_images)):
                crop_images[i] = (crop_images[i] - crop_images[i].min()) / (crop_images[i].max() - crop_images[i].min())
                result.append(classifier.predict(crop_images[i]))
            live_visualize(image, loc, result, labels)
        else:
            before = image
            live_visualize(image)
    sensor.close()

if __name__ == "__main__":
    # main(save_log=True, log_dir=".\\logs")
    #main()
    # test_env()
    test_sensor()
    # test_model(save_log=True, log_dir=".\\logs")