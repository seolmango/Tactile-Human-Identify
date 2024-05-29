from sensors.sensors import SensorEnv
import numpy as np
import cv2
from tactile.cropper import Foot_cropper
from tactile.classifier import Classifier
import numpy as np

cropper = Foot_cropper(
    64,64,25,25,15,15,100
)

classifier = Classifier("../data_make/model.pth")
labels = ['alpha-sch', 'alpha-ksh', 'alpha-hym', 'alpha-ojj', 'alpha-pjj', 'alpha-rhs', 'cant recognize']

def visualize(image, loc, result):
    if image.dtype != np.uint8:
        image *= 255
        image[image < 0] = 0
        image = image.astype(np.uint8)
    image = (image - image.min()) / (image.max() - image.min())
    image = cv2.resize(image, (500, 500))
    temp = []
    for count in range(len(loc[0])):
        temp.append((loc[0][count], loc[1][count]))
    for index,location in enumerate(temp):
        center_x = location[0] + (15 - 1) // 2
        center_y = location[1] + (15 - 1) // 2
        center_x = min(max(center_x, (25 - 1) // 2), 64 - (25 + 1) // 2)
        center_y = min(max(center_y, (25 - 1) // 2), 64 - (25 + 1) // 2)
        cv2.rectangle(image, (int((center_y - 12)*(500/64)), int((center_x - 12)*(500/64))), (int((center_y + 12)*(500/64)), int((center_x + 12)*(500/64))), (255, 0, 0), 2)
        cv2.putText(image, labels[result[index]], (int((center_y - 12)*(500/64)), int((center_x)*(500/64)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Pressure", image)
    if cv2.waitKey(1) & 0xff == 27:
        return False
    else:
        return True

def init_sensor():
    print("initializing sensors...")
    sensor = SensorEnv(
        ports=["COM6", "COM5", 'COM8', 'COM7'],
        #ports=["COM7"],
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
            image = image * 255
            loc = cropper.crop(image, before)
            crop_images = cropper.crop_image(image, loc)
            crop_images = (crop_images - 718080) / (1044225-718080)
            result = []
            before = image
            for i in range(len(crop_images)):
                result.append(classifier.predict(crop_images[i]))
        else:
            before = image
            result = []
            loc = [[], []]
        visualize(image, loc, result)
    sensor.close()

if __name__ == "__main__":
    # main(save_log=True, log_dir=".\\logs")
    #main()
    # test_env()
    test_sensor()
    # test_model(save_log=True, log_dir=".\\logs")