from sensors.sensors import SensorEnv
import numpy as np
import cv2
from tactile import cropper

Cropper = cropper.Foot_cropper(
    32, 32, 25, 25, 15, 15, 100
)

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

def crop_test():
    sensor = init_sensor()
    before = sensor.get()[-1]
    while True:
        images = sensor.get()
        image = images[-1]
        loc = Cropper.crop(image, before)
        x, y = loc
        for i in range(0, len(x)):
            center_x = x[i] + (15 - 1) // 2
            center_y = y[i] + (15 - 1) // 2
            center_x = min(max(center_x, (25-1) // 2), 32 - (25+1) // 2)
            center_y = min(max(center_y, (25-1) // 2), 32 - (25+1) // 2)
            cv2.rectangle(image, (center_y - 12, center_x - 12), (center_y + 13, center_x + 13), (255, 0, 0), 1)
        if not visualize(image):
            break
        print(f"sensor FPS : {sensor.fps}, images shape : {images.shape}")
        before = image
    sensor.close()

if __name__ == "__main__":
    crop_test()