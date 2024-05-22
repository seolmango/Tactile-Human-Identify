from sensors.sensors import SensorEnv
import numpy as np
import cv2
import os
from serial.serialutil import SerialException

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
    file_list = os.listdir(path)
    if f"{data_name}.npy" in file_list:
        print("The file already exists.")
        if str(input("Do you want to add data to the file? (y/n)> ")) == "n":
            print("The program is terminated.")
            return
        else:
            before_data = np.load(f"{path}{data_name}.npy")
    else:
        before_data = []
    max_frame = int(input("Enter the number of frames to collect> "))
    new_data = []
    for i in range(max_frame):
        images = sensor.get()
        image = images[-1]
        print(f"frame : {i+1}, sensor FPS : {sensor.fps}, images shape : {images.shape}", flush=True)
        new_data.append(image)
        visualize(image)
    sensor.close()
    before_data.extend(new_data)
    np.save(f"{path}{data_name}.npy", before_data)
    print(f"Data saved. file path : {path}{data_name}.npy")


if __name__ == "__main__":
    main()