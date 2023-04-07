import tensorflow as tf
import numpy as np
import cv2
import pathlib
from helper_functions import run_odt_and_draw_results_true
from helper_functions import preprocess_image
from helper_functions import detect_objects
import config
from PIL import Image
import os

import rospy
from std_msgs.msg import String
pub = rospy.Publisher('chatter', String, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(10) # 10hz

cwd = os.getcwd()

COLORS = np.random.randint(0, 255, size=(len(config.CLASSES), 3), dtype=np.uint8)

interpreter = tf.lite.Interpreter(model_path="/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/model/hardware1.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.allocate_tensors()

webcam = cv2.VideoCapture(0)
while not rospy.is_shutdown():
        
        ret, frame = webcam.read()
        out,label = run_odt_and_draw_results_true(frame,interpreter=interpreter,threshold=0.5)

        str = label % rospy.get_time()
        rospy.loginfo(str)
        pub.publish(str)
        rate.sleep()
        cv2.imshow("Objects",out)
        cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()