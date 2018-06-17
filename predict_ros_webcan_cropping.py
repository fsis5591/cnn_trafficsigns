"""


Predicao utilizando a camera do webcan
com ROS

rodar em diferentes abas:

-roscore
-rosrun cv_camera cv_camera_node
-rosrun image_view image_view image:=/cv_camera/image_raw
-python predict_ros_webcan.py image:=/cv_camera/image_raw
-rostopic echo /result

"""
import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import argparse
import imutils

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray
import argparse
import cv2
import numpy as np
import tensorflow as tf



def cnn_model_fn(features, mode):   
  """Model function for CNN."""
  input_layer = tf.reshape(features["x"], [-1, 56, 56, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=100,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)     

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=150,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=250,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)   

  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 250])  

  dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu) 

  dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)     

  dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)     

  logits = tf.layers.dense(inputs=dense2, units=62)



  predictions = {"classes":tf.argmax(input=logits, axis=1),}
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	


def detectcrop(image):
	img_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("Image", img_rgb)
	#cv2.waitKey(0)
	img_hsv=cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

	# lower mask (0-10)
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red = np.array([170,50,50])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

	# join masks
	mask = mask0+mask1

	# set my output img to zero everywhere except my mask
	output_img = image.copy()
	output_img[np.where(mask==0)] = 0

	output_hsv = img_hsv.copy()
	output_img = cv2.bitwise_and(output_img, output_img, mask= mask)

	gray1 = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

	blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)

	thresh1 = cv2.threshold(blurred1, 20, 255, cv2.THRESH_BINARY)[1]
	#cv2.imshow("Image", thresh1)
	#cv2.waitKey(0)

	cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	area0 = 0
       	x = 0
       	y = 0
	xm = 0
	ym = 0
	wm = 0
	hm = 0
	for c in cnts:
	    M = cv2.moments(c)

	    x,y,w,h = cv2.boundingRect(c)
	    #print(x,y,w,h)
	    #image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)    
	    #cv2.imshow("Image", image)
	    #cv2.waitKey(0)
	    area = w*h
	    if(area > area0):
	        xm = x
	        ym = y
	        wm = w
	        hm = h
	        area0 = area
	#print(xm,ym,wm,hm, "is the biggest area to croppp")
    	img_rgb = cv2.rectangle(image,(xm,ym),(xm+wm,ym+hm),(0,255,0),2) 
    	#cv2.imshow("Image", img_rgb)
    	#cv2.waitKey(0)
	#cortando com uma borda fina verde
	crop = img_rgb[ ym:hm+ym,xm:wm+xm]
	#cv2.imshow("Image", crop)
	#cv2.waitKey(0)
	#cv2.destroyWindow("Image")	
	return crop


class RosTensorFlow():
    def __init__(self):
        self._cv_bridge = CvBridge()
        self.estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/home/felipe/traffic-signs-tensorflow-master/estimator")
        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', Int16, queue_size=1)
                


    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
	cv_image_cropped = detectcrop(cv_image)
        cv_image_gray = cv2.cvtColor(cv_image_cropped, cv2.COLOR_RGB2GRAY)
        
        
        #ret,cv_image_binary = cv2.threshold(cv_image_gray,128,255,cv2.THRESH_BINARY_INV)
        cv_image_28 = cv2.resize(cv_image_gray,(56,56))
        cv_image_28 = np.float32(cv_image_28)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":cv_image_28},num_epochs=1,shuffle=False)
	predictions = self.estimator.predict(input_fn=predict_input_fn)
	predct = next(predictions, None)
	classes =  predct["classes"] 
        rospy.loginfo('%d' % classes)
        self._pub.publish(classes)
	

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()


