#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
#from sensor_msgs.msg import Joy
from std_msgs.msg import Int16



speed = rospy.get_param('~speed', 0.1)
do_nothing = True
bool()

def cnn_callback(data):
        value = data.data	   
        if value ==1 :
		do_nothing = False
		print('breaking')
 		breaking = True
		twist = Twist()
		twist.linear.x = speed
        	pub.publish(twist)
		do_nothing = False
	elif value == 25:
	        print('acclelerating')
		accel = True
        elif value == 17:
		print('curve left')
		cruve_left = True
        elif value == 18:
		print('curve right')
	        curve_right = True
        else:
		print('do nothing')
		do_nothing = True

def vel_callback(data):		

    
	#twist.linear.x = 4*data.axes[1]
	#twist.angular.z = 4*data.axes[0]
	if do_nothing == True:
        	twist = data
        	pub.publish(twist)
	



# Intializes everything
def start():
	# publishing to "turtle1/cmd_vel" to control turtle1
	global pub
	#pub = rospy.Publisher('turtle1/cmd_vel', Twist)

	pub = rospy.Publisher('RosAria/cmd_vel', Twist, queue_size = 0)

	# subscribed to joystick inputs on topic "joy"
	#rospy.Subscriber("joy", Joy, callback)
    

	#breaking = False
	#accel = False
	#curve_left = False
	#curve_right = False
	#do_nothing = False

	# starts the node
	rospy.init_node('listener')

	rospy.Subscriber('result', Int16, cnn_callback)
	rospy.Subscriber('cmd_vel', Twist, vel_callback)
	
	rospy.spin()

if __name__ == '__main__':
	start()
