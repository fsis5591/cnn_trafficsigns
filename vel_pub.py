#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16


class Server:
	def __init__(self):
        	self.cnn = None
        	self.vel = None

	def cnn_callback(self, msg):
        	# "Store" message received.
		self.cnn = msg.data

		# Compute stuff.
		self.pub_speed()

	def vel_callback(self, msg):
		# "Store" the message received.
		self.vel = msg

		# Compute stuff.
		self.pub_speed()

	def pub_speed(self):
		#if self.orientation is not None and self.velocity is not None:
		#    pass  # Compute something.
		print(self.cnn)
		if self.cnn == 1: 
			print('breaking')
			speed = 0.4
			twist = self.vel
			twist.linear.x = twist.linear.x*speed 
			twist.linear.y = twist.linear.y*speed
			twist.linear.z = twist.linear.z*speed
			pub.publish(twist)
		elif self.cnn == 25:
			print('acclelerating')
			speed = 2
			twist = self.vel
			twist.linear.x = twist.linear.x*speed 
			twist.linear.y = twist.linear.y*speed
			twist.linear.z = twist.linear.z*speed
			pub.publish(twist)
		elif self.cnn == 17:
			print('curve left')
			th = 0.5
			twist = self.vel
			twist.angular.z = th
			pub.publish(twist)
		elif self.cnn == 18:
			print('curve right')
			th = -0.5
			twist = self.vel
			twist.angular.z = th
			pub.publish(twist)
		else:
			print('do nothing')
			twist = Twist()
			twist = self.vel
			pub.publish(twist)

if __name__ == '__main__':
	rospy.init_node('vel_pub')
	global pub
	pub = rospy.Publisher('RosAria/cmd_vel', Twist, queue_size = 100)

	server = Server()

	rospy.Subscriber('result', Int16, server.cnn_callback)
	rospy.Subscriber('cmd_vel', Twist, server.vel_callback)

	rospy.spin()
