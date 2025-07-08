import rospy
from std_msgs.msg import String
from spoof_nodes.msg import patient_data

def patient_raw():
    pub = rospy.Publisher('patient_data', String, queue_size=10)
    rospy.init_node('patient_raw', anonymous=True)
    rate = rospy.Rate(333) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()
  
if __name__ == '__main__':
    try:
        patient_raw()
    except rospy.ROSInterruptException:
        pass