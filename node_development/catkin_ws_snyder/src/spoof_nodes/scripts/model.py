import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
def model_node():
    rospy.init_node('model_node', anonymous=True)

    rospy.Subscriber("patient_data", String, callback)

    rospy.spin()

if __name__ == '__main__':
    model_node()