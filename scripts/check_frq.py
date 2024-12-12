import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import UInt8

class MessageFrequencyChecker:
    def __init__(self):
        self.message_count = 0
        self.last_msg_time = 0
        self.last_id = 0
        rospy.Subscriber('/vrpn_client_node/frisbee1/pose', PoseStamped, self.callback, queue_size=1)

    def callback(self, msg:UInt8):
        current_time = msg.header.stamp.to_sec()
        current_id = msg.header.seq

        frq = 1.0 / (current_time - self.last_msg_time)
        self.last_msg_time = current_time
        self.last_id = current_id

        if frq > 110:
            # rospy.loginfo(f"Frequency: {frq} Hz")
            pass
        if frq < 110 and frq > 100:
            # roswarn
            rospy.logwarn(f"Frequency: {frq} Hz")
        if frq < 100 and frq > 30:
            rospy.logerr(f"Frequency: {frq} Hz")
        if frq < 30:
            # print in purple
            print('\033[95m' + f"Frequency: {frq} Hz" + '\033[0m')



if __name__ == "__main__":
    rospy.init_node("message_frequency_checker", anonymous=True)
    MessageFrequencyChecker()
    rospy.spin()
