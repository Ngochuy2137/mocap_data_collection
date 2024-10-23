import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

class MessageFrequencyChecker:
    def __init__(self):
        self.message_count = 0
        self.last_msg_time = 0
        self.last_id = 0
        rospy.Subscriber('/mocap_pose_topic/frisbee1_pose', PoseStamped, self.callback, queue_size=50)

    def callback(self, msg:PoseStamped):
        current_time = msg.header.stamp.to_sec()
        current_id = msg.header.seq

        # print('\n -------')
        # print('check 111: ', current_time, ' ', self.last_msg_time)
        # print('check 222: ', current_id, ' ', self.last_id)

        frq = 1.0 / (current_time - self.last_msg_time)
        self.last_msg_time = current_time
        self.last_id = current_id

        if frq > 110:
            rospy.loginfo(f"Frequency: {frq} Hz")
        if frq < 110 and frq > 100:
            # roswarn
            rospy.logwarn(f"Frequency: {frq} Hz")
        if frq < 100:
            rospy.logerr(f"Frequency: {frq} Hz")


if __name__ == "__main__":
    rospy.init_node("message_frequency_checker", anonymous=True)
    MessageFrequencyChecker()
    rospy.spin()
