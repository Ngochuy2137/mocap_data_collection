# **Program description:**

- Subscribes to a PoseStamped topic and records the trajectory of the object.
- Start recording when the user presses ENTER
- Interpolate velocity based on central difference method, not backward/forward difference method.
    - I interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if time steps are not equal.
- Swap y and z if necessary.
- Stop when the object reaches a certain height.
- Log a warning if the message frequency is lower than a certain threshold.
- Save the recorded trajectories to a file named 'trajectories.npz' when the user stops the program.

```bash
Data format:  [  
                  [   
                        points: [
                                    [x, y, z, vx, vy, vz, timestamp], 
                                    ...
                                ], 
                        msg_ids: [int],
                        time_stamps: [double]
                        low_freq_num: int
                  ], 
                  ...
              ]
```

- Some other features:
    - Avoiding messages missing:
        - Proper ENTER checking:
            - To get ENTER input from user, I create a thread for this task. However, when ENTER checking thread run continuously, along with main thread,
            it might lock the main thread and cause missing messages (check self.recording_lock variable)
            - Therefore, to avoid missing message, I use threading.Event to activate/deactivate thread that check user ENTER input.
            - The ENTER checking thread is activated only after finishing a new trajectory or when starting the program.
        - Message queueing problem avoidance:
            - When waiting for user to press ENTER, the program should not queue messages. Therefore, check the lock before processing messages (check self.recording_lock.acquire(blocking=False)):
                - If self.recording_lock.acquire(blocking=False), no locking, process the message.
                - If not, skip the message.
        - We can know how many messages are missed by checking topic field /topic/header/seq (read self.last_msg_id variable) (Level 1)
            - Message missing check -level 2: Before process a new trajectory, we also check if data is uncontinuous (check self.is_uncontinuous_data(msg_ids))
                - If data is uncontinuous, that means some messages are missing -> Check /topic/header/seq field

---

# Usage

## 1. Data collection

<aside>

**⚠️Notes**

- *If you get one of these log in terminal, please check the mocap system*
    
    > *Please check the connection between the mocap system and the computer which runs this subscriber.
    The program will stop now.*
    > 
    
    > *The data is not continuous, some messages from publisher might be missed*
    > 
    
    > Current message frequency is too low
    > 
- *If you get this log, please keep your object above MIN_HEIGHT_THRESHOLD before throwing*
    
    > The trajectory needs to have at least 2 points to be saved
    > 
</aside>

### **Maintainer: Huynn**