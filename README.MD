# Program description:

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
                      low_freq_num: int
                  ], 
                  ...
              ]
```

#### Maintainer: Huynn