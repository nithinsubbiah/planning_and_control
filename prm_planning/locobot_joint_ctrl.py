# Import system libraries
import argparse
import os
import sys

# Modify the following lines if you have problems importing the V-REP utilities
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd,'lib'))
sys.path.append(os.path.join(cwd,'utilities'))

# Import application libraries
import numpy as np
import vrep_utils as vu
import matplotlib.pyplot as plt

# Import any other libraries you might want to use ############################
# ...
###############################################################################

class ArmController:

    def __init__(self):
        # Fill out this method ##################################
        # Define any variables you may need here for feedback control
        # ...
        #########################################################
        # Do not modify the following variables

        self.kp = 1.8
        self.kd = 0.6
        self.ki = 0.005



        self.timestamp_prev = 0

        self.error = np.zeros(7)
        self.error_integral = np.zeros(7)
        self.error_prev = np.zeros(7)
        self.error_dot = np.zeros(7)

        self.history = {'timestamp': [],
                        'joint_feedback': [],
                        'joint_target': [],
                        'ctrl_commands': []}
        self._target_joint_positions = None

    def set_target_joint_positions(self, target_joint_positions):
        assert len(target_joint_positions) == vu.N_ARM_JOINTS, \
            'Expected target joint positions to be length {}, but it was length {} instead.'.format(len(target_joint_positions), vu.N_ARM_JOINTS)
        self._target_joint_positions = target_joint_positions

    def calculate_commands_from_feedback(self, timestamp, sensed_joint_positions):
        assert self._target_joint_positions, \
            'Expected target joint positions to be set, but it was not.'


        # Fill out this method ##################################
        # Using the input joint feedback, and the known target joint positions,
        # calculate the joint commands necessary to drive the system towards
        # the target joint positions.
        ctrl_commands = np.zeros(vu.N_ARM_JOINTS)

        target_pos_np = np.array(self._target_joint_positions)
        current_joint_pos_np = np.array(sensed_joint_positions)

        self.error = np.subtract(target_pos_np,current_joint_pos_np)

        self.error_integral+=self.error

        dt = timestamp - self.timestamp_prev

        self.error_dot = np.true_divide(np.subtract(self.error,self.error_prev),dt)

        ctrl_commands = self.kp * self.error + self.kd * self.error_dot + self.ki * self.error_integral

        self.error_prev = self.error
        self.timestamp_prev = timestamp
        # ...
        #########################################################

        # Do not modify the following variables
        # append time history
        self.history['timestamp'].append(timestamp)
        self.history['joint_feedback'].append(sensed_joint_positions)
        self.history['joint_target'].append(self._target_joint_positions)
        self.history['ctrl_commands'].append(ctrl_commands)
        return ctrl_commands

    def has_stably_converged_to_target(self):
        # Fill out this method ##################################
        if np.linalg.norm(self.error,np.inf)<0.5:
            has_stably_converged_to_target = True
        else:
            has_stably_converged_to_target = False
        # ...
        #########################################################
        return has_stably_converged_to_target


def main(args):
    # Connect to V-REP
    print ('Connecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    # Reset simulation in case something was running
    vu.reset_sim(clientID)

    # Initial control inputs are zero
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))

    # Despite the name, this sets the maximum allowable joint force
    vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))

    # One step to process the above settings
    vu.step_sim(clientID)

    deg_to_rad = np.pi/180.

    # Joint targets. Specify in radians for revolute joints and meters for prismatic joints.
    # The order of the targets are as follows:
    #   joint_1 / revolute  / arm_base_link <- shoulder_link
    #   joint_2 / revolute  / shoulder_link <- elbow_link
    #   joint_3 / revolute  / elbow_link    <- forearm_link
    #   joint_4 / revolute  / forearm_link  <- wrist_link
    #   joint_5 / revolute  / wrist_link    <- gripper_link
    #   joint_6 / prismatic / gripper_link  <- finger_r
    #   joint_7 / prismatic / gripper_link  <- finger_l
    joint_targets = [[  0.,
                        0.,
                        0.,
                        0.,
                        0.,
                      - 0.07,
                        0.07], \
                     [-45.*deg_to_rad,
                      -15.*deg_to_rad,
                       20.*deg_to_rad,
                       15.*deg_to_rad,
                      -75.*deg_to_rad,
                      - 0.03,
                        0.03], \
                     [ 30.*deg_to_rad,
                       60.*deg_to_rad,
                      -65.*deg_to_rad,
                       45.*deg_to_rad,
                        0.*deg_to_rad,
                      - 0.05,
                        0.05]]

    # Instantiate controller
    controller = ArmController()

    i=0

    # Iterate through target joint positions
    for target in joint_targets:

        # Set new target position
        controller.set_target_joint_positions(target)

        steady_state_reached = False
        while not steady_state_reached:
            i+=1

            timestamp = vu.get_sim_time_seconds(clientID)
            print('Simulation time: {} sec'.format(timestamp))

            # Get current joint positions
            sensed_joint_positions = vu.get_arm_joint_positions(clientID)

            # Calculate commands
            commands = controller.calculate_commands_from_feedback(timestamp, sensed_joint_positions)

            # Send commands to V-REP
            vu.set_arm_joint_target_velocities(clientID, commands)

            # Print current joint positions (comment out if you'd like)
            #print(sensed_joint_positions)
            vu.step_sim(clientID, 1)

            if(i==1):
                joint_pos = np.array(sensed_joint_positions)
                target_pos = np.array(target)
                time_history = np.array(timestamp)
            else:
                joint_pos = np.vstack((joint_pos,sensed_joint_positions))
                target_pos = np.vstack((target_pos,target))
                time_history = np.hstack((time_history,timestamp))

            # Determine if we've met the condition to move on to the next point
            steady_state_reached = controller.has_stably_converged_to_target()

    vu.stop_sim(clientID)

    plt.figure(1)

    plt.subplot(331)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Angle(rad)')
    plt.title('Joint 1 position over time')
    plt.plot(time_history,joint_pos[:,0],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,0],'-r',label='Target position')

    plt.subplot(332)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Angle(rad)')
    plt.title('Joint 2 position over time')
    plt.plot(time_history,joint_pos[:,1],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,1],'-r',label='Target position')

    plt.subplot(333)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Angle(rad)')
    plt.title('Joint 3 position over time')
    plt.plot(time_history,joint_pos[:,2],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,2],'-r',label='Target position')

    plt.subplot(334)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Angle(rad)')
    plt.title('Joint 4 position over time')
    plt.plot(time_history,joint_pos[:,3],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,3],'-r',label='Target position')

    plt.subplot(335)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Angle(rad)')
    plt.title('Joint 5 position over time')
    plt.plot(time_history,joint_pos[:,4],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,4],'-r',label='Target position')

    plt.subplot(336)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Position(m)')
    plt.title('Joint 6 position over time')
    plt.plot(time_history,joint_pos[:,5],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,5],'-r',label='Target position')

    plt.subplot(337)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Position(m)')
    plt.title('Joint 7 position over time')
    plt.plot(time_history,joint_pos[:,6],'-b',label='Sensed position')
    plt.plot(time_history,target_pos[:,6],'-r',label='Target position')
    plt.show()



    # Post simulation cleanup -- save results to a pickle, plot time histories, etc #####
    # Fill this out here (optional) or in your own script
    # If you use a separate script, don't forget to include it in the deliverables
    # ...
    #####################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
