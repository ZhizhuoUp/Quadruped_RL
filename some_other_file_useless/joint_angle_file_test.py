import pybullet_data as pd
import matplotlib.pyplot as plt
import numpy as np

FR_hip_motor_2_chassis_joint = []
FR_upper_leg_2_hip_motor_joint = []
FR_lower_leg_2_upper_leg_joint = []
FL_hip_motor_2_chassis_joint = []
FL_upper_leg_2_hip_motor_joint = []
FL_lower_leg_2_upper_leg_joint = []
RR_hip_motor_2_chassis_joint = []
RR_upper_leg_2_hip_motor_joint = []
RR_lower_leg_2_upper_leg_joint = []
RL_hip_motor_2_chassis_joint = []
RL_upper_leg_2_hip_motor_joint = []
RL_lower_leg_2_upper_leg_joint = []

joint_1 = FR_hip_motor_2_chassis_joint
joint_2 = FR_upper_leg_2_hip_motor_joint
joint_3 = FR_lower_leg_2_upper_leg_joint
joint_4 = FR_hip_motor_2_chassis_joint
joint_5 = FR_upper_leg_2_hip_motor_joint
joint_6 = FR_lower_leg_2_upper_leg_joint
joint_7 = FR_hip_motor_2_chassis_joint
joint_8 = FR_upper_leg_2_hip_motor_joint
joint_9 = FR_lower_leg_2_upper_leg_joint
joint_10 = FR_hip_motor_2_chassis_joint
joint_11 = FR_upper_leg_2_hip_motor_joint
joint_12 = FR_lower_leg_2_upper_leg_joint


with open("urdf_zzz/data1.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",")
        joints = currentline[2:14]
        value = [float(s) for s in joints]
        FR_hip_motor_2_chassis_joint.append(value[0])
        FR_upper_leg_2_hip_motor_joint.append(value[1])
        FR_lower_leg_2_upper_leg_joint.append(value[2])
        FL_hip_motor_2_chassis_joint.append(value[3])
        FL_upper_leg_2_hip_motor_joint.append(value[4])
        FL_lower_leg_2_upper_leg_joint.append(value[5])
        RR_hip_motor_2_chassis_joint.append(value[6])
        RR_upper_leg_2_hip_motor_joint.append(value[7])
        RR_lower_leg_2_upper_leg_joint.append(value[8])
        RL_hip_motor_2_chassis_joint.append(value[9])
        RL_upper_leg_2_hip_motor_joint.append(value[10])
        RL_lower_leg_2_upper_leg_joint.append(value[11])
    # print(FR_hip_motor_2_chassis_joint)

plt.plot(np.arange(len(joint_4)), joint_4)
plt.show()