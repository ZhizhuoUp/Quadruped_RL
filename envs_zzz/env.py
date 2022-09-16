import pybullet as p
import pybullet_data
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import math
import random
import time
import math
import torch
import os

class LaikagoEnv(gym.Env):

    MaxEpisodeSteps = 700

    def __init__(self, is_render=False, is_good_view=False):
                
        self.RoboID = None
        self.sleep_t = 1. / 240  # decrease the value if it is too slow.
        self.maxVelocity = 10
        self.force = 100
        self.n_sim_steps = 1
        self.startPos = [0, 0, 0.5]
        self.startOri = [np.pi / 2, 0, 0]
        self.initial_action = []
        self.jointIds = []

        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.step_counter = 0

        self.low_action_0 = -0.52 # 30degree
        self.low_action_1 = 0
        self.low_action_2 = -1.80 # >90degree
        self.low_action_3 = -0.52
        self.low_action_4 = 0
        self.low_action_5 = -1.80
        self.low_action_6 = -0.52
        self.low_action_7 = 0
        self.low_action_8 = -1.80
        self.low_action_9 = -0.52
        self.low_action_10 = 0
        self.low_action_11 = -1.80

        self.high_action_0 = 0.52
        self.high_action_1 = 1.04 # 60degree
        self.high_action_2 = -0.52
        self.high_action_3 = 0.52
        self.high_action_4 = 1.04
        self.high_action_5 = -0.52
        self.high_action_6 = 0.52
        self.high_action_7 = 1.04
        self.high_action_8 = -0.52
        self.high_action_9 = 0.52
        self.high_action_10 = 1.04
        self.high_action_11 = -0.52

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # action：各关节角度
        self.action_space = spaces.Box(
            low=np.array([self.low_action_0, self.low_action_1, self.low_action_2, self.low_action_3, 
                        self.low_action_4, self.low_action_5, self.low_action_6, self.low_action_7, 
                        self.low_action_8, self.low_action_9, self.low_action_10, self.low_action_11, ]),
            high=np.array([self.high_action_0, self.high_action_1, self.high_action_2, self.high_action_3, 
                        self.high_action_4, self.high_action_5, self.high_action_6, self.high_action_7, 
                        self.high_action_8, self.high_action_9, self.high_action_10, self.high_action_11, ]),
            dtype=np.float32)

        # observation：角度和角速度
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(14,))
        
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.step_counter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")  # URDF Id = 0
        self.RoboID = p.loadURDF("laikago/laikago.urdf", self.startPos,
                                 p.getQuaternionFromEuler(self.startOri))  # URDF Id = 1

        # for zzz in range(p.getNumJoints(self.RoboID)):
        #     joint_id = zzz
        #     joint_info = p.getJointInfo(self.RoboID, joint_id)
        #     print(joint_info)

        self.jointOffsets = []
        self.jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]

        for i in range(4):
            self.jointOffsets.append(0)
            self.jointOffsets.append(-0.7)
            self.jointOffsets.append(0.7)

        for j in range(p.getNumJoints(self.RoboID)):
            p.changeDynamics(self.RoboID, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.RoboID, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)

        p.stepSimulation()

        # base的 位置/角度 和 速度/角速度
        pos = p.getBasePositionAndOrientation(self.RoboID)
        # print('the pos of base is: ', pos[0][1])
        ang = p.getBaseVelocity(self.RoboID)
        # print('the velocity of base is: ', ang[0])

        # x: pitch y: yaw z: roll
        pitch = pos[1][0]
        yaw = pos[1][1]
        roll = pos[1][2]
        pitch_velocity = ang[1][0]
        yaw_velocity = ang[1][1]
        roll_velocity = ang[1][2]

        obs_angle = []
        obs_angle.append(p.getLinkState(self.RoboID, 1)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 2)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 4)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 5)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 7)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 8)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 10)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 11)[1][0])
        
        obs_base_angle = pitch + yaw + roll
        obs_velocity = pitch_velocity + yaw_velocity + roll_velocity

        # self.observation = obs_angle + obs_base_angle + obs_velocity
        observation = []
        observation = [*obs_angle, pitch, yaw, roll, pitch_velocity, yaw_velocity, roll_velocity]
        # observation = np.array(observation).reshape(14, 1)
        # print("observation的大小")
        # print(observation)
        # obs_angle + obs_base_angle + obs_velocity


        return observation

    def step(self, action):

        for j in range(12):
            targetPos = float(action[j])
            targetPos = self.jointDirections[j] * targetPos + self.jointOffsets[j]
            p.setJointMotorControl2(bodyIndex=self.RoboID,
                                    jointIndex=self.jointIds[j],
                                    targetPosition=targetPos,
                                    controlMode=p.POSITION_CONTROL,
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        
        p.stepSimulation()
        # for _ in range(self.n_sim_steps):
        #     p.stepSimulation()
        #     time.sleep(1. / 50)
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        return self._reward()

    def _reward(self):
        
        # base的 位置/角度 和 速度/角速度
        pos = p.getBasePositionAndOrientation(self.RoboID)
        # print('the pos of base is: ', pos[0][1])
        ang = p.getBaseVelocity(self.RoboID)
        # print('the velocity of base is: ', ang[0])

        # x: pitch y: yaw z: roll
        pitch = pos[1][0]
        yaw = pos[1][1]
        roll = pos[1][2]
        pitch_velocity = ang[1][0]
        yaw_velocity = ang[1][1]
        roll_velocity = ang[1][2]

        # target的位置
        target = [0, -5]
        distance = math.sqrt((pos[0][0] - target[0]) ** 2 + (pos[0][1] - target[1]) ** 2)

        if distance < 0.5:
            reward = 1
            self.terminated = True
        elif yaw > 0.52 or yaw < -0.52: # +-30degree
            reward = -0.0001
            self.terminated = False
        elif roll > 0.26 or roll < -0.26: # +-15degree
            reward = -0.0001
            self.terminated = False
        elif self.step_counter > self.MaxEpisodeSteps:
            reward = 0.1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False

        obs_angle = []
        obs_angle.append(p.getLinkState(self.RoboID, 1)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 2)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 4)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 5)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 7)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 8)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 10)[1][0])
        obs_angle.append(p.getLinkState(self.RoboID, 11)[1][0])
        
        # obs_base_angle = pitch + yaw + roll
        # obs_velocity = pitch_velocity + yaw_velocity + roll_velocity

        observation = []
        observation = [*obs_angle, pitch, yaw, roll, pitch_velocity, yaw_velocity, roll_velocity]
        # observation = np.array(observation).reshape(14, 1)
        # observation.append(obs_angle)
        # observation.append(pitch)
        # observation.append(yaw)
        # observation.append(roll)
        # observation.append(pitch_velocity)
        # observation.append(yaw_velocity)
        # observation.append(roll_velocity)

        return observation, reward, self.terminated

    def close(self):
        p.disconnect()

if __name__ == '__main__':
    # 这一部分是做baseline
    import matplotlib.pyplot as plt

    env = LaikagoEnv(is_render=True, is_good_view=True)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    for _ in range(1000):
        action=env.action_space.sample()
        print(action)
        zzz_try = env.step(action)
