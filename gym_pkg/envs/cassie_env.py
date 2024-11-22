import gym
from gym import spaces
from .cassiemujoco import pd_in_t, CassieSim, CassieVis
import os
import random
import time
import copy
import gym
from gym import spaces
import numpy as np
from math import floor
from .reward.clockreward import create_phase_reward
from .utils.quaternion_function import quaternion2euler


class CassieEnv(gym.Env):
    # Metadata for the rendering options, setting it to 'human' mode
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self, simrate=60,config="./cassiemujoco/cassie.xml", target_height=0.9, 
                 fall_height=0.4, power_threshold=150, debug=False):
        self.config = config
        self.cassim = CassieSim(self.config)
        self.cassvis = None
        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.pdctrl = pd_in_t()
        self.time = 0
        self.last_pelvis_pos = self.cassim.qpos()[0:3]
        self.simrate = simrate
        
        self.encoder_noise = 0.01
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.motor_encoder_noise = np.zeros(10)
        # 定义动作空间，假设动作是 10 维连续的控制输入，范围是 [-1, 1]
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.debug = debug

        self.target_height = target_height
        self.fall_height = fall_height
        self.power_threshold = power_threshold

        # self.midfoot_offset = np.array([0.1762, 0.05219, 0., 0.1762, -0.05219, 0.])
        self.midfoot_offset = np.array([0., 0., 0., 0., 0., 0.])

        # 获取初始位置和速度
        self.init_qpos = np.copy(self.cassim.qpos())  # 35 维
        self.init_qvel = np.copy(self.cassim.qvel())
        obs_dim = len(self.init_qpos) + len(self.init_qvel)
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.speed = 0
        self.phase = 0
        self.phase_add = 1
        self.phaselen = 32
        self.strict_relaxer = 0.1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.have_incentive = True
        self.max_speed = 4.0
        self.min_speed = 1.0
        self.prev_action = None
        self.prev_torque = None
        self.neutral_foot_orient = np.array(
            [-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])

    
    def step(self, action, return_omniscient_state=False, f_term=0):
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Action {action} is invalid, expected shape: {self.action_space.shape}")
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        foot_pos = np.zeros(6)
        
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.l_foot_orient_cost = 0
        self.time += 1
        pos_pre = np.copy(self.cassim.qpos())[0]

        for _ in range(self.simrate):
            self._step_simulation(action)

            foot_forces = self.cassim.get_foot_forces()
            print('===============',self.cassim.get_foot_forces())
            if foot_forces is None:
                foot_forces = [1, 1]
            self.l_foot_frc += foot_forces[0]
            self.r_foot_frc += foot_forces[1]
            self.cassim.foot_pos(foot_pos)
            self.l_foot_pos += foot_pos[0:3]
            self.r_foot_pos += foot_pos[3:6]
            self.l_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.cassim.xquat("left-foot")) ** 2)
            self.r_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.cassim.xquat("right-foot")) ** 2)

        self.l_foot_frc /= self.simrate
        self.r_foot_frc /= self.simrate
        self.l_foot_pos /= self.simrate
        self.r_foot_pos /= self.simrate
        self.l_foot_orient_cost /= self.simrate
        self.r_foot_orient_cost /= self.simrate
        self.phase += self.phase_add

        if self.phase > self.phaselen:
            self.last_pelvis_pos = self.cassim.qpos()[0:3]
            self.phase = 0

        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])
        self.prev_action = action
        # update previous torque
        self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])

        rl_foot_pos = np.concatenate((self.l_foot_pos, self.r_foot_pos))
        rl_foot_grf = np.array([self.l_foot_frc, self.r_foot_frc])


        height = self.cassim.qpos()[2]

        reward = self._get_reward(rl_foot_pos) \
                 - self._get_cost(rl_foot_grf) if not (height < 0.4) or (height > 3.0) else 0.0
        
        
        

        
        terminated = bool((height < 0.4) or (height > 3.0))
        # truncated = False  # 如果有时间步数限制，可以设置为 True，这里假设没有额外的截断条件

        obs = self._get_obs()

        return obs, reward, terminated, {}

    def reset(self):
        # Reset the time step counter
        self.time = 0
        self.last_pelvis_pos = self.cassim.qpos()[0:3]
        self.cassie_state = self.cassim.step_pd(self.pdctrl)
        self.phase = random.randint(0, floor(self.phaselen))
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0

        self.speed = np.random.uniform(self.min_speed, self.max_speed)

        total_duration = (0.9 - 0.25 / 3.0 * abs(self.speed)) / 2
        self.swing_duration = (0.30 + ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
        self.stance_duration = (0.70 - ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
        self.stance_mode = "zero"
        self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration,
                                                                             self.stance_duration,
                                                                               self.strict_relaxer, self.stance_mode,
                                                                               self.have_incentive,
                                                                               FREQ=2000 // self.simrate)
        
        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
        
        
        # Apply small random noise to the initial positions and velocities
        c = 0.01  # Small noise constant
        new_qpos = self.init_qpos
        new_qvel = self.init_qvel

        # Set the new positions and velocities
        self.cassim.set_qpos(new_qpos)
        self.cassim.set_qvel(new_qvel)
        # self.set_slope()  # reset degree every time
        obs= self._get_obs()

        return obs

    
    
    # def render(self):
    #     if self.cassvis is None:
    #         self.cassvis = CassieVis(self.cassim, self.config)

    #     return self.cassvis.draw(self.cassim)
    def render(self, mode='human'):
        if mode == 'human':
            if self.cassvis is None:
                self.cassvis = CassieVis(self.cassim, self.config)
            self.cassvis.draw(self.cassim)
        else:
            super(CassieEnv, self).render(mode=mode)

    def close(self):
        if self.cassvis is not None:
            del self.cassvis  # Free the visualization resources
            self.cassvis = None

    def _step_simulation(self, action):
        
        # Set the target positions for the PD controller from the action
        target = action + self.offset
        target -= self.motor_encoder_noise
        
        
        foot_pos = np.zeros(6)
        self.cassim.foot_pos(foot_pos)
        prev_foot = copy.deepcopy(foot_pos)
        # Initialize the PD input structure
        self.pdctrl  = pd_in_t()
        foot_pos = np.zeros(6)
        self.cassim.foot_pos(foot_pos)
        # Loop through the first 5 motors for both legs and set the control gains
        for i in range(5):
            self.pdctrl.leftLeg.motorPd.pGain[i] = self.P[i]  # define the Left leg P-gain
            self.pdctrl.rightLeg.motorPd.pGain[i] = self.P[i]  # Right leg P-gain

            self.pdctrl.leftLeg.motorPd.dGain[i] = self.D[i]  # Left leg D-gain
            self.pdctrl.rightLeg.motorPd.dGain[i] = self.D[i]  # Right leg D-gain

            self.pdctrl.leftLeg.motorPd.torque[i] = 0  # Zero feedforward torque
            self.pdctrl.rightLeg.motorPd.torque[i] = 0  # Zero feedforward torque

            self.pdctrl.leftLeg.motorPd.pTarget[i] = target[i]  # Set target position for left leg
            self.pdctrl.rightLeg.motorPd.pTarget[i] = target[i + 5]  # Set target position for right leg

            self.pdctrl.leftLeg.motorPd.dTarget[i] = 0  # Zero velocity target for left leg
            self.pdctrl.rightLeg.motorPd.dTarget[i] = 0  # Zero velocity target for right leg

        # Step the simulation forward by one step with the PD control input
        self.cassie_state = self.cassim.step_pd(self.pdctrl)
        self.cassim.foot_pos(foot_pos)
        self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
        # Update the time step counter
        # self.time += 1



    def _get_reward(self, foot_pos, rw=(0.15, 0.15, 0.3, 0.2, 0.2, 0, 0), multiplier=500):
        qpos = np.copy(self.cassim.qpos())
        qvel = np.copy(self.cassim.qvel())

        left_foot_pos = foot_pos[:3]
        right_foot_pos = foot_pos[3:]

        # midfoot position
        foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

        # A. Task Rewards

        # 1. Pelvis Orientation 
        target_pose = np.array([1, 0, 0, 0])
        pose_error = 1 - np.inner(qpos[3:7], target_pose) ** 2

        r_pose = np.exp(-1e5 * pose_error ** 2)

        # 2. CoM Position Modulation
        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                  0.5 * (foot_pos[1] + foot_pos[4])])

        xy_com_pos = np.exp(-np.sum(qpos[:2] - xy_target_pos) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        height_thresh = 0.1  # m = 10 cm
        z_target_pos = self.target_height

        if qpos[2] < z_target_pos - height_thresh:
            z_com_pos = np.exp(-100 * (qpos[2] - (z_target_pos - height_thresh)) ** 2)
        elif qpos[2] > z_target_pos + 0.1:
            z_com_pos = np.exp(-100 * (qpos[2] - (z_target_pos + height_thresh)) ** 2)
        else:
            z_com_pos = 1.

        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # 3. CoM Velocity Modulation
        target_speed = np.array([self.speed, 0, 0]) # Only care the vel. along x axis
        r_com_vel = np.exp(-np.linalg.norm(target_speed - qvel[:3]) ** 2)

        # 4. Feet Width
        width_thresh = 0.02     # m = 2 cm
        target_width = 0.18     # m = 18 cm seems to be optimal
        feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]])

        if feet_width < target_width - width_thresh:
            r_foot_width = np.exp(-multiplier * (feet_width - (target_width - width_thresh)) ** 2)
        elif feet_width > target_width + width_thresh:
            r_foot_width = np.exp(-multiplier * (feet_width - (target_width + width_thresh)) ** 2)
        else:
            r_foot_width = 1.

        # 5. Foot/Pelvis Orientation (may need to be revisited when doing turning)
        _, _, pelvis_yaw = quaternion2euler(qpos[3:7])
        foot_yaw = np.array([qpos[8], qpos[22]])
        left_foot_orient = np.exp(-multiplier * (foot_yaw[0] - pelvis_yaw) ** 2)
        right_foot_orient = np.exp(-multiplier * (foot_yaw[1] - pelvis_yaw) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_width
                  + rw[4] * r_fp_orient)

        if self.debug:
            print('Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}]]'.format(r_pose,
                                                                                       r_com_pos,
                                                                                       r_com_vel,
                                                                                       r_foot_width,
                                                                                       r_fp_orient))

        
        return reward


    def _get_cost(self, foot_grf, cw=(0, 0.1, 0.5)):
        # 1. Ground Contact (At least 1 foot must be on the ground)
        # TODO: Only valid for walking gaits
        qpos = np.copy(self.cassim.qpos())
        c_contact = 1 if (foot_grf[0] + foot_grf[1]) == 0 else 0

        # 2. Power Consumption
        # Specs taken from RoboDrive datasheet for ILM 115x50

        # in Newton-meters
        max_motor_torques = np.array([4.66, 4.66, 12.7, 12.7, 0.99,
                                      4.66, 4.66, 12.7, 12.7, 0.99])

        # in Watts
        power_loss_at_max_torque = np.array([19.3, 19.3, 20.9, 20.9, 5.4,
                                             19.3, 19.3, 20.9, 20.9, 5.4])

        gear_ratios = np.array([25, 25, 16, 16, 50,
                                25, 25, 16, 16, 50])

        # calculate power loss constants
        power_loss_constants = power_loss_at_max_torque / np.square(max_motor_torques)

        # get output torques and velocities
        output_torques = np.array(self.cassie_state.motor.torque[:10])
        output_velocity = np.array(self.cassie_state.motor.velocity[:10])

        # calculate input torques
        input_torques = output_torques / gear_ratios

        # get power loss of each motor
        power_losses = power_loss_constants * np.square(input_torques)

        # calculate motor power for each motor
        motor_powers = np.amax(np.diag(output_torques).dot(output_velocity.reshape(10, 1)), initial=0, axis=1)

        # estimate power
        power_estimate = np.sum(motor_powers) + np.sum(power_losses)

        c_power = 1. / (1. + np.exp(-(power_estimate - self.power_threshold)))

        # 3. Falling
        c_fall = 1 if qpos[2] < self.fall_height else 0

        # Total Cost
        cost = cw[0] * c_contact + cw[1] * c_power + cw[2] * c_fall

        return cost
    
    # def _get_reward(self, action):

    #     qpos = np.copy(self.cassim.qpos())
    #     qvel = np.copy(self.cassim.qvel())

    #     # 标准化脚的力量和速度
    #     desired_max_foot_frc = 350
    #     desired_max_foot_vel = 3.0
    #     orient_targ = np.array([1, 0, 0, 0])

    #     com_vel = qvel[0]  #只关注X方向速度
    #     # 对FRC和vel设置一个上限
    #     normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    #     normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    #     normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    #     normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    #     com_orient_error = 0
    #     foot_orient_error = 0
    #     com_vel_error = 0
    #     com_orient_error += 1 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)
    #     foot_orient_error += 1 * (self.l_foot_orient_cost + self.r_foot_orient_cost)
    #     com_vel_error += np.linalg.norm(self.speed - com_vel)
    #     straight_diff = np.abs(qpos[1])  # straight difference penalty
    #     if straight_diff < 0.05:
    #         straight_diff = 0
    #     height_diff = np.abs(qpos[2] - 0.9)
    #     deadzone_size = 0.05 + 0.05 * self.speed
    #     if height_diff < deadzone_size:
    #         height_diff = 0
    #     pelvis_motion = straight_diff + height_diff
    #     left_frc_clock = self.left_clock[0](self.phase)
    #     right_frc_clock = self.right_clock[0](self.phase)
    #     left_vel_clock = self.left_clock[1](self.phase)
    #     right_vel_clock = self.right_clock[1](self.phase)


    #     left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    #     left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    #     right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    #     right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    #     foot_frc_score = left_frc_score + right_frc_score
    #     foot_vel_score = left_vel_score + right_vel_score

    #     reward = 0.250 * foot_frc_score + \
    #              0.350 * foot_vel_score + \
    #              0.200 * np.exp(-com_vel_error) + \
    #              0.100 * np.exp(-(com_orient_error + foot_orient_error)) + \
    #              0.100 * np.exp(-pelvis_motion)

    #     return reward

  

    def _get_obs(self):
        qpos = np.copy(self.cassim.qpos())  # 获取当前位置
        qvel = np.copy(self.cassim.qvel())  # 当前速度
        obs = np.concatenate([qpos[:], qvel[:]]).astype(np.float32)
        clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                 np.cos(2 * np.pi * self.phase / self.phaselen)]
        ext_state = np.concatenate((clock, [self.speed]))
        #obs = np.concatenate([obs, ext_state])

        return obs



