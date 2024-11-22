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


class CassieEnv(gym.Env):
    # Metadata for the rendering options, setting it to 'human' mode
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self, simrate=60,config="./cassiemujoco/cassie.xml"):
        self.config = config
        self.cassim = CassieSim(self.config)
        self.cassvis = None
        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.pdctrl = pd_in_t()
        self.time = 0
        self.last_pelvis_pos = self.cassim.qpos()[0:3]
        self.simrate = simrate
        # 定义动作空间，假设动作是 10 维连续的控制输入，范围是 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

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
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.phase_add = 1
        self.phaselen = 32
        self.strict_relaxer = 0.1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.have_incentive = True
        self.max_speed = 4.0
        self.min_speed = -0.3
        self.prev_action = None
        self.prev_torque = None
        self.neutral_foot_orient = np.array(
            [-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])

    @property
    def dt(self):
        return 1 / 2000 * self.simrate

    def step(self, action):
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Action {action} is invalid, expected shape: {self.action_space.shape}")
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        foot_pos = np.zeros(6)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
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
        reward = self._get_reward(action)

        height = self.cassim.qpos()[2]
        terminated = bool((height < 0.4) or (height > 3.0))
        

        obs = self._get_obs()

        return obs, reward, terminated,  {}

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
        total_duration = (0.9 - 0.25 / 3.0 * abs(self.speed)) / 2
        self.swing_duration = (0.30 + ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
        self.stance_duration = (0.70 - ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
        self.stance_mode = "zero"
        self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration,
                                                                             self.stance_duration,
                                                                               self.strict_relaxer, self.stance_mode,
                                                                               self.have_incentive,
                                                                               FREQ=2000 // self.simrate)
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        # Apply small random noise to the initial positions and velocities
        c = 0.01  # Small noise constant
        new_qpos = self.init_qpos
        new_qvel = self.init_qvel

        # Set the new positions and velocities
        self.cassim.set_qpos(new_qpos)
        self.cassim.set_qvel(new_qvel)
        # self.set_slope()  # reset degree every time
        obs= self._get_obs()

        return obs, {}

    
    
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
        target = action.squeeze().tolist() if isinstance(action, np.ndarray) else action
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



    def _get_reward(self, action):
    
        qpos = np.copy(self.cassim.qpos())
        qvel = np.copy(self.cassim.qvel())
        phase_diff = self.phase - np.floor(self.phase)
        ref_pos_prev, ref_vel_prev = self._get_ref_state(int(np.floor(self.phase)))
        if phase_diff != 0:
            ref_pos_next, ref_vel_next = self._get_ref_state(int(np.ceil(self.phase)))
            ref_pos_diff = ref_pos_next - ref_pos_prev
            ref_vel_diff = ref_vel_next - ref_vel_prev
            ref_pos = ref_pos_prev + phase_diff*ref_pos_diff
            ref_vel = ref_vel_prev + phase_diff*ref_vel_diff
        else:
            ref_pos = ref_pos_prev
            ref_vel = ref_vel_prev

        ref_pos, ref_vel = self._get_ref_state(self.phase)

        # TODO: should be variable; where do these come from?
        # TODO: see magnitude of state variables to gauge contribution to reward
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

        joint_error       = 0
        com_error         = 0
        orientation_error = 0
        spring_error      = 0

        # each joint pos
        for i, j in enumerate(self.pos_idx):
            target = ref_pos[j]
            actual = qpos[j]

            joint_error += 30 * weight[i] * (target - actual) ** 2

        # center of mass: x, y, z
        for j in [0, 1, 2]:
            target = ref_pos[j]
            actual = qpos[j]

            # NOTE: in Xie et al y target is 0

            com_error += (target - actual) ** 2

        # COM orientation: qx, qy, qz
        for j in [4, 5, 6]:
            target = ref_pos[j] # NOTE: in Xie et al orientation target is 0
            actual = qpos[j]

            orientation_error += (target - actual) ** 2

        # left and right shin springs
        for i in [15, 29]:
            target = ref_pos[i] # NOTE: in Xie et al spring target is 0
            actual = qpos[i]

            spring_error += 1000 * (target - actual) ** 2      

        reward = 0.5 * np.exp(-joint_error) +       \
                0.3 * np.exp(-com_error) +         \
                0.1 * np.exp(-orientation_error) + \
                0.1 * np.exp(-spring_error)

        # orientation error does not look informative
        # maybe because it's comparing euclidean distance on quaternions
        # print("reward: {8}\njoint:\t{0:.2f}, % = {1:.2f}\ncom:\t{2:.2f}, % = {3:.2f}\norient:\t{4:.2f}, % = {5:.2f}\nspring:\t{6:.2f}, % = {7:.2f}\n\n".format(
        #             0.5 * np.exp(-joint_error),       0.5 * np.exp(-joint_error) / reward * 100,
        #             0.3 * np.exp(-com_error),         0.3 * np.exp(-com_error) / reward * 100,
        #             0.1 * np.exp(-orientation_error), 0.1 * np.exp(-orientation_error) / reward * 100,
        #             0.1 * np.exp(-spring_error),      0.1 * np.exp(-spring_error) / reward * 100,
        #             reward
        #         )
        #     )

        return reward



        return reward

    def _get_obs(self):
        qpos = np.copy(self.cassim.qpos())  # 获取当前位置
        qvel = np.copy(self.cassim.qvel())  # 当前速度
        obs = np.concatenate([qpos[:], qvel[:]]).astype(np.float32)
        clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                 np.cos(2 * np.pi * self.phase / self.phaselen)]
        ext_state = np.concatenate((clock, [self.speed]))
        #obs = np.concatenate([obs, ext_state])

        return obs


    def _get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        desired_ind = phase * self.simrate if not self.aslip_traj else phase
        # phase_diff = desired_ind - math.floor(desired_ind)
        # if phase_diff != 0:       # desired ind is an int
        #     pos_prev = np.copy(self.trajectory.qpos[math.floor(desired_ind)])
        #     vel_prev = np.copy(self.trajectory.qvel[math.floor(desired_ind)])
        #     pos_next = np.copy(self.trajectory.qpos[math.ceil(desired_ind)])
        #     vel_next = np.copy(self.trajectory.qvel[math.ceil(desired_ind)])
        #     pos = pos_prev + phase_diff * (pos_next - pos_prev)
        #     vel = vel_prev + phase_diff * (vel_next - vel_prev)
        # else:
        # print("desired ind: ", desired_ind)
        pos = np.copy(self.trajectory.qpos[int(desired_ind)])
        vel = np.copy(self.trajectory.qvel[int(desired_ind)])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter * self.speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        if not self.aslip_traj:
            vel[0] *= self.speed

        return pos, vel
