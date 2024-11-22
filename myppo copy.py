"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy
import gym
from sympy.codegen import Print

import gym_pkg
import time
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

from torch.nn.utils.rnn import pad_sequence

import time

import numpy as np
import os, sys

import ray


from rl.envs.normalize import get_normalization_params, PreNormalizer
from net.neural_network import line_model
from net.neural_network  import LSTM_model
from net.neural_network  import LSTM_Critic
import pickle


class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []

        self.ep_returns = []  # for logging  存储每个轨迹（episode）的总奖励
        self.ep_lens = []  # 存储每个轨迹的长度

        self.gamma, self.lam = gamma, lam  # 折扣因子

        self.ptr = 0  # 指向当前存储位置的指针，用于跟踪数据的添加
        self.traj_idx = [0]  # 存储每个轨迹的起始位置，初始为 [0]

    def __len__(self):
        return len(self.states)  # 返回 states 列表的长度，即缓冲区中存储的条目数

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states += [state.flatten()]  # 将状态展平
        self.actions += [action.flatten()]  # 展平
        print('---------', reward)
        if reward is None:
            reward = 0.0
        self.rewards += [reward]  # 奖励一般是标量，不需要展平
        self.values += [value.flatten()]

        self.ptr += 1  # 每次存储后指针 ptr 自增 1

    def finish_path(self, last_val=None):  # 用于标记一个轨迹的结束，并计算对应的回报
        self.traj_idx += [self.ptr]  # 这个列表存储每次轨迹开始和结束的索引
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]  # 这行代码提取了从上一个轨迹终止位置到当前终止
        # 位置的所有奖励，self.traj_idx[-2] 表示上一个轨迹的终止位置（也是当前轨迹的起始位置），self.traj_idx[-1] 表示当前轨迹的终止位置。

        returns = []  # returns：用来存储当前轨迹计算出来的回报。回报表示未来所有奖励的折现和

        R = 0 if last_val is None else last_val.squeeze()  # R 是一个变量，用于存储从当前时间步开始计算的回报
        # last_val 表示最后一个状态的价值估计。如果有传入 last_val，
        # 会用这个值初始化 R，以便将未来的折现奖励考虑在内

        for reward in reversed(rewards):

            R = self.gamma * R + reward
            returns.insert(0, R)  # TODO: self.returns.insert(self.path_idx, R) ?
            # also technically O(k^2), may be worth just reversing list
            # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]  # 记录轨迹奖励总和与轨迹长度
        self.ep_lens += [len(rewards)]

    def get(self):
        return (
            self.states,
            self.actions,
            self.returns,
            self.values
        )


class PPO:
    def __init__(self, args, save_path):
        self.policy = None
        self.critic = None
        self.old_policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.env_name = args['env_name']
        self.gamma = args['gamma']
        self.lam = args['lam']  # GAE的参数
        self.lr = args['lr']  # 学习率
        self.eps = args['eps']  # ε 参数
        self.entropy_coeff = args['entropy_coeff']  # 设置熵正则化系数
        self.clip = args['clip']  # PO 损失函数中裁剪的系数
        self.minibatch_size = max(2, args['minibatch_size'])  # 每次更新策略时，
        # 数据集被拆分为若干个小批量，每次用一个小批量的数据进行参数更新，以降低计算复杂度，并帮助训练更稳定
        self.epochs = args['epochs']  # 作用：设置每次采样后的训练轮数。
        # 在更新策略时，使用采样到的数据进行多次（epochs 次）参数更新，以充分利用采样到的数据，提高数据效率
        self.num_steps = args['num_steps']  # 决定在每次交互中，智能体和环境要交互的最少时间步数
        self.max_traj_len = args['max_traj_len']  # 轨迹最大长度
        self.use_gae = args['use_gae']  # 设置是否使用 GAE（广义优势估计）。
        self.n_proc = args['num_procs']  # PPO 中通常使用多个并行环境来加速数据采集。n_proc 决定了并行执行的环境数
        self.grad_clip = args['max_grad_norm']  # 设置梯度裁剪的最大范数
        self.recurrent      = args['recurrent']         #设置是否使用循环神经网络（RNN）
        self.total_steps = 0  # 初始化总的时间步数为 0。
        self.highest_reward = -1  # 初始化最高奖励为 -1
        self.limit_cores = 0  # 设置是否限制使用的 CPU 核心数
        self.save_path = save_path  # 在训练过程中，需要定期保存策略和价值网络的参数，save_path 用于指定保存的位置

    def save(self, policy, critic):

        try:
            os.makedirs(self.save_path)  # 保存路径
        except OSError:
            pass
        filetype = ".pt"  # pytorch model
        torch.save(policy, os.path.join(self.save_path, "actor" + filetype))  # 保存为pt文件
        torch.save(critic, os.path.join(self.save_path, "critic" + filetype))

    @ray.remote(num_gpus=1) # 并行化采样
    @torch.no_grad()  # 禁止梯度计算，加快速率
    def sample(self, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"sample device. = {device}")
        torch.set_num_threads(1)  # 限制 PyTorch 只使用一个线程
        # print("Enter sample.")
        policy = policy.to(device)
        critic = critic.to(device)
        env = gym.make("me5418-Cassie-v0")  # 初始化环境和存储器 memory

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:  # 进入一个循环，直到采样的步数达到 min_steps
            state = torch.Tensor(np.array(env.reset()[0])).to(device) # 每次采样新轨迹时，环境都会被重置为初始状态，并转换成张量

            done = False
            value = 0
            traj_len = 0

            if hasattr(policy, 'init_hidden_state'):  # 检查策略网络是否有 init_hidden_state 方法，从而
                # 每次让隐藏状态都重置到初始状态
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):  # 检查价值网络
                critic.init_hidden_state()

            # 在轨迹结束前，生成动作，计算状态的价值，进行环境交互，并将结果存储到 memory
            while not done and traj_len < max_traj_len:  # 进入一个循环，直到轨迹结束或者轨迹长度达到max_traj_len
                action = policy(state.to(device), deterministic=False, anneal=anneal)  # 使用策略网络生成当前状态下的动作，
                action_np = action.cpu().numpy().squeeze()
                # deterministic=False：表示使用随机策略，允许探索行为。如果设置为 True，策略将输出确定性动作，这通常用于评估阶段
                # anneal=anneal：可以是一个控制探索与利用权衡的参数，用于调整动作的随机性，通常用于逐步减少探索
                value = critic(state.to(device))
                # 在环境中执行动作，并获取执行后的结果，生成的动作转换为 NumPy 数组并传递给环境，以便在环境中执行
                next_state, reward, terminated, _ = env.step(action_np)
                done = terminated 
                # env.step() 执行一部动作，并返回下一个状态，奖励 是否结束 附加信息
                memory.store(state.cpu().numpy(), action.cpu().numpy(), reward, value.cpu().numpy())# 存储

                state = torch.Tensor(next_state).to(device)  # 环境返回的下一个状态转换为 PyTorch 张量

                traj_len += 1
                num_steps += 1

            value = critic(state).to(device)
            memory.finish_path(last_val=(not done) * value.cpu().numpy())
        return memory

    def sample_parallel(self, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0,
                        term_thresh=0):
        # print("1 Enter sample_parallel.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        worker = self.sample  # self.sample指向当前类实例（self）中 sample 方法的引用
        args = (self, policy, critic, min_steps // self.n_proc, max_traj_len, deterministic, anneal, term_thresh)
        # print("2 Enter sample_parallel.")
        # Create pool of workers, each getting data for min_steps
        workers = [worker.remote(*args) for _ in range(self.n_proc)]  # for _ in range(self.n_proc)用来创建一个包含多个并行工作者的列表 workers
        # args：解包 args 元组，将其中的每个参数传递给 worker.remote()，相当于直接把参数逐一传入 sample 方法中
        result = []
        total_steps = 0

        while total_steps < min_steps:
            # get result from a worker
            '''
            eady_ids, _ = ray.wait(workers, num_returns=1)：
            ray.wait() 是 Ray 提供的一个函数，用于等待并行工作者完成任务。
            workers：是并行工作的工作者列表，每个工作者都在运行 sample 方法进行轨迹采样。
            num_returns=1：表示至少等待一个工作者完成任务并返回结果。ray.wait() 会返回两个列表：
            ready_ids：已经完成任务的工作者的 ID 列表。
            _：表示还在运行的工作者的 ID 列表（这个部分没有被使用）。
            通过 ray.wait()，可以在多个并行任务中确定哪些已经完成，从而进一步处理它们的结果。
            '''
            print("Enter loop.")
            ready_ids, _ = ray.wait(workers, num_returns=1)
            # update result
            result.append(ray.get(ready_ids[0]))  # ray.get(ready_ids[0]) 获取已经完成任务的工作者的结果，并将其添加到 result

            # remove ready_ids from workers (O(n)) but n isn't that big
            workers.remove(ready_ids[0])  # 从工作者列表 workers 中移除已经完成任务的工作者 (ready_ids[0])

            # update total steps
            total_steps += len(result[-1])  # 更新总的采样步数

            # start a new worker
            workers.append(worker.remote(*args))  # 启动一个新的工作者继续采样，将其添加到 workers 列表中，以保证有足够的工作者在并行运行

        # O(n)
        def merge(buffers):  # 于合并多个 PPOBuffer 实例
            merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:  # buffers 是多个 PPOBuffer 的列表
                offset = len(merged)

                merged.states += buf.states
                merged.actions += buf.actions
                merged.rewards += buf.rewards
                merged.values += buf.values
                merged.returns += buf.returns

                merged.ep_returns += buf.ep_returns
                merged.ep_lens += buf.ep_lens

                merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
                merged.ptr += buf.ptr  # 将每个 PPOBuffer 中的 states 和 actions 添加到合并后的
                # merged 中，确保所有轨迹数据被合并为一个缓冲区

            return merged

        total_buf = merge(result)

        return total_buf

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        return_batch = return_batch.to(device)
        advantage_batch = advantage_batch.to(device)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

            # 将 mask 移动到指定设备
        mask = mask.to(device)

        policy = self.policy.to(device)
        critic = self.critic.to(device)
        old_policy = self.old_policy.to(device)

        values = critic(obs_batch)  # critic 计算value
        pdf = policy.distribution(obs_batch)  # actor计算policy

        # TODO, move this outside loop?
        with torch.no_grad():  # 旧策略只用来比较和计算更新，因此不用计算梯度
            old_pdf = old_policy.distribution(obs_batch)  # 旧策略的状态分布
            old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 计算批量动作 action_batch 的对数概率

        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 计算当前策略网络的动作概率分布下，
        # 批量动作 action_batch 的对数概率,假设我们有一个状态 s，在这个状态下旧策略（old_policy）生成了一个动作 a。现在我们已经更新了策略，
        # 得到了新策略（policy），我们希望确保新策略不会对该状态 s 生成与旧策略完全不同的动作。为了衡量这种变化，我们通过新旧策略分别计算在状
        # 态 s 下执行动作 a 的概率。如果新策略的概率相较于旧策略变化太大，PPO 会限制这种更新。

        ratio = (log_probs - old_log_probs).exp()  # 用前面的两个结果来计算比率，ratio 的形状是 (20, 2, 1)。advantage_batch 的形状是 (20, 2, N)，N 代表列数（具体值可能较大）。mask 的形状是 (20, 2)


        mask = mask.unsqueeze(-1)
        cpi_loss = ratio * advantage_batch * mask  # 计算 PPO 的策略损失
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()


        values = values.squeeze(-1)  # 将 (20, 2, 1) 变为 (20, 2)，return_batch 的形状为 (20, 2)。values 的形状为 (20, 2, 1)。mask 的形状为 (20, 2, 1)
        mask = mask.squeeze(-1)
        critic_loss = 0.5 * ((return_batch - values) * mask).pow(2).mean()  # 计算价值网络的损失 critic_loss


        #print("1111111111111", self.entropy_coeff)
        #print("2222222222222", pdf.entropy())
        #print("3333333333", mask)
        mask = mask.unsqueeze(-1)

        # 将 mask 扩展为与 pdf.entropy() 相同的形状 (20, 2, 10)
        mask = mask.expand_as(pdf.entropy())
        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()  # 计算熵惩罚项 entropy_penalty，以增加策略的探索性

        # Mirror Symmetry Loss


        self.actor_optimizer.zero_grad()  # 将策略网络优化器的梯度设置为零，以清除之前的梯度累积，为下一次反向传播做好准备。
        (actor_loss  + entropy_penalty).backward()  # 将策略损失 (actor_loss)、镜像损失 (mirror_loss)
        # 和熵惩罚 (entropy_penalty) 相加后，执行反向传播以计算梯度

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)  # 对actor的梯度进行裁剪，
        # 以防止梯度过大导致不稳定的更新
        self.actor_optimizer.step()  # 应用梯度更新策略网络的参数

        self.critic_optimizer.zero_grad()  # 对critic的优化器梯度设置为零
        critic_loss.backward()  # 反向传播

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)  # 对价值网络的梯度进行裁剪
        self.critic_optimizer.step()  # 梯度更新

        with torch.no_grad():  # 在禁用梯度计算的上下文中计算当前策略 (pdf)
            # 和旧策略 (old_pdf) 之间的 Kullback-Leibler (KL) 散度，用于衡量策略更新的程度
            kl = kl_divergence(pdf, old_pdf)


        return actor_loss.item(), pdf.entropy().mean().item(), critic_loss.item(), ratio.mean().item(), kl.mean().item()

    def train(self,
              policy,
              critic,
              n_itr,  # n_itr：训练的总迭代次数
              logger=None, anneal_rate=1.0):  # logger：日志记录器（可选）anneal_rate：退火系数，用于控制探索的强度
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        critic.to(device)
        self.old_policy = deepcopy(policy).to(device)
        self.policy = policy
        self.critic = critic
        # 使用 Adam 优化器创建策略网络和价值网络的优化器，设置学习率 lr 和 eps 参数
        self.actor_optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()  # 记录训练时间

        curr_anneal = 1.0  # 退火系数
        curr_thresh = 0  # curr_thresh：当前的终止阈值，初始为 0
        start_itr = 0  # 迭代计数
        ep_counter = 0  # 用于记录完成的轨迹数量，初始为 0
        do_term = False  # 布尔值，用于标记是否需要提前终止，初始为 False
        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()  # sample_start = time.time()
            if self.highest_reward > (2 / 3) * self.max_traj_len and curr_anneal > 0.5:  # 如果最高奖励 (highest_reward) 超过了最大轨迹长度的三分之二，并且当前的退火系数
                # (curr_anneal) 大于 0.5，则将 curr_anneal 乘以退火率 (anneal_rate) 以逐步减小探索的强度
                curr_anneal *= anneal_rate
            # print("1 time elapsed: {:.2f} s".format(time.time() - start_time))
            if do_term and curr_thresh < 0.35:
                curr_thresh = .1 * 1.0006 ** (itr - start_itr)
            # print("2 time elapsed: {:.2f} s".format(time.time() - start_time))
            # 调用 sample_parallel() 方法并行采样数据，得到包含轨迹数据的批量 (batch)
            batch = self.sample_parallel(self.policy, self.critic, min_steps=self.num_steps * 4, max_traj_len=self.max_traj_len,
                                         anneal=curr_anneal, term_thresh=curr_thresh)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            samp_time = time.time() - sample_start
            print("sample time elapsed: {:.2f} s".format(samp_time))
            # 从 batch 中获取轨迹数据（状态、动作、回报和价值），并将它们转换为 PyTorch 张量
            observations, actions, returns, values = map(torch.Tensor, batch.get())
            # 计算优势函数 (advantages)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            # 设置小批量的大小为minibatch_size，如果没有指定，则默认使用整个批次
            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())
            self.total_steps += advantages.numel()  # 更新 self.total_steps，用于跟踪训练中的总时间步数

            self.old_policy.load_state_dict(policy.state_dict())  # 将当前策略网络 (policy) 的参数拷贝到
            # 旧策略网络 (old_policy) 中，保证旧策略与当前策略在更新之前是一致的

            optimizer_start = time.time()  # 记录优化开始的时间，以便计算优化阶段的耗时


            for epoch in range(self.epochs):
                # 初始化空列表 losses、entropies 和 kls，分别用于存储每次更新的损失、熵和 KL 散度
                kl = 0.0
                losses = []
                entropies = []
                entropy = 0.0
                kls = []
                if self.recurrent:  # 是否适用lstm
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx) - 1))  # 首先使用 SubsetRandomSampler 从轨迹
                    # 的索引中随机抽取数据 (random_indices)，SubsetRandomSampler是一个pytorch的采样器
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)  # 使用 BatchSampler 创建一个小批
                    # 量采样器 (sampler)，将大批次数据集拆分成若干小批次，根据随机抽取的轨迹进行采样，drop_last=False 表示保留所有小批次
                    # ，即使最后一个小批次不完整
                else:
                    random_indices = SubsetRandomSampler(range(advantages.numel()))  # 如果不使用循环神经网络，使用
                    # SubsetRandomSampler 从所有优势函数的样本中随机抽取数据 (random_indices)
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                for indices in sampler:  # 迭代sampler 获取每个小批量的索引
                    if max(indices) >= len(observations):
                        indices = [i for i in indices if i < len(observations)]
                    # 判断是否使用循环神经网络（如LSTM）。如果是，则按轨迹为单位来处理输入数据。循环神经网络需要知道数
                    # 据在时间维度上的顺序，因此按轨迹处理尤其重要。
                    if self.recurrent:
                        obs_batch = [observations[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
                        # 使用上面的范围从observations中提取第i条轨迹的数据。
                        action_batch = [actions[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in
                                        indices]  # action_batch
                        # 是一个包含多个轨迹的动作序列的列表，通过对列表actions切片获得
                        '''
                        假设 return_batch 包含两条轨迹的回报数据：
                        轨迹 1 的回报数据 r_1 是 [1.0, 1.2, 1.5]（长度为 3）。
                        轨迹 2 的回报数据 r_2 是 [0.8, 1.0, 0.9, 1.1]（长度为 4）。
                        return_batch = [
                        torch.tensor([1.0, 1.2, 1.5]),       # 轨迹 1 的回报数据
                        torch.tensor([0.8, 1.0, 0.9, 1.1])    # 轨迹 2 的回报数据
                        ]
                        通过 mask = [torch.ones_like(r) for r in return_batch]，得到的掩码 mask 为：
                        对于轨迹 1，掩码是 [1, 1, 1]。
                        对于轨迹 2，掩码是 [1, 1, 1, 1]
                        注意,returns是拼接成一个列表，return_batch是列表嵌套
    '''
                        return_batch = [returns[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
                        advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
                        mask = [torch.ones_like(r) for r in return_batch]

                        # pad_sequence() 会使用填充值 0 对较短的序列进行填充，直到它们与最长的序列长度一致
                        obs_batch = pad_sequence(obs_batch, batch_first=False)
                        action_batch = pad_sequence(action_batch, batch_first=False)
                        return_batch = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask = pad_sequence(mask, batch_first=False)  # 较短的轨迹会被 pad_sequence() 填充，
                        # 填充部分的掩码会自动设置为 0，表示这些时间步是无效的
                        '''
                        比如：
                        轨迹 1 的观测值：[s1_1, s1_2, s1_3]
                        轨迹 2 的观测值：[s2_1, s2_2, s2_3, s2_4]
                        轨迹 3 的观测值：[s3_1, s3_2]
                        则
                                            [
                        [s1_1, s2_1, s3_1],
                        [s1_2, s2_2, s3_2],
                        [s1_3, s2_3, 0],     # 轨迹 3 被填充到长度 4
                        [0,    s2_4, 0]      # 轨迹 1 和轨迹 3 被填充
                        ]

                        [
                         [1, 1, 1],  # 第 1 时间步有效
                         [1, 1, 1],  # 第 2 时间步有效
                         [1, 1, 0],  # 第 3 时间步：轨迹 3 无效
                         [0, 1, 0]   # 第 4 时间步：只有轨迹 2 有效
                        ]                      
                        '''
                    else:
                        obs_batch = observations[indices]
                        action_batch = actions[indices]
                        return_batch = returns[indices]
                        advantage_batch = advantages[indices]
                        mask = 1

                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask)
                    actor_loss, entropy, critic_loss, ratio, kl = scalars  # 更新参数

                    entropies.append(entropy)
                    kls.append(kl)
                    losses.append([actor_loss, entropy, critic_loss, ratio, kl])


                # TODO: add verbosity arguments to suppress this
                losses_mean = np.mean(losses, axis=0)
                losses_mean = np.atleast_1d(losses_mean)  # 确保结果至少是一维的
                print(' '.join(["%g" % x for x in losses_mean]))

                # Early stopping
                if len(kls) > 0 and np.mean(kls) > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

            opt_time = time.time() - optimizer_start  # 计算时间
            print("optimizer time elapsed: {:.2f} s".format(opt_time))

            if np.mean(batch.ep_lens) >= self.max_traj_len * 0.75:  # 如果批次中所有轨迹的平均长度 (batch.ep_lens) 大于等于最大
                # 轨迹长度的 75%，则轨迹计数器 ep_counter（用于记录完成的轨迹数量）初始为 0 增加 1
                ep_counter += 1
            if do_term == False and ep_counter > 50:  # 如果 do_term （用于标记是否需要提前终止）为 False 并且轨迹计数器 ep_counter 大于 50，则将 do_term
                # 设为 True，并更新开始迭代计数 (start_itr) 为当前迭代次数。这是一个动态调整终止条件的策略，用于控制训练过程中的轨迹长度
                do_term = True
                start_itr = itr

            if logger is not None:
                evaluate_start = time.time()
                test = self.sample_parallel(self.policy, self.critic, self.num_steps // 2, self.max_traj_len,
                                            deterministic=True)
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                avg_batch_reward = np.mean(batch.ep_returns)
                avg_ep_len = np.mean(batch.ep_lens)


                lossesmean = np.nanmean(losses, axis=0)
                lossesmean = np.atleast_1d(lossesmean)
                mean_losses = list(lossesmean)

                while len(mean_losses) < 5:
                    mean_losses.append(0.0)

                    # print("avg eval reward: {:.2f}".format(avg_eval_reward))

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                entropy = np.mean(entropies)  # 计算熵 (entropy) 和 KL 散度 (kl) 的均值
                kl = np.mean(kls)

                logger.add_scalar("Test/Return", avg_eval_reward, itr)
                logger.add_scalar("Train/Return", avg_batch_reward, itr)
                logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
                logger.add_scalar("Train/Mean KL Div", kl, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)

                logger.add_scalar("Misc/Critic Loss", mean_losses[2], itr)
                logger.add_scalar("Misc/Actor Loss", mean_losses[0], itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

                logger.add_scalar("Misc/Sample Times", samp_time, itr)
                logger.add_scalar("Misc/Optimize Times", opt_time, itr)
                logger.add_scalar("Misc/Evaluation Times", eval_time, itr)
                logger.add_scalar("Misc/Termination Threshold", curr_thresh, itr)

            # TODO: add option for how often to save model
            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                self.save(policy, critic)  # 果当前的评估奖励 (avg_eval_reward) 大于之前记录的最高奖励avg_eval_reward
                # 则更新最高奖励并保存模型的参数


def run_experiment(args):
    from util.log import create_logger

    # wrapper function for creating parallelized envs
    #env = gym.make("me5418-Cassie-v0", simrate=args.simrate)

    state_dim = 67
    action_dim = 10
    # Set up Parallelism
    os.environ['OMP_NUM_THREADS'] = '1'  # 设置并行性：将 OMP_NUM_THREADS 环境变量设置为 '1'，
    # 限制 PyTorch 在并行计算中只使用一个线程，以避免与 Ray 的多进程并行出现冲突
    if not ray.is_initialized():
        if args.redis_address is not None:
            ray.init(num_cpus=args.num_procs, num_gpus=args.num_gpus, redis_address=args.redis_address)
        else:
            ray.init(num_cpus=args.num_procs, num_gpus=args.num_gpus)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.previous is not None:
        policy = torch.load(os.path.join(args.previous, "actor.pt"), map_location=device)
        critic = torch.load(os.path.join(args.previous, "critic.pt"), map_location=device)
        # TODO: add ability to load previous hyperparameters, if this is something that we event want

        # with open(args.previous + "experiment.pkl", 'rb') as file:
        #     args = pickle.loads(file.read())
        print("loaded model from {}".format(args.previous))
    else:
        if args.recurrent:
            # 创建policy和critic
            print('!!!!!!!!!!!!!!!!!!use lstm!!!!!!!!!!!!!!!!!!!!!!!!!!')
            policy = LSTM_model(state_dim, action_dim)
            critic = LSTM_Critic(state_dim)
        else:
            print('!!!!!!!!!!!!!!!!!!use line!!!!!!!!!!!!!!!!!!!!!!!!!!')
            policy = line_model(state_dim, action_dim)
            critic = LSTM_Critic(state_dim)

#       with torch.no_grad():
#policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env, procs=args.num_procs))
#        critic.obs_mean = policy.obs_mean
#        critic.obs_std = policy.obs_std

    policy.train()
    critic.train()

    print("state_dim: {}, action_dim: {}".format(state_dim, action_dim))

    # create a tensorboard logging object
    logger = create_logger(args)

    algo = PPO(args=vars(args), save_path=logger.dir)

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print(" ├ recurrent:      {}".format(args.recurrent))
    print(" ├ run name:       {}".format(args.run_name))
    print(" ├ max traj len:   {}".format(args.max_traj_len))
    print(" ├ seed:           {}".format(args.seed))
    print(" ├ num procs:      {}".format(args.num_procs))
    print(" ├ lr:             {}".format(args.lr))
    print(" ├ eps:            {}".format(args.eps))
    print(" ├ lam:            {}".format(args.lam))
    print(" ├ gamma:          {}".format(args.gamma))
    print(" ├ learn stddev:  {}".format(args.learn_stddev))
    print(" ├ std_dev:        {}".format(args.std_dev))
    print(" ├ entropy coeff:  {}".format(args.entropy_coeff))
    print(" ├ clip:           {}".format(args.clip))
    print(" ├ minibatch size: {}".format(args.minibatch_size))
    print(" ├ epochs:         {}".format(args.epochs))
    print(" ├ num steps:      {}".format(args.num_steps))
    print(" ├ use gae:        {}".format(args.use_gae))
    print(" ├ max grad norm:  {}".format(args.max_grad_norm))
    print(" └ max traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(policy, critic, args.n_itr, logger=logger, anneal_rate=args.anneal)

