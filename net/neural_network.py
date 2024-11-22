import torch
import torch.nn as nn
#from my_base_net import ResidualBlock
#from my_base_net import Net
# 设置设备为 GPU（如果有的话），否则为 CPU
import torch
import torch.nn as nn
from torch import sqrt

LOG_STD_HI = -1.5
LOG_STD_LO = -20

def normc_fn(m):  # 初始化
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ResidualBlock(nn.Module):  # 残差网络
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            #nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),  # 随机丢弃神经元
            nn.Linear(out_features, out_features),
            #nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


# The base class for an actor. Includes functions for normalizing state (optional)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.env_name = None

    def initialize_parameters(self):
        self.apply(normc_fn)


class line_model(Net):  # 定义线性模型
    def __init__(self, input_size, action_dim, layers=(128, 128), env_name=None, nonlinearity=torch.nn.functional.relu, normc_init=False, bounded=False, fixed_std=None):
        super(line_model, self).__init__()
        self.model = nn.Sequential(  # 调用 sequential 容器
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(64, layers[-1]),
            nn.Tanh()  # 假设动作范围在 [-1, 1] 之间
        )

        self.means = nn.Linear(layers[-1], action_dim)

        if fixed_std is None:  # probably don't want to use this for ppo, always use fixed std
            self.log_stds = nn.Linear(layers[-1], action_dim)
            self.learn_std = True
        else:
            self.fixed_std = fixed_std
            self.learn_std = False

        self.action = None
        self.action_dim = action_dim
        self.env_name = env_name
        self.nonlinearity = nonlinearity

        self.obs_std = 1.0
        self.obs_mean = 0.0
        self.normc_init = normc_init
        self.bounded = bounded


        self.init_parameters()
        # 初始化模型参数
        if normc_init:
            self.initialize_parameters()

    def init_parameters(self):
        if self.normc_init:
            self.apply(normc_fn)
            self.means.weight.data.mul_(0.01)

    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std
        x = state


        x = self.model(x)  # 通过模型


        mean = self.means(x)
        if self.bounded:
            mean = torch.tanh(mean)

        if self.learn_std:
            sd = (-2 + 0.5 * torch.tanh(self.log_stds(x))).exp()
        else:
            sd = self.fixed_std

        return mean, sd

    def forward(self, state, deterministic=True, anneal=1.0):
        mu, sd = self._get_dist_params(state)
        sd *= anneal

        if not deterministic:
            self.action = torch.distributions.Normal(mu, sd).sample()
        else:
            self.action = mu

        return self.action

    def get_action(self):
        return self.action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


class LSTM_model(Net):
    def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=nn.Tanh, fixed_std=None, normc_init=False,):#layers 每一层隐藏单元的数量
        super(LSTM_model, self).__init__()


        self.actor_layers = nn.ModuleList() #创建接受LSTM的容器
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]#加入第一层LSTM
        self.actor_layers += [nn.LSTMCell(layers[0], layers[1])]#加入第二层LSTM
        self.network_out = nn.Linear(layers[-1], action_dim)#全连接层

        self.action = None
        self.action_dim = action_dim# 类属性 action_dim
        self.init_hidden_state() #初始化 LSTM 层的隐藏状态
        self.env_name = env_name
        self.nonlinearity = nonlinearity()


        self.obs_std = 1.0
        self.obs_mean = 0.0
        if fixed_std is None:
            self.log_stds = nn.Linear(layers[-1], action_dim)
            self.learn_std = True
        else:
            self.fixed_std = fixed_std
            self.learn_std = False
        if normc_init:
            self.initialize_parameters()

        self.act = self.forward



    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std

        dimension = len(state.size())
        x = state
        if dimension == 3:  # if we get a batch of trajectories,x_t 表示时间步 t 的数据，它的形状可能是 [batch_size, features]
            self.init_hidden_state(batch_size=x.size(1))  # TBD
            y = []
            for t, x_t in enumerate(x):  # x_t是一个时间步的数据，其形状为 [batch_size, features]
                for idx, layer in enumerate(self.actor_layers):  # 每一层
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (
                    h, c))  # 前向传播，这个二维数据的形状是 [batch_size, input_size]，用于表示一批样本的特征在同一时间步的状态
                    x_t = self.hidden[idx]

                y.append(x_t)
            x = torch.stack([x_t for x_t in y])  # 存储
        elif dimension == 2:  # 处理二维输入 (B, state_dim)
             self.init_hidden_state(batch_size=x.size(0))  # 初始化隐藏状态，批次大小为 B
             for idx, layer in enumerate(self.actor_layers):  # 对每一层进行前向传播
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))  # 前向传播
                x = self.hidden[idx]

        else:
            if dimension == 1:  # 当输入是一维 tensor 时，x 代表单一时间步长的数据。
                #这意味着它仅包含当前时间步的状态（state），而没有序列和batch
                x = x.view(1, -1)#将输入的形状变为 [1, state_dim]
            for idx, layer in enumerate(self.actor_layers):#对每一层进行前向传播
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))#前向传播
                x = self.hidden[idx]

            if dimension == 1:
                x = x.view(-1)#因为输入是一维，所以让输出也是一维

        mu = self.network_out(x)
        if self.learn_std:
            sd = torch.clamp(self.log_stds(x), LOG_STD_LO, LOG_STD_HI).exp()
        else:
            sd = self.fixed_std

        return mu, sd

    def get_hidden_state(self):#函数返回 LSTM 网络的隐藏状态和细胞状态
        return self.hidden, self.cells


    def init_hidden_state(self, batch_size=1):#始化 LSTM 网络每一层的隐藏状态和细胞状态
      self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]#初始化隐藏状态
      self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]#初始化细胞状态


    def forward(self, state, deterministic=True, anneal=1.0):#前向传播函数，x 是输入状态，dims 存储输入 x 的维度
        mu, sd = self._get_dist_params(state)
        sd *= anneal

        if not deterministic:
            self.action = torch.distributions.Normal(mu, sd).sample()
        else:
            self.action = mu
        return self.action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)

    def get_action(self):
        return self.action
    
class LSTM_Critic(Net):
    def __init__(self, input_dim, layers=(128, 128), env_name='NOT SET', normc_init=True):
        super(LSTM_Critic, self).__init__()
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.LSTMCell(input_dim, layers[0])]
        self.critic_layers += [nn.LSTMCell(layers[0], layers[1])]
        self.network_out = nn.Linear(layers[-1], 1)
        self.env_name = env_name
        self.init_hidden_state()

        if normc_init:
            self.initialize_parameters()

    def get_hidden_state(self):
        return self.hidden, self.cells

    def init_hidden_state(self, batch_size=1):
        # 获取模型参数所在的设备
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]

    def forward(self, state):
        # 将输入移动到 GPU
        dimension = len(state.size())

        if dimension == 3:  # batch of trajectories
            self.init_hidden_state(batch_size=state.size(1))
            value = []
            for t, state_batch_t in enumerate(state):
                x_t = state_batch_t
                for idx, layer in enumerate(self.critic_layers):
                    h, c = self.hidden[idx], self.cells[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                x_t = self.network_out(x_t)
                value.append(x_t)
            x = torch.stack([a.float() for a in value])
        elif dimension == 2:  # batch of states (batch_size, state_dim)
            self.init_hidden_state(batch_size=state.size(0))
            x = state
            for idx, layer in enumerate(self.critic_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.network_out(x)
        else:
            if dimension == 1:
                x = state.view(1, -1)
            else:
                x = state
            self.init_hidden_state(batch_size=x.size(0))
            for idx, layer in enumerate(self.critic_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.network_out(x)
            if dimension == 1:
                x = x.view(-1)
        return x


        

