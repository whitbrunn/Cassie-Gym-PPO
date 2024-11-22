import torch
import torch.nn as nn
#from my_base_net import ResidualBlock
#from my_base_net import Net
# 设置设备为 GPU（如果有的话），否则为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
from torch import sqrt


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
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),  # 随机丢弃神经元
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
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


class line_model(Net):#定义线性模型
    def __init__(self, input_size, output_size):
        super(line_model, self).__init__()
        self.model = nn.Sequential(                 #调用sequential容器
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),        
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(64, output_size),
            nn.Tanh()  # 假设动作范围在[-1, 1]之间
        )
    
    def forward(self, x):   #定义前向传播函数
        # x 的形状为 (sequence_length, batch_size, state_dim)
        if x.dim() == 1:
            # 如果输入是 [state_dim]，增加 batch 和序列维度
            x = x.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, state_dim]
        elif x.dim() == 2:
            # 如果输入是 [batch_size, state_dim]，增加序列维度
            x = x.unsqueeze(0)  # 变为 [1, batch_size, state_dim]
        elif x.dim() == 3:
            # 输入已经是 [sequence_length, batch_size, state_dim]，不需要处理
            pass
        else:
            raise ValueError(f"Expected 1D, 2D or 3D input, but got {x.dim()}D input")

            # 现在 x 的形状为 [sequence_length, batch_size, state_dim]
        sequence_length, batch_size, state_dim = x.size()
        x = x.view(sequence_length * batch_size, state_dim)  # 展平为 (sequence_length * batch_size, state_dim)
        x = self.model(x)  # 通过模型
        x = x.view(sequence_length, batch_size, -1)  # 重塑回 (sequence_length, batch_size, output_size)
        return x

class LSTM_model(Net):
    def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=nn.Tanh, max_action=1):#layers 每一层隐藏单元的数量
        super(LSTM_model, self).__init__()

        self.actor_layers = nn.ModuleList() #创建接受LSTM的容器
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]#加入第一层LSTM
        self.actor_layers += [nn.LSTMCell(layers[0], layers[1])]#加入第二层LSTM
        self.network_out = nn.Linear(layers[-1], action_dim)#全连接层
        self.action = None
        self.action_dim = action_dim# 类属性 action_dim
        self.init_hidden_state() #初始化 LSTM 层的隐藏状态
        self.env_name = env_name
        self.nonlinearity = nonlinearity
        self.max_action = max_action
        self.tanh = nn.Tanh()
    def get_hidden_state(self):#函数返回 LSTM 网络的隐藏状态和细胞状态
        return self.hidden, self.cells


    def init_hidden_state(self, batch_size=1):#始化 LSTM 网络每一层的隐藏状态和细胞状态
      self.hidden = [torch.zeros(batch_size, l.hidden_size, device=device) for l in self.actor_layers]#初始化隐藏状态
      self.cells = [torch.zeros(batch_size, l.hidden_size, device=device) for l in self.actor_layers]#初始化细胞状态


    def forward(self, x, deterministic=True):#前向传播函数，x 是输入状态，dims 存储输入 x 的维度
        dimension = len(x.size())#

        if dimension == 3:  # if we get a batch of trajectories,x_t 表示时间步 t 的数据，它的形状可能是 [batch_size, features]
            self.init_hidden_state(batch_size=x.size(1)) # TBD
            y = []
            for t, x_t in enumerate(x):#x_t是一个时间步的数据，其形状为 [batch_size, features]
                for idx, layer in enumerate(self.actor_layers):#每一层
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))#前向传播，这个二维数据的形状是 [batch_size, input_size]，用于表示一批样本的特征在同一时间步的状态
                    x_t = self.hidden[idx]
                
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])#存储


        elif dimension == 2:  # 处理二维输入 (B, state_dim)
             self.init_hidden_state(batch_size=x.size(0))  # 初始化隐藏状态，批次大小为 B
             for idx, layer in enumerate(self.actor_layers):  # 对每一层进行前向传播
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))  # 前向传播
                x = self.hidden[idx]
             x = self.nonlinearity(self.network_out(x))  # 非线性激活函数
        else:
            if dimension == 1:  # 当输入是一维 tensor 时，x 代表单一时间步长的数据。
                #这意味着它仅包含当前时间步的状态（state），而没有序列和batch
                x = x.view(1, -1)#将输入的形状变为 [1, state_dim]
            for idx, layer in enumerate(self.actor_layers):#对每一层进行前向传播
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))#前向传播
                x = self.hidden[idx]
            x = self.nonlinearity(self.network_out(x))#非线性激活函数        

            if dimension == 1:
                x = x.view(-1)#因为输入是一维，所以让输出也是一维

        self.action = self.network_out(x) #将输出的动作值赋值给类属性 action
        self.action = self.tanh(self.action)
        return self.action

    def get_action(self):
        return self.action
    
class LSTM_Critic(Net):
  def __init__(self, input_dim, layers=(128, 128), env_name='NOT SET', normc_init=True):
    super(LSTM_Critic, self).__init__()

    self.critic_layers = nn.ModuleList() #创建接受LSTM的容器
    self.critic_layers += [nn.LSTMCell(input_dim, layers[0])]#加入第一层LSTM
    self.critic_layers += [nn.LSTMCell(layers[0], layers[1])]#加入第二层LSTM
    self.network_out = nn.Linear(layers[-1], 1)#全连接层

    self.init_hidden_state()
    self.env_name = env_name

    if normc_init:
      self.initialize_parameters()

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self, batch_size=1, device=torch.device('cpu')):
      self.hidden = [torch.zeros(batch_size, l.hidden_size, device=device) for l in self.critic_layers]#初始化隐藏状态
      self.cells = [torch.zeros(batch_size, l.hidden_size, device=device) for l in self.critic_layers]#初始化细胞状态
  
  def forward(self, state):
    device = state.device
    dimension = len(state.size())

    if dimension == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=state.size(1), device=device)
      value = []
      for t, state_batch_t in enumerate(state):
        x_t = state_batch_t.to(device)
        for idx, layer in enumerate(self.critic_layers):
          c, h = self.cells[idx].to(device), self.hidden[idx].to(device)
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.network_out(x_t)
        value.append(x_t)

      x = torch.stack([a.float() for a in value])

    elif dimension == 2:  # if we get a batch of states (batch_size, state_dim)
      self.init_hidden_state(batch_size=state.size(0), device=device)
      x = state.to(device)

      for idx, layer in enumerate(self.critic_layers):
        c, h = self.cells[idx].to(device), self.hidden[idx].to(device)
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]

      x = self.network_out(x)


    else:
      x = state.to(device)
      if dimension == 1:
        x = x.view(1, -1)

      for idx, layer in enumerate(self.critic_layers):
        c, h = self.cells[idx].to(device), self.hidden[idx].to(device)
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]
      x = self.network_out(x)

      if dimension == 1:
        x = x.view(-1)

    return x


        

