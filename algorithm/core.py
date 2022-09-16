import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    '''
    创建一个mlp序列神经网络模型
    sizes是一个列表，是每层的神经元个数,包括输入和输出层的神经元个数, 如 [10, 100, 3]
    两个列表直接相加会得到一个新的列表： [1,2,3] + [3,4] = [1,2,3,3,4]
    nn.Identity这个激活函数代表不加任何激活，直接输出，也就是说默认输出层没有激活函数
    '''

    # sizes = [14,64,64,8]
    # len(sizes)

    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class cnn_model(nn.Module):
    def __init__(self, num_inputs, num_out, activation=nn.ReLU):
        super(cnn_model, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # The output image size is calculated through
        # (I - K + 2*P) / S + 1
        # where: I : image size, the initial image size is 84x84, so I == 84 here.
        #        K : kernel size, here is 3
        #        P : padding size, here is 1
        #        S : stride, here is 2
        # So 84x84 image will become 6x6 through the ConvNet above. And 32 is the filters number.
        self.fc1 = nn.Linear(32 * 6 * 6, 512)
        self.fc_out = nn.Linear(512, num_out)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc_out(x)
        return out.squeeze()


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):
    '''
    继承Actor类，并修改一些基类的方法，产生分布类型是高斯分布，产生的分布是PDF，用于处理连续动作空间 Box
    可以实例化，输入inputs如下，其中hidden_sizes是隐藏层各层的神经元数量数组或者列表,如[100,100]
    再次注意，列表加和还是列表
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''
        高斯actor只有一个mu神经网络
        而高斯actor的log_std也就是log sigma^2不需要由神经网络输出，直接单独作为训练参数
        具体是先产生和动作维度一样的初值，再把这组数变成可训练参数
        '''
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)) 
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +[act_dim], activation)

    def _distribution(self, obs):
        '''
        高斯分布
        '''
        mu = self.mu_net(obs)         # mu的形状为 [N, act_dim]
        std = torch.exp(self.log_std) # log_std的形状为[act_dim]
        return Normal(mu, std)        # 虽然形状不匹配，但是可以产生[N, act_dim]的分布，log_std可以广播
    
    def _log_prob_from_distribution(self, pi, act):
        '''输出形为[N,]的logprob'''
        return pi.log_prob(act).sum(axis=-1)    # 最后一维的和, 因为输入act是[N, act_dim],需要返回形状为[N,]的求和结果
                                                # 连续动作空间，比如多关节环境，动作的维度为3，那么每次输出的动作维度也是3维，这是连续动作区间的特点
                                                # 比如说一个动作是[1.1,0.1,2.0]，计算出来的dist.logprob是(-1,-1,-2)，虽然是三维，但是毕竟这就只是一个动作
                                                # 因此函数_log_prob_from_distribution计算出来的是这一个动作的log概率=-4
                                                # 输入1个(组)动作，得到1个这个(组)动作对应的概率.


# Critic 只有一个基础MLP类，不需要基础类，直接一个可以实例化的类就行了。
class MLPCritic(nn.Module):
    '''Critic的输出维度只有[N,1]，输入是状态'''
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # 保证Critic输出的价值的维度也是 [N,]


# 把Actor和Critic合并成一类
class MLPActorCritic(nn.Module):
    '''
    创建一个默认参数的，可以调用的ActorCritic网络
    '''
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        
        obs_dim = observation_space.shape[0]

        # 根据环境是离散动作区间还是连续动作区间来建立对应的Actor策略pi
        if isinstance(action_space, Box): # 如果动作区间的类是Box，也就是连续动作空间
            act_dim = action_space.shape[0]
            self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        elif isinstance(action_space, Discrete): # 如果动作空间类是 Discrete，也就是离散动作空间
            act_dim = action_space.n
            self.pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
        
        # 建立Critic策略v
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        '''
        只接受1个obs，用于驱动环境运行
        这个函数是计算出的 old_logpa
        不用梯度，测试的输出该状态下
        使用策略得到的动作， 状态的价值， 动作对应的log p(a)
        '''
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            a = dist.sample()
            print(f'a={a}')

            logp_a = self.pi._log_prob_from_distribution(dist, a)
            print(f'logp_a={logp_a}')

            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
    
    def act(self, obs):
        '''
        这个函数，仅仅用在ppo_test里面，给一个状态，得到一个动作，用于测试。
        '''
        return self.step(obs)[0]


class CNNGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, activation, pretrain=None):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = cnn_model(obs_dim, act_dim, activation=activation)
        if pretrain != None:
            print('\n\nLoading pretrained from %s.\n\n' % pretrain)
        print(self.mu_net)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        print(f'log_prob(act):{pi.log_prob(act)}')
        return pi.log_prob(act).sum(axis=-1)

    # def forward(self, obs, act=None):
    #     mu = self.mu_net(obs)
    #     std = torch.exp(self.log_std)
    #     pi = Normal(mu, std)
    #     logp_a = None
    #     if act is not None:
    #         logp_a = pi.log_prob(act).sum(axis=-1)
    #     return pi, logp_a


class CNNCritic(nn.Module):
    def __init__(self, obs_dim, activation):
        super().__init__()
        # cnn_net([obs_dim] + list(hidden_sizes) + [1], activation)
        self.v_net = cnn_model(obs_dim, 1, activation=activation)
        print(self.v_net)

    def forward(self, obs):
        v = self.v_net(obs)
        return torch.squeeze(v, -1)  # Critical to ensure v has right shape.


class CNNActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(obs_dim, act_dim, activation)
        elif isinstance(action_space, Discrete):
            #self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
            raise NotImplementedError

        # build value functionp
        self.v = CNNCritic(obs_dim, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            print(f'a={a}')

            logp_a = self.pi._log_prob_from_distribution(pi, a)
            print(f'logp_a={logp_a}')
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
