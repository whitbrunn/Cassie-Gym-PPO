import torch
import sys, pickle, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simrate", default=20, type=int, help="simrate of environment")################simrate##############3
    parser.add_argument("--run_name", default=None)  # run name
    parser.add_argument('--num_gpus', type=int, default=1, help='GPU')#GPU数量
    parser.add_argument("--learn_gains", default=False, action='store_true', dest='learn_gains')####是否学习增益#######
    parser.add_argument("--previous", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="./trained_models/ppo/")  # Where to log diagnostics to
    parser.add_argument("--seed", default=42, type=int)  # 设置Gym随机种子
    parser.add_argument("--history", default=0, type=int)  # number of previous states
    parser.add_argument("--redis_address", type=str, default=None)  # redis
    parser.add_argument("--env_name", default="me5418-Cassie-v0")
    # PPO algo args
    parser.add_argument("--input_norm_steps", type=int, default=50)###########输入归一化的步数################
    parser.add_argument("--n_itr", type=int, default=1000, help="Number of iterations of the learning algorithm")#############迭代轮数###############
    parser.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate")  ################# 学习率######################
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam)")### Adam 优化器的 epsilon 值########
    parser.add_argument("--lam", type=float, default=0.95, help="GAE")#####广义优势估计####
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP")
    parser.add_argument("--learn_stddev", default=False, action='store_true', help="learn std_dev or keep it fixed")
    parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")##退火###
    parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.1,help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", type=int, default=32, help="Batch size for PPO updates")###############PPO 更新时的minibatch大小############
    parser.add_argument("--epochs", type=int, default=10, help="Number of optimization epochs per PPO update")  ############epoch################
    parser.add_argument("--num_steps", type=int, default=1000,help="Number of sampled ")##每次梯度估计采样步数###
    parser.add_argument("--use_gae", type=bool, default=True,help="GAE")
    parser.add_argument("--num_procs", type=int, default=10, help="Number of threads to train on")###################并行化采样的步数#############
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")#梯度裁剪的最大值#
    parser.add_argument("--max_traj_len", type=int, default=20, help="Max episode horizon")#最大轨迹长度#
    parser.add_argument("--recurrent", default=True)#####是否使用LSTM##############
    parser.add_argument("--bounded", type=bool, default=False)

    args = parser.parse_args()

    from myppo import run_experiment
    run_experiment(args)


