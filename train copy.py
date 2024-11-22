import torch
import sys, pickle, argparse
from util.logo import print_logo



if __name__ == "__main__":

    # print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")
    parser = argparse.ArgumentParser()

    """
        General arguments for configuring the environment
    """
    # command input, state input, env attributes
    parser.add_argument("--simrate", default=20, type=int, help="simrate of environment")
    parser.add_argument("--run_name", default=None)  # run name
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument("--env_name", default="Cassie-v0")  # environment name
    parser.add_argument("--learn_gains", default=False, action='store_true', dest='learn_gains')
    #parser.add_argument("--previous", type=str, default="/home/du/Cassie/1/trained_models/ppo/Cassie-v0/09b48d-seed0/")
    parser.add_argument("--previous", type=str, default=None)

    from myppo import run_experiment
        # general args
    parser.add_argument("--logdir", type=str, default="./trained_models/ppo/")  # Where to log diagnostics to
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--history", default=0, type=int)  # number of previous states to use as input
    parser.add_argument("--redis_address", type=str, default=None)  # address of redis server (for cluster setups)
    parser.add_argument("--viz_port", default=8097)  # (deprecated) visdom server port
    # PPO algo args
    parser.add_argument("--input_norm_steps", type=int, default=50)#输入归一化的布数
    parser.add_argument("--n_itr", type=int, default=100, help="Number of iterations of the learning algorithm")#############
    parser.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate")  ################# 学习率
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")
    parser.add_argument("--learn_stddev", default=False, action='store_true', help="learn std_dev or keep it fixed")
    parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.1,help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", type=int, default=32, help="Batch size for PPO updates")##############################3
    parser.add_argument("--epochs", type=int, default=10, help="Number of optimization epochs per PPO update")  ############
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of sampled timesteps per gradient estimate")
    parser.add_argument("--use_gae", type=bool, default=True,
                        help="Whether or not to calculate returns using Generalized Advantage Estimation")
    parser.add_argument("--num_procs", type=int, default=10, help="Number of threads to train on")###################
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
    parser.add_argument("--max_traj_len", type=int, default=20, help="Max episode horizon")
    parser.add_argument("--recurrent", default=True)
    parser.add_argument("--bounded", type=bool, default=False)
    args = parser.parse_args()

    #args = parse_previous(args)

    run_experiment(args)


