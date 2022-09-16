import sys
sys.path.append('/home/zhizhuo/Pybullet_zzz/Quadruped_RL')
from envs_zzz.env import LaikagoEnv
from algorithm.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import algorithm.core as core

env=LaikagoEnv()

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--is-render',action="store_true")
    parser.add_argument('--is-good-view',action="store_true")

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='zzz_RL')
    parser.add_argument('--log_dir', type=str, default="./logs")
    args = parser.parse_args()

    env=LaikagoEnv(is_render=args.is_render,is_good_view=args.is_good_view)

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir=args.log_dir)

    print(dict(hidden_sizes=[args.hid] * args.l))

    ppo(env,
        actor_critic=core.MLPActorCritic,
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=100*args.cpu,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l))

# o = env.reset()  # 只有在reset的时候，o才是一个字典！
# for i in range(100):
#     o, r, d, _ = env.step(env.action_space.sample())  # 在运行过程中，o不是字典
