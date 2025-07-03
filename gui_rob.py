import os
import time

import click
import numpy as np

import pybullet as p

from drl_am.agent.amppo import AMPPOAgent
from drl_am.gym.robot_arm import RobotArmGym
from drl_am.trainer.amppo import AMPPOTrainer
from drl_am.utils.misc import setup_gpu, find_config_model, load_config
from train_rob import RobotArmParams


@click.command()
@click.argument("config", default=None, required=False)
@click.option('--gpu', is_flag=True)
@click.option('--gpu_id', default=None, help="GPU ID to use. 0 by default if --gpu flag is set.")
@click.option('-p', '--params', nargs=2, multiple=True)
@click.option('--rand', is_flag=True, help="Use random actions instead of the trained policy.")
@click.option('--feas', is_flag=True, help="Use feasible actions only.")
@click.option('--stochastic', is_flag=True, help="Use stochastic actions instead of deterministic.")
@click.option('--verbose', is_flag=True, help="Print network summary.")
def main(**kwargs):
    setup_gpu(kwargs)

    config = find_config_model(kwargs["config"])
    params: RobotArmParams = load_config(RobotArmParams, config, overrides=kwargs['params'])
    log_dir = config.rsplit("/", maxsplit=1)[0]

    physicsId = p.connect(p.GUI)

    gym = RobotArmGym(params.env, physicsClientId=physicsId)

    agent = AMPPOAgent(params.agent, obs_space=gym.observation_space, action_dim=gym.action_dim)
    trainer = AMPPOTrainer(params.trainer, gym=gym, agent=agent, logger=None)

    model_dir = log_dir + "/models"
    agent.load_weights(model_dir)
    if kwargs['verbose']:
        print("Network summary:")
        if hasattr(agent, "feasibility_policy"):
            print("Feasibility policy:")
            agent.feasibility_policy.summary()
        print("Objective policy:")
        agent.actor.summary()
        print("Critic summary:")
        agent.critic[0].summary()

    exploit = not kwargs["stochastic"]
    state = gym.reset()
    gym.render(state, update_obstacles=True)

    while True:
        obs = gym.get_obs(state)
        if kwargs["rand"]:
            action = np.random.uniform(-1.0, 1.0, size=(1, 1, gym.action_dim)).astype(np.float32)
        else:
            if kwargs["feas"]:
                action, latent = agent.get_feasible_action(obs)
            else:
                action, latent, log_prob = agent.get_action(obs, exploit=exploit)
            action = action[None, None, :]


        feas = gym.action_feasibility(state, action)
        state, reward, terminate, timeout, info = gym.step(state, action[:, 0, :])
        gym.render(state, update_obstacles=False)
        dist, angle = gym.difference_poses(state.ee_pose, state.target)

        feas = np.squeeze(feas.numpy())

        new_state = gym.reset_state(state, terminate | timeout)

        if terminate[0] or timeout[0]:
            if terminate[0]:
                print(f"Terminal state reached after {state.steps[0].numpy()}. Resetting environment.")
                if state.is_collision[0]:
                    print("Collision detected.")
                if not state.is_inside_joint_limits[0]:
                    print("Joint limits violated.")
                if not state.is_in_speed_limits[0]:
                    print("Speed limits violated.")
            else:
                print(f"Timeout reached after {state.steps[0].numpy()}. Resetting environment.")

            time.sleep(2)
            gym.render(new_state, update_obstacles=True)

        state = new_state


if __name__ == "__main__":
    main()
