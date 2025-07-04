import copy
import os
import pickle

import click
import numpy as np
import pygame

from drl_am.agent.amsac import AMSACAgent
from drl_am.gym.path_planning import PathPlanningGym
from drl_am.trainer.amsac import AMSACTrainer
from drl_am.utils.misc import setup_gpu, find_config_model, load_config
from train_pp import PathPlanningParams


def get_actions(agent, obs, exploit=False, num_actions=1):
    output = agent.get_action(obs, exploit) if num_actions == 1 else agent.get_actions(obs, num_actions=num_actions)
    if isinstance(output, np.ndarray):
        action = output
    else:
        action = output[0]
    return action if num_actions > 1 else [action]


@click.command()
@click.argument("model", default=None, required=True)
@click.option('--gpu', is_flag=True)
@click.option('--gpu_id', default=None, help="GPU ID to use. 0 by default.")
@click.option('-p', '--params', nargs=2, multiple=True, help="Overrides params in the config file. Provide 'path.to.param value'")
@click.option('-s', default=800, help="Size of the window.")
def main(**kwargs):
    setup_gpu(kwargs)

    config = find_config_model(kwargs["model"])
    params: PathPlanningParams = load_config(PathPlanningParams, config, overrides=kwargs['params'])
    log_dir = config.rsplit("/", maxsplit=1)[0]
    size = int(kwargs["s"])

    pygame.init()
    win = pygame.display.set_mode((size, size))
    env = PathPlanningGym(params.env, win=win, offset=(0, 0), win_shape=(size, size))

    pygame.display.set_caption("Spline Gym")

    agent = AMSACAgent(params.agent, obs_space=env.observation_space, action_dim=env.action_dim)

    model_dir = log_dir + "/models"
    agent.load_weights(model_dir)

    state, obs = env.reset()

    trainer = AMSACTrainer(params.trainer, env, agent, None)

    run = True
    step = False
    stepping = False
    autoreload = True
    num_actions = 1
    feasible_only = False
    compute_actions = True
    splines = []
    draw = True
    stochastic = True
    show_trajectory = False
    init_state = copy.deepcopy(state)
    while run:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    run = False
                if event.key == pygame.K_r:
                    state = copy.deepcopy(init_state)
                    obs = env.get_obs(state)
                    compute_actions = True
                if event.key == pygame.K_y:
                    stepping = True
                if event.key == pygame.K_s:
                    step = not stepping
                    stepping = False
                if event.key == pygame.K_f:
                    feasible_only = not feasible_only
                    compute_actions = True
                if event.key == pygame.K_t:
                    stochastic = not stochastic
                    print("Stochastic:", stochastic)
                    compute_actions = True
                if event.key == pygame.K_o:
                    state, obs = env.reset()
                    init_state = copy.deepcopy(state)
                    compute_actions = True
                    stepping = False
                if event.key == pygame.K_l:
                    agent.load_weights(model_dir)
                    print("Loaded weights from", model_dir)
                    compute_actions = True
                if pygame.K_1 <= event.key <= pygame.K_9:
                    num_actions = 2 ** (event.key - pygame.K_1)
                    compute_actions = True
                if event.key == pygame.K_m:
                    show_trajectory = not show_trajectory
                    draw = True
                if event.key == pygame.K_p:
                    # Save a screenshot as
                    filename = input("Enter filename: [screenshot].png \n")
                    if filename == "":
                        filename = "screenshot"
                    pygame.image.save(win, f"{filename}.png")
                if event.key == pygame.K_h:
                    # print help
                    print("Controls:")
                    print("  - ESC or q: quit")
                    print("  - r: reset environment to the same initial state")
                    print("  - o: reset environment to a new initial state")
                    print("  - y: start automatic stepping through the environment")
                    print("  - s: step through the environment")
                    print("  - f: toggle feasible actions only")
                    print("  - t: toggle stochastic/deterministic actions")
                    print("  - l: load agent weights")
                    print("  - 1-9: set number of actions to generate (1, 2, 4, ..., 512)")
                    print("  - m: toggle showing trajectory")
                    print("  - p: save a screenshot (with command line query for filename)")


        if stepping or step:
            obs, reward, terminal, truncated, info = env.step(state, splines[0], render=True,
                                                                     log_trajectory=True)
            if terminal or truncated:
                env.render(state, show_trajectory=True)
                # Wait for key
                waiting = True
                stepping = False
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                                run = False
                                waiting = False
                                break
                            elif event.key == pygame.K_r:
                                state = copy.deepcopy(init_state)
                                obs = env.get_obs(state)
                                stepping = False
                            elif event.key == pygame.K_a:
                                env.animate_trajectory(state)
                                env.render(state, show_trajectory=True)
                                continue
                            elif event.key == pygame.K_p:
                                # Save a screenshot as
                                filename = input("Enter filename: [screenshot].png \n")
                                if filename == "":
                                    filename = "screenshot"
                                pygame.image.save(win, f"paper_docs/{filename}.png")
                                continue
                            elif event.key == pygame.K_o:
                                state, obs = env.reset()
                            elif event.key == pygame.K_h:
                                # print help
                                print("Controls:")
                                print("  - ESC or q: quit")
                                print("  - r: reset environment to the same initial state")
                                print("  - o: reset environment to a new initial state")
                                print("  - a: animate trajectory")
                                print("  - p: save a screenshot (with command line query for filename)")
                            else:
                                continue
                            waiting = False
                            break
                init_state = copy.deepcopy(state)
            if autoreload:
                agent.load_weights(model_dir)
            compute_actions = True
            step = False

        if compute_actions:
            if feasible_only:
                actions, _ = agent.get_feasible_actions(obs, num_actions=num_actions)
            else:
                if stochastic:
                    actions = get_actions(agent, obs, num_actions=num_actions)
                else:
                    actions = get_actions(agent, obs, exploit=True, num_actions=1)

            splines = [env.action_to_spline(state, action, check_feasibility=True) for action in actions]
            compute_actions = False
            draw = True

        if draw:
            env.render(state, splines, show_trajectory=show_trajectory)


    pygame.quit()


if __name__ == '__main__':
    main()
