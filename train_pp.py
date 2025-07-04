import click
from pydantic import BaseModel

from drl_am.agent.amsac import AMSACAgent
from drl_am.gym.path_planning import PathPlanningGym
from drl_am.trainer.amsac import AMSACTrainer
from drl_am.utils.evaluator import Evaluator
from drl_am.utils.logger import Logger
from drl_am.utils.misc import save_params, setup_gpu, load_config, create_log_dir, find_config_model


class PathPlanningParams(BaseModel):
    env: PathPlanningGym.Params = PathPlanningGym.Params()
    trainer: AMSACTrainer.Params = AMSACTrainer.Params()
    agent: AMSACAgent.Params = AMSACAgent.Params()
    evaluator: Evaluator.Params = Evaluator.Params()
    logger: Logger.Params = Logger.Params()

@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--gpu_id', default=None, help="GPU ID to use. 0 by default.")
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True, help="Overrides params in the config file. Provide 'path.to.param value'")
@click.option('--id', default=None, help="Run ID for logging.")
@click.option('--feas', default=None, help="If set load feasibility policy instead of training it.")
@click.option('--verbose', is_flag=True, help="Print network summary.")
def train(**kwargs):
    if kwargs['generate']:
        save_params(PathPlanningParams(), kwargs['config'])
        return

    setup_gpu(kwargs)

    params: PathPlanningParams = load_config(PathPlanningParams, kwargs['config'], overrides=kwargs['params'])
    log_dir = create_log_dir(kwargs['config'], kwargs['id'])
    save_params(params, log_dir + "config.json")

    env = PathPlanningGym(params.env)

    agent = AMSACAgent(params.agent, obs_space=env.observation_space, action_dim=env.action_dim)
    if kwargs['verbose']:
        print("Network summary:")
        if hasattr(agent, "feasibility_policy"):
            print("Feasibility policy:")
            agent.feasibility_policy.summary()
        print("Objective policy:")
        agent.actor.summary()
        print("Critic summary:")
        agent.critic[0].summary()

    evaluator = Evaluator(params.evaluator, env, agent)
    logger = Logger(params.logger, log_dir=log_dir, agent=agent, evaluator=evaluator)
    trainer = AMSACTrainer(params.trainer, gym=env, agent=agent, logger=logger)
    evaluator.trainer = trainer

    logger.log_params(params)

    if kwargs['feas'] is not None:
        feas_path = find_config_model(kwargs["feas"])
        feas_path = feas_path.rsplit("/", maxsplit=1)[0] + "/models"
        agent.load_weights_feasibility(feas_path)
        print(f"Loaded feasibility policy from {feas_path}")
        print("Training objective agent.")
        trainer.train(skip_feas=True)
    else:
        trainer.train()

    agent.save_weights(log_dir + "/models")


if __name__ == '__main__':
    train()
