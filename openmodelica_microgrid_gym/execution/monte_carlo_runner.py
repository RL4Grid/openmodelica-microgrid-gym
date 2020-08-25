import numpy as np
from typing import Dict, Any
from tqdm import tqdm
from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv


class MonteCarloRunner:
    """
    This class will execute an agent on the environment.
    It handles communication between agent and environment and handles the execution of multiple epochs
    Additionally to runner, the Monte-Carlo runner has an additional loop to perform n_MC experiments using one
    (controller) parameter set before update the (controller) parameters.
    Therefore, the agent.observe function is used.
    Inside the MC-loop the observe function is called with terminated = False to only update the return.
    The return is stored in an array at the end of the MC-loop.
    After finishing the MC-loop, the average of the return-array is used to update the (controller) parameters.
    Therefore, the agetn-observe function is called with terminated = True
    """

    def __init__(self, agent: Agent, env: ModelicaEnv):
        """

        :param agent: Agent that acts on the environment
        :param env: Environment tha Agent acts on
        """
        self.env = env
        self.agent = agent
        self.agent.env = env
        self.run_data = dict()  # type: Dict[str,Any]
        """
        Dictionary storing information about the experiment.

        - "best_env_plt": environment best plots
        - "best_episode_idx": index of best episode
        - "agent_plt": last agent plot
        """

    def run(self, n_episodes: int = 10, n_MC: int = 5, visualise: bool = False):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param n_MC: number of Monte-Carlo experiments using the same parameter set before updating the latter
        :param visualise: turns on visualization of the environment
        """
        self.agent.reset()
        self.env.history.cols = self.env.history.structured_cols(None) + self.agent.measurement_cols
        self.agent.obs_varnames = self.env.history.cols

        performance_MC = np.zeros(n_MC)

        if not visualise:
            self.env.viz_mode = None
        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):

            for m in tqdm(range(n_MC), desc='episodes', unit='epoch'):

                obs = self.env.reset()
                done, r = False, None
                for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                    self.agent.observe(r, False)
                    act = self.agent.act(obs)
                    self.env.measurement = self.agent.measurement
                    obs, r, done, info = self.env.step(act)
                    self.env.render()
                    if done:
                        self.agent.observe(r, False) # take the last reward into account, too, but without update_params
                        performance_MC[m] = self.agent.performance
                        break

            self.agent.performance = np.mean(performance_MC)
            _, env_fig = self.env.close()
            self.agent.observe(r, True)

            if visualise:
                agent_fig = self.agent.render()

            self.run_data['last_agent_plt'] = agent_fig

            if i == 0 or self.agent.has_improved:
                self.run_data['best_env_plt'] = env_fig
                self.run_data['best_episode_idx'] = i

            if i == 0 or self.agent.has_worsen:
                self.run_data['worst_env_plt'] = env_fig
                self.run_data['worst_episode_idx'] = i

            print(self.agent.unsafe)
