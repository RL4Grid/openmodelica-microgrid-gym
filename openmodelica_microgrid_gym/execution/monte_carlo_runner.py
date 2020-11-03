from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from typing import Dict, Any
from tqdm import tqdm
from openmodelica_microgrid_gym.agents.episodic import EpisodicLearnerAgent
from openmodelica_microgrid_gym.env import ModelicaEnv
import matplotlib.pyplot as plt


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
    Therefore, the agent-observe function is called with terminated = True
    """

    def __init__(self, agent: EpisodicLearnerAgent, env: ModelicaEnv):
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

    def run(self, n_episodes: int = 10, n_mc: int = 5, visualise: bool = False, prepare_mc_experiment=lambda: True):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param n_mc: number of Monte-Carlo experiments using the same parameter set before updating the latter
        :param visualise: turns on visualization of the environment
        """
        t = np.linspace(0, self.env.max_episode_steps*self.env.net.ts, self.env.max_episode_steps)
        self.agent.reset()
        self.env.history.cols = self.env.history.structured_cols(None) + self.agent.measurement_cols
        self.agent.obs_varnames = self.env.history.cols

        initial_performance_mc = np.zeros(n_mc)
        performance_mc = np.zeros(n_mc)

        if not visualise:
            self.env.viz_mode = None
        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            done, r = False, None
            np.random.seed(0)
            for m in tqdm(range(n_mc), desc='monte_carlo_run', unit='epoch', leave=False):
                prepare_mc_experiment()

                r_vec = np.zeros(self.env.max_episode_steps)

                obs = self.env.reset()

                for p in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                    self.agent.observe(r, False)
                    act = self.agent.act(obs)
                    self.env.measurement = self.agent.measurement
                    self.env.history.amend(self.agent.measurement, (-len(self.agent.measurement),None))
                    obs, r, done, info = self.env.step(act)
                    r_vec[p] = r
                    self.env.render()
                    if done:
                        self.agent.observe(r,
                                           False)  # take the last reward into account, too, but without update_params

                        # v_max = np.abs(self.env.history.df[[f'lc.capacitor{j}.v' for j in '123']].to_numpy()).max()
                        # if v_max > 25:
                        #    asd = self.agent.episode_return + v_max

                        #self.agent.performance = self.agent._iterations / (
                        #        self.agent.episode_return * self.agent.initial_performance)
                        self.agent.performance = (self.agent.episode_return - self.agent.min_performance) \
                                                 / (self.agent.initial_performance - self.agent.min_performance)

                        if m == 0 and i == 0:
                            self.agent.initial_performance = self.agent.episode_return
                            self.agent.performance = 1  # instead of perf/initial_perf
                            self.agent.last_best_performance = self.agent.performance
                            self.agent.last_worst_performance = self.agent.performance

                            self.agent.best_episode = self.agent.history.df.shape[0]
                            self.agent.last_best_performance = self.agent.performance
                            self.agent.worst_episode = self.agent.history.df.shape[0]
                            self.agent.last_worst_performance = self.agent.performance

                        performance_mc[m] = self.agent.performance
                        initial_performance_mc[m] = self.agent.episode_return
                        # set iterations and episode return = 0
                        self.agent.prepare_episode()

                        break

                plt.plot(t, r_vec)
                plt.ylabel('Reward')
                plt.grid(True)
                plt.savefig('Ki_rewTest' + '/Rew_abc_ohneBuffer{}.pdf'.format(m))
                plt.show()

                plt.plot(t, np.cumsum(r_vec))
                plt.ylabel('Cummulated Reward')
                plt.grid(True)
                plt.savefig('Ki_rewTest'+'/cumRew_abc_ohneBuffer{}.pdf'.format(m))
                plt.show()


                _, env_fig = self.env.close()

                # vor break?
                if (m == 0 and i == 0):  # or self.agent.has_improved:
                    self.run_data['best_env_plt'] = env_fig
                    self.run_data['best_episode_idx'] = i
                    self.agent.last_best_performance = self.agent.performance

                if (m == 0 and i == 0):  # or self.agent.has_worsened:
                    self.run_data['worst_env_plt'] = env_fig
                    self.run_data['worst_episode_idx'] = i
                    self.agent.last_worst_performance = self.agent.performance

                # plt.close(env_fig)

            if i == 0:
                # performance was normalized to first run -> use average of first episode so that J_initial for first
                # is 1

                eps_ret = performance_mc*(self.agent.initial_performance - self.agent.min_performance) +self.agent.min_performance
                self.agent.initial_performance = np.mean(eps_ret)
                performance_mc = (eps_ret - self.agent.min_performance) \
                                                 / (self.agent.initial_performance - self.agent.min_performance)

            self.agent.performance = np.mean(performance_mc)
            self.agent.update_params()
            #if self.agent.unsafe:
                # toDo: remove - only needed for lengthscale search
                #break

            if visualise:
                agent_fig = self.agent.render()

            self.run_data['last_agent_plt'] = agent_fig
