#####################################
# Example using an FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt


import logging
from functools import partial
from itertools import repeat
from time import strftime, gmtime
from typing import List
import os

import GPy
import gym
import numpy as np

import matplotlib.pyplot as plt

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQCurrentSourcingController, \
    MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.aux_ctl.observers import Lueneberger
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.stochastic_components import Load, Noise
from openmodelica_microgrid_gym.execution.monte_carlo_runner import MonteCarloRunner
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory


include_simulate = True
show_plots = True
balanced_load = False
do_measurement = False

# If True: Results are stored to directory mentioned in: REBASE to DEV after MERGE #60!!
safe_results = False

# Simulation definitions
net = Network.load('../net/net_single-inv-Paper_Loadstep.yaml')
delta_t = 1e-4  # simulation time step size / s
max_episode_steps = 2000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
n_MC = 1
v_DC = 650  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
vLimit = 1000  # inverter current limit / A
vNominal = 600  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 0.0  # virtual droop gain for active power / W/Hz
QDroopGain = 0.0  # virtual droop gain for reactive power / VAR/V

#vLimit = 100
#vNominal = 25

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, r'Paper_Vctrl_obs_sim_2')
os.makedirs(save_folder, exist_ok=True)

L_filt = 2.3e-3  # / H
R_filt = 400e-3  # 170e-3  # 585e-3  # / Ohm
C_filt = 48e-6#13.6e-6   # / F
#C_filt = 10e-6   # / F
R = 20#7.15  # 170e-3  # 585e-3  # / Ohm

tau_plant = L_filt/R_filt
gain_plant = 1/R_filt

# take inverter into account uning s&h (exp(-s*delta_T/2))

Tn = tau_plant   # Due to compensate
Kp_init = tau_plant/(2*delta_t*gain_plant*v_DC)
Ki_init = Kp_init/(Tn)

# Observer matrices
A = np.array([[-R_filt, -1/L_filt, 0       ],
              [1/C_filt, 0,       -1/C_filt],
              [0,        0,        0       ]])

B = np.array([[1/L_filt, 0, 0]]).T

C = np.array([[1, 0, 0],
              [0, 1, 0]])

# L_iL_iL = 6e3       #4.0133e+03
# L_vc_iL = -431.87   #-316.2432     # influence from delta_y_vc onto xdot = iL
#
# L_iL_vc = 7.3524e4  #7.2952e+04
# L_vc_vc = 1e4       #6.9871e+03
#
# L_iL_io = 28.5158   #-1.6880e+03
# L_vc_io = -557.6191  #-148.1438

# for C = 13.6e-6*10:!!!!
# L_iL_iL = 6e3       #4.0133e+03
# L_vc_iL = -434.7826   #-316.2432     # influence from delta_y_vc onto xdot = iL
#
# L_iL_vc = 7.3529  #7.2952e+04
# L_vc_vc = 1e4       #6.9871e+03
#
# L_iL_io = 4.4165e-13   #-1.6880e+03
# L_vc_io = -5576#-557.6191  #-148.1438


# # for C = 48e-6:!!!!
L_iL_iL = 6e3       #4.0133e+03
L_vc_iL = -435.5632   #-316.2432     # influence from delta_y_vc onto xdot = iL

L_iL_vc = 2.0833e4  #7.2952e+04
L_vc_vc = 1e4       #6.9871e+03

L_iL_io = -2.3581   #-1.6880e+03
L_vc_io = -1.968e3

# # for C = 80e-6:!!!!
# L_iL_iL = 6e3       #4.0133e+03
# L_vc_iL = -434.7826   #-316.2432     # influence from delta_y_vc onto xdot = iL
#
# L_iL_vc = 1.25e4  #7.2952e+04
# L_vc_vc = 1e4       #6.9871e+03
#
# L_iL_io = -1.75e-12   #-1.6880e+03
# L_vc_io = -3280

L = np.array([[L_iL_iL, L_vc_iL],
              [L_iL_vc, L_vc_vc],
              [L_iL_io, L_vc_io]])


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                 [f'lc.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0']])

    def rew_fun(self, cols: List[str], data: np.ndarray) -> float:
        """
        Defines the reward function for the environment. Uses the observations and set-points to evaluate the quality of
        the used parameters.
        Takes current and voltage measurements and set-points to calculate the mean-root control error and uses a
        logarithmic barrier function in case of violating the current limit. Barrier function is adjustable using
        parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        phase = data[idx[1]]  # phase from the master controller needed for transformation
        vabc_master = data[idx[3]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_dq0_master = data[idx[2]]  # setting dq current reference
        isp_abc_master = dq0_to_abc(isp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates
        vsp_dq0_master = data[idx[4]]  # setting dq voltage reference
        vsp_abc_master = dq0_to_abc(vsp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = (np.sum((np.abs((vsp_abc_master - vabc_master)) / vLimit) ** 0.5, axis=0)\
                + -np.sum(mu * np.log(1 - np.maximum(np.abs(vabc_master) - vNominal, 0) / (vLimit - vNominal)), axis=0)) \
                / max_episode_steps
            #np.sum((np.abs((isp_abc_master - iabc_master)) / iLimit) ** 0.5, axis=0) \
             #   + -np.sum(mu * np.log(1 - np.maximum(np.abs(iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \


        return -error.squeeze()


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    prior_mean = 0  # 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001# ** 2  # measurement noise sigma_omega
    prior_var = 2  # prior variance of the GP

    # Choose Kp and Ki (current and voltage controller) as mutable parameters (below) and define bounds and lengthscale
    # for both of them
    #bounds = [(0.0, 0.03), (0, 300), (0.0, 0.03),
    #          (0, 300)]  # bounds on the input variable current-Ki&Kp and voltage-Ki&Kp
    #lengthscale = [.005, 50., .005,
    #               50.]  # length scale for the parameter variation [current-Ki&Kp and voltage-Ki&Kp] for the GP
    bounds = [(0.0, 0.2), (0, 2000)]  # bounds on the input variable current-Ki&Kp and voltage-Ki&Kp
    #bounds = [(0.0, 0.05), (0, 250)]  # bounds on the input variable current-Ki&Kp and voltage-Ki&Kp
    #lengthscale = [.02, 100.]  # length scale for the parameter variation [current-Ki&Kp and voltage-Ki&Kp] for the GP
    lengthscale = [.05, 200.]  # length scale for the parameter variation [current-Ki&Kp and voltage-Ki&Kp] for the GP

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means: performance measurement for optimization are seen as
    # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 0.5

    # The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
    # expanding points eventually.
    # The following variable is multiplied with the first performance of the initial set by the factor below:
    explore_threshold = 0

    # Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
    # limit exceeded
    abort_reward = -10

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers
    # Choose Kp and Ki for the current and voltage controller as mutable parameters
    #mutable_params = dict(currentP=MutableFloat(10e-3), currentI=MutableFloat(10), voltageP=MutableFloat(0.05),
    #                      voltageI=MutableFloat(75))


#    mutable_params = dict(voltageP=MutableFloat(0.5), voltageI=MutableFloat(120))
    mutable_params = dict(voltageP=MutableFloat(0.03), voltageI=MutableFloat(12))

    # Vdc = 650 V
    mutable_params = dict(voltageP=MutableFloat(0.125), voltageI=MutableFloat(70.871))
    mutable_params = dict(voltageP=MutableFloat(0.02), voltageI=MutableFloat(7))

    voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                    limits=(-iLimit, iLimit))
    #current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))
    #current_dqp_iparams = PI_params(kP=0.0062, kI=24.9, limits=(-1, 1))     # Best set from paper III-D
    #current_dqp_iparams = PI_params(kP=0.2, kI=33.3, limits=(-1, 1))     # Best set from paper III-D

    # Vdc = 650 V
    current_dqp_iparams = PI_params(kP=0.017, kI=3, limits=(-1, 1))     # Best set from paper III-D

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    # Define a voltage forming inverter using the PIPI and droop parameters from above
    ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param, qdroop_param,
                                       observer = [Lueneberger(*params) for params in repeat((A, B, C, L, delta_t, v_DC),3)], undersampling=1, name='master')

    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    # Arguments described above
    # History is used to store results
    agent = SafeOptAgent(mutable_params,
                         abort_reward,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold),
                         [ctrl],
                         dict(master=[[f'lc.inductor{k}.i' for k in '123'],
                                      [f'lc.capacitor{k}.v' for k in '123'],
                                      ]),
                         history=FullHistory()
                         )


    class PlotManager:

        def __init__(self, used_agent: SafeOptAgent, used_r_load: Load, used_r_filt: Load, used_l_filt: Load, used_c_filt: Load, used_i_noise: Noise):
            self.agent = used_agent
            self.r_load = used_r_load
            self.r_filt = used_r_filt
            self.l_filt = used_l_filt
            self.c_filt = used_c_filt
            self.i_noise = used_i_noise

            # self.r_load.gains =  [elem *1e3 for elem in self.r_load.gains]
            # self.l_load.gains =  [elem *1e3 for elem in self.r_load.gains]

        def set_title(self):
            plt.title('Simulation: J = {:.2f}; R = {} \n L = {}; \n noise = {}'.format(self.agent.performance,
                                                                                       ['%.4f' % elem for elem in
                                                                                        self.r_load.gains],
                                                                                       ['%.6f' % elem for elem in
                                                                                        self.l_filt.gains],
                                                                                       ['%.4f' % elem for elem in
                                                                                        self.i_noise.gains]))

        def save_abc(self, fig):
            if safe_results:
                fig.savefig(save_folder + '/J_{}_i_abc.pdf'.format(self.agent.performance))

        def save_abc_v(self, fig):
            if safe_results:
                fig.savefig(save_folder + '/J_{}_v_abc.pdf'.format(self.agent.performance))

        def save_dq0(self, fig):
            if safe_results:
                fig.savefig(save_folder + '/J_{}_i_dq0.pdf'.format(self.agent.performance))


    #####################################
    # Definition of the environment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed.
    #   Labels and grid is adjusted using the PlotTmpl (For more information, see UserGuide)
    #   generated figures are stored to file
    # - inputs to the models are the connection points to the inverters (see user guide for more details)
    # - model outputs are the 3 currents through the inductors and the 3 voltages across the capacitors

    # Defining unbalanced loads sampling from Gaussian distribution with sdt = 0.2*mean
    #r_load = Load(R, 0.1 * R, balanced=balanced_load, tolerance=0.1)
    #l_load = Load(L, 0.1 * L, balanced=balanced_load, tolerance=0.1)
    #i_noise = Noise([0, 0, 0], [0.0822, 0.103, 0.136], 0.05, 0.2)


    #r_filt = Load(R_filt, 0 * R_filt, balanced=balanced_load)
    #l_filt = Load(L_filt, 0 * L_filt, balanced=balanced_load)
    #c_filt = Load(C_filt, 0 * C_filt, balanced=balanced_load)
    #r_load = Load(R, 0 * R, balanced=balanced_load)
    #meas_noise = Noise([0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0)

    r_filt = Load(R_filt, 0.1 * R_filt, balanced=balanced_load)
    l_filt = Load(L_filt, 0.1 * L_filt, balanced=balanced_load)
    c_filt = Load(C_filt, 0.1 * C_filt, balanced=balanced_load)
    r_load = Load(R, 0.1 * R, balanced=balanced_load)
    meas_noise = Noise([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.45, 0.39, 0.42, 0.0023, 0.0015, 0.0018, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, 0.5)

    #i_noise = Noise([0, 0, 0], [0.0023, 0.0015, 0.0018], 0.0005, 0.32)


    def reset_loads():
        r_load.reset()
        r_filt.reset()
        l_filt.reset()
        c_filt.reset()
        meas_noise.reset()


    # plotter = PlotManager(agent, [r_load, l_load, i_noise])
    plotter = PlotManager(agent, r_load, r_filt, l_filt, c_filt, meas_noise)

    class Multiplot():

        def __init__(self):
            self.axes_list = []
            self.n_plts = 3

        def record(self, fig):
            if len(self.axes_list) == self.n_plts:
                plt.show()


    def xylables(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        #plotter.set_title()
        plotter.save_abc(fig)
        #plt.xlim([0.1215, 0.123])
        # plt.title('Simulation')
        # time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # if safe_results:
        #    fig.savefig(save_folder + '/abc_current' + time + '.pdf')
        # fig.savefig('Sim_vgl/abc_currentJ_{}_abcvoltage.pdf'.format())
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_hat(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{o estimate,abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        #plotter.set_title()
        plotter.save_abc(fig)
        #plt.xlim([0.1215, 0.123])
        # plt.title('Simulation')
        # time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # if safe_results:
        #    fig.savefig(save_folder + '/abc_current' + time + '.pdf')
        # fig.savefig('Sim_vgl/abc_currentJ_{}_abcvoltage.pdf'.format())
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


    def xylables_v(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        #plotter.set_title()
        plotter.save_abc_v(fig)
        #plt.xlim([0.1215, 0.123])
        # plt.title('Simulation')
        # time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # if safe_results:
        #    fig.savefig(save_folder + '/abc_current' + time + '.pdf')
        # fig.savefig('Sim_vgl/abc_currentJ_{}_abcvoltage.pdf'.format())
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def xylables_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        plotter.set_title()
        #plotter.save_dq0(fig)
        #plt.ylim(0, 36)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


    def xylables_mdq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$m_{\mathrm{dq0}}\,/\,\mathrm{}$')
        plt.title('Simulation')
        ax.grid(which='both')
        # plt.ylim(0,36)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)




    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   #time_step=delta_t,
                   viz_cols=[
                       PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'master.SPV{i}' for i in 'abc']],
                                callback=xylables_v,
                                color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                style=[[None], ['--']]
                                ),
                       PlotTmpl([[f'lc.inductor{i}.i' for i in '123'], [f'master.SPI{i}' for i in 'abc']],
                                callback=xylables,
                                color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                style=[[None], ['--']]
                                ),
                       PlotTmpl([[f'master.I_hat{i}' for i in 'abc'], [f'r_load.resistor{i}.i' for i in '123'],],
                                callback=xylables_hat,
                                color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                style = [[None], ['--']]
                                ),
                       # PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                       #          callback=xylables_R,
                       #          color=[['b', 'r', 'g']],
                       #          style=[[None]]
                       #          ),
                       # PlotTmpl([[f'master.I_hat{i}' for i in 'dq0'], [f'master.CVi{i}' for i in 'dq0'], ],
                       #          callback=xylables_hat,
                       #          color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                       #          style=[[None], ['--']]
                       #          )
                   ],
                   # viz_cols = ['inverter1.*', 'rl.inductor1.i'],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   # model_params={'inverter1.gain.u': v_DC},
                   model_params={'lc.resistor1.R': R_filt,#partial(r_filt.load_step, n=0),
                                 'lc.resistor2.R': R_filt,#partial(r_filt.load_step, n=1),
                                 'lc.resistor3.R': R_filt,#partial(r_filt.load_step, n=2),
                                 'lc.inductor1.L': L_filt,#partial(l_filt.load_step, n=0),
                                 'lc.inductor2.L': L_filt,#partial(l_filt.load_step, n=1),
                                 'lc.inductor3.L': L_filt,#partial(l_filt.load_step, n=2),
                                 'lc.capacitor1.C': C_filt,#partial(c_filt.load_step, n=0),
                                 'lc.capacitor2.C': C_filt,#partial(c_filt.load_step, n=1),
                                 'lc.capacitor3.C': C_filt,#partial(c_filt.load_step, n=2),
                                 'r_load.resistor1.R': partial(r_load.load_step, n=0),
                                 'r_load.resistor2.R': partial(r_load.load_step, n=1),
                                 'r_load.resistor3.R': partial(r_load.load_step, n=2),
                                 },
                   net=net,
                   #model_path='../omg_grid/omg_grid.Grids.Paper_Loadstep.fmu',
                   model_path='../fmu/grid.paper_loadstep.fmu',
                   #model_input=['i1p1', 'i1p2', 'i1p3'],
                   # model_output=dict(#rl=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                   #model_output=dict(lc=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                   #                       ['capacitor1.v', 'capacitor2.v', 'capacitor3.v'],
                   #                       ['capacitor1.i', 'capacitor2.i', 'capacitor3.i']],  # ,
                   #                  # ['resistor1.R', 'resistor2.R', 'resistor3.R'],
                   #                  # ['inductor1.L', 'inductor2.L', 'inductor3.L']]
                   #                  r_load=[['resistor1.i', 'resistor2.i', 'resistor3.i']]
                   #                  # inverter1=['inductor1.i', 'inductor2.i', 'inductor3.i']
                   #                  ),
                   history=FullHistory(),
                   state_noise=meas_noise,
                   action_time_delay=1
                   )

    runner = MonteCarloRunner(agent, env)

    runner.run(num_episodes, n_mc=n_MC, visualise=True, prepare_mc_experiment=reset_loads)
    env.history.df.to_hdf('env_hist_obs2plt.hd5', 'hist')
    #df2 = pd.read_hdf('env_hist_obs2plt.hd5', 'hist')
    print(agent.unsafe)

    #if agent.unsafe:
    #    unsafe_vec[ll] = 1
    #else:
    #    unsafe_vec[ll] = 0

    #####################################
    # Performance results and parameters as well as plots are stored in folder pipi_signleInvALT
    # Show best episode measurment (current) plot
    best_env_plt = runner.run_data['best_env_plt']
    ax = best_env_plt[0].axes[0]
    ax.set_title('Best Episode')
    best_env_plt[0].show()
    # best_env_plt[0].savefig('best_env_plt.png')

    # Show worst episode measurment (current) plot
    best_env_plt = runner.run_data['worst_env_plt']
    ax = best_env_plt[0].axes[0]
    ax.set_title('Worst Episode')
    best_env_plt[0].show()
    # best_env_plt[0].savefig('worst_env_plt.png')

    best_agent_plt = runner.run_data['last_agent_plt']
    ax = best_agent_plt.axes[0]
    ax.grid(which='both')
    ax.set_axisbelow(True)


    agent.params.reset()
    ax.set_ylabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
    ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
    ax.get_figure().axes[1].set_ylabel(r'$J$')
    plt.title('Lengthscale = {}; balanced = '.format(lengthscale, balanced_load))
        # ax.plot([0.01, 0.01], [0, 250], 'k')
        # ax.plot([mutable_params['currentP'].val, mutable_params['currentP'].val], bounds[1], 'k-', zorder=1,
        #         lw=4,
        #         alpha=.5)
    best_agent_plt.show()
    if safe_results:
        best_agent_plt.savefig(save_folder + '/_agent_plt.pdf')
        best_agent_plt.savefig(save_folder + '/_agent_plt.pgf')
        agent.history.df.to_csv(save_folder + '/_result.csv')

    print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df.round({'J': 4, 'Params': 4})))
    print('\n Experiment finished with best set: \n')
    print('\n  Current-Ki&Kp and voltage-Ki&Kp = {}'.format(
        agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
    print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
    print('\n\nBest experiment results are plotted in the following:')

    # Show best episode measurment (current) plot
    #best_env_plt = runner.run_data['best_env_plt']
    #for ii in range(len(best_env_plt)):
    #    ax = best_env_plt[ii].axes[0]
    #    ax.set_title('Best Episode')
    #    best_env_plt[ii].show()

        # best_env_plt[0].savefig('best_env_plt.png')
