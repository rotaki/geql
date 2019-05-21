import getch
import os
import sys
import signal
import time
import imageio
import os
import pickle

import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros.actions import *
import MarioRLAgent
import TrainingStats

import impl.EpsilonGreedyActionPolicy as EGAP
import impl.ClusterEpsilonGreedyActionPolicy as CEGAP
import impl.TabularQEstimator as TabQ
import impl.GBoostedQEstimator as GBQ
import impl.AggressiveDSPolicy as ADSP
import numpy as np
import pandas as pd

from StateEncodingParams import StateEncodingParams

class MarioRLUI(MarioRLAgent.IMarioRLAgentListener):
    def __init__(self,
                 environment,
                 learning_policy,
                 q_estimator,
                 action_policy,
                 action_set,
                 action_interval = 6,
                 headless=True):
        
        self.q_estimator = q_estimator if q_estimator is not None else None
        self.rl_agent = MarioRLAgent.MarioRLAgent(
            environment,
            self.q_estimator,
            action_policy,
            action_set,
            action_interval = action_interval,
            listener = self,
            learning_policy=learning_policy)

        self.headless = headless
        if self.headless:
            self.rl_agent.render_option = MarioRLAgent.RenderOption.NoRender
        else:
            self.rl_agent.render_option = MarioRLAgent.RenderOption.ActionFrames
            
        self.paused = False
        self.verbose = False
        self.should_quit = False
        self.ask_movie = False
        self.best_fitness = 0
        self.best_time = float('inf')
        self.last_episode_finished = 0

        self.output_dir = 'output_{}/'.format(time.strftime('%Y-%m-%d_%H%M%S'))
        os.mkdir(self.output_dir)

        self.sync_interval = 5000
        self.training_stats = TrainingStats.TrainingStats(q_estimator.summary(),
                                                          action_policy.summary(),
                                                          learning_policy.describe(),
                                                          ma_width=100)
        if not self.headless:
            self.training_stats.plot()
            
        signal.signal(signal.SIGINT, self.make_signal_handler())

    def make_signal_handler(self):
        def handler(signum, frame):
            if self.paused:
                if self.confirm_quit():
                    self.should_quit = True
            else:
                print('Ctrl-C caught! Pausing...')
                self.paused = True
        return handler
        
    def episode_finished(self,
                         episode_number,
                         wall_time_elapsed,
                         game_time_elapsed,
                         n_frames,
                         fitness,
                         sa_sequence):
        self.last_episode_finished = episode_number
        self.training_stats.add_episode_stats(wall_time_elapsed,
                                              game_time_elapsed,
                                              n_frames,
                                              fitness)

        if (fitness == self.best_fitness and game_time_elapsed < self.best_time) or \
           fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_time = game_time_elapsed
            self.make_movie(sa_sequence,
                            self.output_dir + 'best_f{}_t{}_e{}.mp4'.format(
                                fitness, game_time_elapsed, episode_number))
        elif self.ask_movie:
            while True:
                print('Export movie of last episode? (Y/N)')
                try:
                    char = getch.getch()
                except OverflowError:
                    print('Invalid input')
                    continue

                if char == 'y' or char == 'Y':
                    self.make_movie(sa_sequence,
                                    self.output_dir +
                                    'recording_f{}_t{}_e{}.mp4'.format(
                                        fitness, game_time_elapsed, episode_number))
                    break
                elif char =='n' or char == 'N':
                    print('No movie exported')
                    break
                else:
                    print('Invalid input')

        self.training_stats.export(self.output_dir + 'training_stats.txt')
        if self.headless:
            self.training_stats.print_stats()
        else:
            self.training_stats.plot()

        # Always sync first episode to get early indication on error
        if episode_number % self.sync_interval == 0 or episode_number == 1:
            if self.headless:
                self.make_snapshot()
                self.sync_home()
        
    def main_loop(self):
        while not self.should_quit:
            print('\nMarioRL: [(v)erbose: {}] [(r)endering: {}] [re(c)ording: {}]'.
                  format(
                      self.verbose,
                      str(self.rl_agent.render_option),
                      'Ask each episode' if self.ask_movie else 'Best'
                  ))
            print('Commands: (t)rain, (s)tep, (q)uit, sna(p)shot')
            try:
                char = getch.getch()
            except OverflowError:
                continue
                
            if char == 'v':
                self.toggle_verbose()
            elif char == 'r':
                self.toggle_rendering()
            elif char == 'c':
                self.toggle_recording()
            elif char == 't':
                print('Training... (Ctrl-C to pause and return to menu)')
                self.train()
            elif char == 'p':
                self.make_snapshot()
                self.sync_home()
            elif char == 's':
                self.step()
                if self.verbose:
                    if isinstance(self.rl_agent.action_policy, CEGAP.ClusterEpsilonGreedyActionPolicy):
                        if self.rl_agent.action_policy.cluster_model is not None:
                            print('M(c, a) table:')
                            sep = '+'
                            action_count_table = pd.DataFrame(data = self.rl_agent.action_policy.show_action_count().astype('int'),
                                                              columns = np.array([sep.join(i) for i in self.rl_agent.action_set]),
                                                              index = range(self.rl_agent.action_policy.s_e_p.n_clusters))
                            print(action_count_table)
                        else:
                            print("No cluster yet")
                    if isinstance(self.rl_agent.action_policy, ADSP.AggressiveDSPolicy):
                        print("M(c, a) table : up to 40 states")
                        sep = '+'
                        d = self.rl_agent.action_policy.show_action_count()
                        action_count_table = pd.DataFrame.from_dict(d,
                                                                    orient = 'index',
                                                                    columns = np.array([sep.join(i) for i in self.rl_agent.action_set]))
                        action_count_table.index = range(len(d))
                        print("Number of collected states: {}".format(len(d)))
                        print(action_count_table.head(40))
                        
            elif char == 'q':
                if self.confirm_quit():
                    self.should_quit = True
                    break


    def toggle_verbose(self):
        self.verbose = not self.verbose
        self.rl_agent.verbose = self.verbose
        self.q_estimator.verbose = self.verbose

    def toggle_rendering(self):
        if self.headless:
            # Force norender
            self.rl_agent.render_option = MarioRLAgent.RenderOption.NoRender
        elif self.rl_agent.render_option == MarioRLAgent.RenderOption.NoRender:
            self.rl_agent.render_option = MarioRLAgent.RenderOption.ActionFrames
        elif self.rl_agent.render_option == MarioRLAgent.RenderOption.ActionFrames:
            self.rl_agent.render_option = MarioRLAgent.RenderOption.All
        elif self.rl_agent.render_option == MarioRLAgent.RenderOption.All:
            self.rl_agent.render_option = MarioRLAgent.RenderOption.NoRender
        else:
            raise RuntimeError('Unknown render option')

    def toggle_recording(self):
        self.ask_movie = not self.ask_movie
        
    def train(self):
        self.paused = False
        while not self.paused:
            self.rl_agent.step()
                    
    def step(self):
        self.rl_agent.step()
        
    def confirm_quit(self):
        try:
            print('Are you sure you would like to quit (Y)?')
            char = getch.getch()
            return char == 'y' or char == 'Y'
        except OverflowError:
            return False

    def make_snapshot(self):
        if not self.headless:
            print('Not making a snapshot, since snaphots are only supported in headless mode')
        filename = self.output_dir + 'snapshot_e{}.dat'.format(
            self.last_episode_finished)
        pickle.dump(self, open(filename, 'wb'))
        print('Snapshot saved to {}'.format(filename))

    def sync_home(self):
        rsync_command = 'rsync -avRzq {} mario@portalgatan.mynetgear.com:.'.format(
            self.output_dir[:-1]) # Omit the trailing slash for rsync
        os.system(rsync_command)
        
    def make_movie(self, sa_sequence, filename):
        frames = []
        writer = imageio.get_writer(filename, fps=60.0, quality=10.0)
        for (s, a) in sa_sequence:
            writer.append_data(s)

        writer.close()
        print('Saved episode animation to {}'.format(filename))

            
if __name__ == '__main__':
    # Set up the model

    env = gym_smb.make('SuperMarioBros-v0')
    action_set = COMPLEX_MOVEMENT
    env = BinarySpaceToDiscreteSpaceEnv(env, action_set)
    action_list = list(range(env.action_space.n))
    # action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list,
    #                                                epsilon=0.1,
    #                                                decay_factor = 0.5,
    #                                                decay_interval = 10000)



    state_encoding_params = StateEncodingParams(resize_factor=8,
                                                pixel_intensity=8)

    # action_policy = CEGAP.ClusterEpsilonGreedyActionPolicy(actions=action_list,
    #                                                        epsilon=0.1,
    #                                                        state_encoding_params=state_encoding_params)
        
    action_policy = ADSP.AggressiveDSPolicy(actions=action_list,
                                            epsilon=0.1,
                                            state_encoding_params = state_encoding_params)


    greedy_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list,
                                                   epsilon=0)

    learning_policy = MarioRLAgent.LearningPolicy.SARSA


    # q_estimator = TabQ.TabularQEstimator(discount=0.5,
    #                                      steps=10,
    #                                      learning_rate=0.1,
    #                                      learning_policy=learning_policy,
    #                                      q_action_policy=None)
    q_estimator = GBQ.GBoostedQEstimator(discount=0.9,
                                         steps=1,
                                         learning_rate=0.5,
                                         learning_policy=learning_policy,
                                         q_action_policy=greedy_policy)
    
    
    app = MarioRLUI(env,
                    learning_policy,
                    q_estimator,
                    action_policy,
                    action_set,
                    headless=True)
    
    app.main_loop()
    env.close()
