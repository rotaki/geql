import getch
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
import numpy as np
import pandas as pd

from StateCollector import PretrainingAgent
from MakeCluster import Cluster
from StateEncodingParams import StateEncodingParams


class MarioRLUI(MarioRLAgent.IMarioRLAgentListener):
    def __init__(self,
                 environment,
                 q_estimator,
                 action_policy,
                 action_set,
                 learning_policy = MarioRLAgent.LearningPolicy.SARSA,
                 action_interval = 10,
                 pretraining = False,
                 clustering_method = "kmeans",
                 n_clusters = 40,
                 sample_collect_interval=2,
                 resize_factor=8,
                 pixel_intensity=32,
                 headless=True):
        self.q_estimator = q_estimator if q_estimator is not None else None
        self.rl_agent = MarioRLAgent.MarioRLAgent(
            environment,
            self.q_estimator,
            action_policy,
            action_set,
            learning_policy,
            action_interval,
            self)
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
        
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters if n_clusters is not None else n_clusters
        self.sample_collect_interval = sample_collect_interval
        self.resize_factor = resize_factor
        self.pixel_intensity = pixel_intensity

        self.training_stats = TrainingStats.TrainingStats(q_estimator.summary(),
                                                          action_policy.summary(),
                                                          learning_policy.describe(),
                                                          ma_width=100)
        if not self.headless:
            self.training_stats.plot()

        if pretraining:
            self.training_stats.close()
            
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
                if self.verbose:
                    print('M(c, a) table:')
                self.step()
                if self.verbose:
                    sep = '+'
                    action_count_table = pd.DataFrame(data = self.rl_agent.action_policy.cluster.show_action_count().astype('int'),
                                                  columns = np.array([sep.join(i) for i in self.rl_agent.action_set]),
                                                  index = range(self.n_clusters))
                    print(action_count_table)
                
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

    def pretraining(self):
        # Store pretraining states
        if not os.path.exists("pretraining_states_ds{}_pi{}.npz".format(self.resize_factor, self.pixel_intensity)):
            from pathlib import Path
            Path('pretraining_states_ds{}_pi{}.npz'.format(self.resize_factor, self.pixel_intensity)).touch()
            np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(self.resize_factor, self.pixel_intensity), [-1])

        # Remove cluster image file if there
        if os.path.exists("cluster_img_ds{}_pi{}".format(self.resize_factor, self.pixel_intensity)):
            import shutil
            shutil.rmtree("cluster_img_ds{}_pi{}".format(self.resize_factor, self.pixel_intensity))

        # Create a new one
        os.makedirs("./cluster_img_ds{}_pi{}".format(self.resize_factor, self.pixel_intensity))

        encoding_info = StateEncodingParams(default_shape = self.rl_agent.env.observation_space.shape,
                                            resize_factor=self.resize_factor,
                                            pixel_intensity=self.pixel_intensity)
        # steps/sample_collect_interval >= n_clusters
        
        TA = PretrainingAgent(environment=self.rl_agent.env,
                              q_estimator=self.q_estimator,
                              action_policy=self.rl_agent.action_policy,
                              action_set = self.rl_agent.action_set,
                              action_interval=self.rl_agent.action_interval,
                              clustering_method=self.clustering_method,
                              n_clusters = self.n_clusters,
                              sample_collect_interval=self.sample_collect_interval,
                              state_encoding_params=encoding_info)
        
        C = Cluster(state_encoding_params = encoding_info,
                    action_space_size=self.rl_agent.env.action_space.n,
                    clustering_method=self.clustering_method,
                    n_clusters=self.n_clusters)

        pretraining_states = TA.get_pretraining_states()
        if pretraining_states.shape[0] < self.n_clusters:
            raise ValueError("Number of collected state is too small!!")
        C.cluster(pretraining_states)
        return C

            
if __name__ == '__main__':
    # Set up the model
    
    env = gym_smb.make('SuperMarioBros-v0')
    action_set = COMPLEX_MOVEMENT
    env = BinarySpaceToDiscreteSpaceEnv(env, action_set)
    action_list = list(range(env.action_space.n))
    
    action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list,
                                                   epsilon=0.1,
                                                   decay_factor = 0.5,
                                                   decay_interval = 10000)
    
    greedy_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list,
                                                   epsilon=0)
    
    learning_policy = MarioRLAgent.LearningPolicy.SARSA

    # q_estimator = TabQ.TabularQEstimator(discount=0.5,
    #                                      steps=10,
    #                                      learning_rate=0.1,
    #                                      learning_policy=learning_policy,
    #                                      q_action_policy=None)
    q_estimator = GBQ.GBoostedQEstimator(discount=0.9,
                                         steps=30,
                                         learning_rate=0.5,
                                         learning_policy=learning_policy,
                                         q_action_policy=greedy_policy)

    #app = MarioRLUI(env,
    #                q_estimator,
    #                action_policy,
    #                action_set,
    #                learning_policy,
    #                pretraining=True)
    
    #cluster = app.pretraining()
    
    # save cluster image to ./cluster_img
    #cluster.save_cluster_image()
        
    #action_policy = CEGAP.ClusterEpsilonGreedyActionPolicy(actions=action_list,
    #                                               epsilon=0.1,
    #                                               cluster=cluster)
    app = MarioRLUI(env,
                    q_estimator,
                    action_policy,
                    action_set,
                    learning_policy)
    app.main_loop()
    env.close()
