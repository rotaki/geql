import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class TrainingStats:
    def __init__(self, q_estimator_desc, action_policy_desc, comment=None, ma_width=20): 
        self.comment = '' if comment is None else '\t' + comment
        self.ma_width = ma_width
        self.n_episodes = 0
        self.episode_fitness = []
        self.episode_game_time = []
        self.episode_time = []
        self.episode_frame_count = []
        self.q_estimator_desc = q_estimator_desc
        self.action_policy_desc = action_policy_desc
        self.fig = plt.figure()
        self.fig.suptitle('$Q(s,a)$: ' + q_estimator_desc +
                          '\t $\pi(s,a)$:' + action_policy_desc +
                          self.comment, fontsize=8)
        spec = gridspec.GridSpec(ncols = 1, nrows = 4, figure = self.fig)
        self.episode_fitness_graph = self.fig.add_subplot(spec[0:3,0])
#        self.time_graph = self.episode_fitness_graph.twinx()
        self.eps_graph = self.fig.add_subplot(spec[3,0], sharex = self.episode_fitness_graph)
        self.fps_graph = self.eps_graph.twinx()
        plt.ion()

    def moving_average(x, w):
        if len(x) == 0:
            return np.array([])
        convolved = np.convolve(x, np.ones(w), 'full')
        # Normalize the first elements separately (they are not over w samples)
        first_element_normalizers = np.array(range(1, w))
        convolved[0:w-1] = convolved[0:w-1] / first_element_normalizers
        # Normalize the rest of the elements
        convolved[w-1:len(x)] /= w
        return convolved[0:len(x)]

    def export(self, filename):
        episode_number = list(range(1, self.n_episodes + 1))
        table = np.column_stack([episode_number,
                                 self.episode_fitness,
                                 self.episode_game_time,
                                 self.episode_time,
                                 self.episode_frame_count])
        np.savetxt(filename,
                   table,
                   fmt=['%d',
                        '%.2f',
                        '%d',
                        '%.5f',
                        '%d'],
                   header='episode_number episode_fitness game_time wall_time frame_count\t' +
                   ' Q: {} P: {} Other: {}'.format(self.q_estimator_desc,
                                                   self.action_policy_desc,
                                                   self.comment))

    def print_stats(self):
        # TODO A bit overkill to calculate MA for the entire sequence when we only
        # want the last
        ma = TrainingStats.moving_average(self.episode_fitness, self.ma_width)
        fps = self.episode_frame_count[-1] / self.episode_time[-1]
        print('Episode #{} stats: fitness={} (MA{}={}), game_time={}, fps={}, frame_count={}, wall_time={}\n'.format(
            self.n_episodes,
            self.episode_fitness[-1],
            self.ma_width,
            ma[-1],
            self.episode_game_time[-1],
            fps,
            self.episode_frame_count[-1],
            self.episode_time[-1]
        ))
        
    def add_episode_stats(self, real_time_elapsed, game_time_elapsed, frames, fitness):
        self.episode_time.append(real_time_elapsed)
        self.episode_game_time.append(game_time_elapsed)
        self.episode_fitness.append(fitness)
        self.episode_frame_count.append(frames)
        self.n_episodes += 1
    
    def plot(self):
        n_episodes = len(self.episode_fitness)
        # Episode fitness
        self.episode_fitness_graph.clear()
        self.episode_fitness_graph.set_ylabel('fitness')
        self.episode_fitness_graph.tick_params(axis='y', colors='b')

        x = list(range(1, n_episodes + 1))
        # Samples (dots)
        self.episode_fitness_graph.plot(x, self.episode_fitness,
                                        color='cornflowerblue',
                                        marker='.',
                                        linestyle='',
                                        zorder=5)
        # Moving average
        ma = TrainingStats.moving_average(self.episode_fitness, self.ma_width)
        self.episode_fitness_graph.plot(x, ma, 'b--', zorder=10)
        self.episode_fitness_graph.set_ylim(bottom=0)
        # Show x on the lowest subgraph instead
        self.episode_fitness_graph.grid(b=True, axis='both')
        self.episode_fitness_graph.tick_params(axis='x', bottom=False, top=False, colors='w')
        
        # Time
        # self.time_graph.clear()
        # self.time_graph.plot(x, self.episode_game_time,
        #                      color='salmon',
        #                      marker='.',
        #                      linestyle='',
        #                      zorder=1)
        # self.time_graph.set_ylim(bottom=0)
        # self.time_graph.tick_params(axis='y', colors='r')
        # self.time_graph.set_ylabel('episode time')



        # EPS
        self.eps_graph.clear()
        eps = (60*60) / np.array(self.episode_time)
        eps_ma = TrainingStats.moving_average(eps, self.ma_width)
        self.eps_graph.plot(x, eps,
                            color='cornflowerblue',
                            marker='.',
                            linestyle='')
        self.eps_graph.plot(x, eps_ma, 'b--')
        self.eps_graph.set_ylim(bottom=0)
        self.eps_graph.set_ylabel('EPH')
        self.eps_graph.tick_params(axis='y', colors='b')

        
        # FPS
        self.fps_graph.clear()
        fps = np.array(self.episode_frame_count) / np.array(self.episode_time)
        self.fps_graph.plot(x, fps, 'r')
        self.fps_graph.set_ylabel('FPS')
        self.fps_graph.set_ylim(bottom=0)
        self.fps_graph.tick_params(axis='y', colors='r')

        self.eps_graph.set_xlabel('episode')
        self.eps_graph.set_xlim(left=1, right=max(2,n_episodes))
        self.eps_graph.grid(b=True, axis='both')
    
        plt.pause(0.1)

    def close(self):
        plt.close('all')
