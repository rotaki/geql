import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros.actions import *
import os
import wx
import MarioRLAgent
import TrainingStats

import impl.EpsilonGreedyActionPolicy as EGAP
import impl.TabularQEstimator as TabQ



# Set up the model
action_set = COMPLEX_MOVEMENT
env = gym_smb.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, action_set)
action_list = list(range(env.action_space.n))
action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.05)
learning_policy = MarioRLAgent.LearningPolicy.SARSA
q_estimator = TabQ.TabularQEstimator(discount=0.8, learning_rate=0.2)

class MarioRLGUI(wx.App, MarioRLAgent.IMarioRLAgentListener):
    def __init__(self,
                 environment,
                 q_estimator,
                 action_policy,
                 action_set,
                 learning_policy = MarioRLAgent.LearningPolicy.SARSA,
                 action_interval = 6):
        wx.App.__init__(self)
        self.main_frame = MarioRLFrame(self)
        self.rl_agent = MarioRLAgent.MarioRLAgent(
            environment,
            q_estimator,
            action_policy,
            action_set,
            learning_policy,
            action_interval,
            self)
        self.rl_agent.render_option = MarioRLAgent.RenderOption.ActionFrames
        self.paused = False
        self.verbose = False
        self.training_stats = TrainingStats.TrainingStats(q_estimator.summary(),
                                                          action_policy.summary())
        self.training_stats.plot()
        self.Bind(wx.EVT_IDLE, self.on_idle)
        
    def on_idle(self, event):
        if not self.paused:
            self.rl_agent.step()
            event.RequestMore()

    def on_step(self, event):
        # Step only works when paused
        if self.paused:
            self.rl_agent.step()
            print('Step')
        else:
            print('Stepping is only enabled when paused')

    def on_toggle_pause(self, event):
        if self.paused:
            self.paused = False
            self.main_frame.is_running(True)
            print('Unpaused')
        else:
            self.paused = True
            self.main_frame.is_running(False)
            print('Paused')
            
    def on_verbose(self, event):
        if event.IsChecked():
            self.verbose = True
            self.rl_agent.verbose = True
            print('Verbose output enabled')
        else:
            self.verbose = False
            self.rl_agent.verbose = False
            print('Verbose output disabled')

    def on_render_option(self, event):
        new_option_str = event.GetString()
        if new_option_str == 'NoRender':
            new_option = MarioRLAgent.RenderOption.NoRender
            self.rl_agent.render_option = MarioRLAgent.RenderOption.NoRender
        elif new_option_str == 'ActionFrames':
            new_option = MarioRLAgent.RenderOption.ActionFrames
        elif new_option_str == 'All':
            new_option = MarioRLAgent.RenderOption.All
        else:
            raise RuntimeError('Unknown render option: {}'.
                               format(new_option_str))
        self.rl_agent.render_option = new_option

    def episode_finished(self,
                         episode_number,
                         wall_time_elapsed,
                         game_time_elapsed,
                         n_frames,
                         fitness):
        self.training_stats.add_episode_stats(wall_time_elapsed,
                                              game_time_elapsed,
                                              n_frames,
                                              fitness)
        self.training_stats.plot()
                
class MarioRLFrame(wx.Frame):
    def __init__(self, gui_app):
        wx.Frame.__init__(self, None, title='MarioRL Control', size=(200, 600))
        frame_panel = wx.Panel(self)
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)
        control_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.run_btn = wx.Button(frame_panel, label='Pause')
        self.Bind(wx.EVT_BUTTON, gui_app.on_toggle_pause, self.run_btn)
        control_sizer.Add(self.run_btn, 0, 0, 0)

        self.step_btn = wx.Button(frame_panel, label='Step')
        self.step_btn.Disable()
        self.Bind(wx.EVT_BUTTON, gui_app.on_step, self.step_btn)
        control_sizer.Add(self.step_btn, 0, 0, 0)

        control_sizer.SetSizeHints(self)
        vertical_sizer.Add(control_sizer, 0, 0, 0)
        
        self.verbose_checkbox = wx.CheckBox(frame_panel, label='Verbose')
        self.Bind(wx.EVT_CHECKBOX, gui_app.on_verbose, self.verbose_checkbox)
        vertical_sizer.Add(self.verbose_checkbox, 0, 0, 0)

        self.render_combobox = wx.ComboBox(frame_panel,
                                      choices=['All', 'ActionFrames', 'NoRender'])
        self.render_combobox.SetSelection(1)
        self.Bind(wx.EVT_COMBOBOX, gui_app.on_render_option, self.render_combobox)
        vertical_sizer.Add(self.render_combobox, 0, 0, 0)

        vertical_sizer.SetSizeHints(self)
        
        self.SetSizer(vertical_sizer)
        self.Show(True)

    def enable_stepping(self, enabled):
        if enabled:
            self.step_btn.Disable()
        else:
            self.step_btn.Enable()

    def is_running(self, running):
        if running:
            self.run_btn.SetLabel('Pause')
            self.enable_stepping(True)
        else:
            self.run_btn.SetLabel('Run')
            self.run_btn.Update()
            self.enable_stepping(False)
            
if __name__ == '__main__':
    app = MarioRLGUI(env,
                     q_estimator,
                     action_policy,
                     action_set,
                     learning_policy)
    app.MainLoop()
