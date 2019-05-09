from collections import namedtuple
from impl.EpsilonGreedyActionPolicy import EpsilonGreedyActionPolicy
from RLInterfaces import IQEstimator
import numpy as np

Transition = namedtuple('Transition', [
    'action',    # action taken in previous state
    'reward',    # reward associated with taking A in S
    'state',     # resulting state
    'terminal',  # True iff action resulted in terminal state
    'lp_action'  # action selected in state by learning policy (or None)
 ])

QSample = namedtuple('QSample', [
    'state',
    'action',
    'q',
    'importance'
])

class Trajectory:
    def __init__(self, initial_state, is_terminal=False):
        self.transitions = [Transition(None, None, initial_state, is_terminal, None)]
        
    def add_transition(self, action, reward, state, terminal, lp_action):
        self.transitions.append(Transition(action, reward, state, terminal, lp_action))

    def sarsa_backup(self, entry, discount, steps, q_estimator):
        entry += 1
        discounted_reward = 0
        reached_terminal_state = False
        for forward_step in range(steps):
            if entry + forward_step >= len(self.transitions):
                raise RuntimeError('Trajectory does not contain the required number of transitions')
            forward_entry = self.transitions[entry + forward_step]
            discounted_reward += pow(discount, forward_step) * forward_entry.reward
            if forward_entry.terminal:
                reached_terminal_state = True
                break
            
        if not reached_terminal_state:
            # Include the Q(s, a) of the last step on the horizon
            horizon_entry = self.transitions[entry + steps - 1]
            discounted_reward += pow(discount, steps) * \
                q_estimator.estimate(horizon_entry.state, horizon_entry.lp_action)

        return QSample(self.transitions[entry - 1].state,
                       self.transitions[entry].action,
                       discounted_reward,
                       1.0) # Importance always 1.0 for SARSA
            
    def sarsa_episode_backup(self, discount, steps, q_estimator):
        return [self.sarsa_backup(i, discount, steps, q_estimator)
                for i in range(len(self.transitions) - 1)]

    def q_backup(self, entry, discount, steps, q_estimator, q_action_policy):
        if steps != 1:
            # Check Sutton chapter 7.3 and 7.5 for possible implementation
            raise NotImplementedError('Not yet implemented')
        entry += 1
        discounted_reward = self.transitions[entry].reward
        if not self.transitions[entry].terminal:
            q_action = q_action_policy.get_action(self.transitions[entry].state, q_estimator)
            discounted_reward += discount * q_estimator.estimate(self.transitions[entry].state, q_action)
        return QSample(self.transitions[entry - 1].state,
                       self.transitions[entry].action,
                       discounted_reward,
                       1.0) # If we implement n-step off policy, importance factor will need to change
    
    def q_episode_backup(self, discount, steps, q_estimator, q_action_policy):
        return [self.q_backup(i, discount, steps, q_estimator, q_action_policy)
                for i in range(len(self.transitions) - 1)]
            
### Unit tests below

import unittest
class MockEstimator(IQEstimator):
    def __init__(self):
        self.entries = dict()
        
    def estimate(self, state, action):
        state_key = state.tobytes()
        return self.entries.get((state_key, action), 0.0)

    def batch_estimate(self, state, actions):
        return list(map(lambda a: (a, self.estimate(state, a)), actions))

    def assign(self, state, action, value):
        state_key = state.tobytes()
        self.entries[(state_key, action)] = value
        
class TestBackupTrajectory(unittest.TestCase):
    def setUp(self):
        states = [np.random.rand(240, 256, 3) for _ in range(10)]
        self.trajectory_data = [
            # S0        A1  R2  S'3        A'4
            (states[0], 0,  3,  states[1], 1),
            (states[1], 1,  2,  states[2], 2),
            (states[2], 2, -1,  states[3], 0),
            (states[3], 0, -9,  states[4], 0),
            (states[4], 0,  5,  states[5], 1),
            (states[5], 1, 50,  states[6], None)
        ]

        self.estimator = MockEstimator()
        self.estimator.assign(states[0], 0, 1)
        self.estimator.assign(states[0], 1, 3)
        self.estimator.assign(states[0], 2, 5)
        
        self.estimator.assign(states[1], 0, 7)
        self.estimator.assign(states[1], 1, 11)
        self.estimator.assign(states[1], 2, 13)
        
        self.estimator.assign(states[2], 0, 17)
        self.estimator.assign(states[2], 1, 19)
        self.estimator.assign(states[2], 2, 23)

        self.estimator.assign(states[3], 0, 29)
        self.estimator.assign(states[3], 1, 31)
        self.estimator.assign(states[3], 2, 37)

        self.estimator.assign(states[4], 0, 41)
        self.estimator.assign(states[4], 1, 43)
        self.estimator.assign(states[4], 2, 47)

        self.estimator.assign(states[5], 0, 54)
        self.estimator.assign(states[5], 1, 59)
        self.estimator.assign(states[5], 2, 61)


    def test_sarsa_1step(self):
        trajectory = Trajectory(self.trajectory_data[0][0])
        for t in self.trajectory_data:
            trajectory.add_transition(t[1], t[2], t[3], t[4] == None, t[4])
        tl = trajectory.sarsa_episode_backup(0.8, 1, self.estimator)

        # Manually calculated values
        self.assertAlmostEqual(tl[0].q, 11.8)
        self.assertAlmostEqual(tl[1].q, 20.4)
        self.assertAlmostEqual(tl[2].q, 22.2)
        self.assertAlmostEqual(tl[3].q, 23.8)
        self.assertAlmostEqual(tl[4].q, 52.2)
        self.assertAlmostEqual(tl[5].q, 50.0)

        # Check we're getting the correct array as "from"-state
        self.assertTrue(np.array_equal(tl[1].state, self.trajectory_data[0][3]))
        # ...and correct action
        self.assertEqual(tl[1].action, 1)
        self.assertEqual(tl[1].importance, 1.0)
        
    def test_sarsa_3step(self):
        trajectory = Trajectory(self.trajectory_data[0][0])
        for t in self.trajectory_data:
            trajectory.add_transition(t[1], t[2], t[3], t[4] == None, t[4])

        tl = trajectory.sarsa_episode_backup(0.8, 3, self.estimator)

        # Manually calculated values
        self.assertAlmostEqual(tl[0].q, 18.808)
        self.assertAlmostEqual(tl[1].q, 16.432)
        self.assertAlmostEqual(tl[2].q, 25.208)
        self.assertAlmostEqual(tl[3].q, 27)
        self.assertAlmostEqual(tl[4].q, 45)
        self.assertAlmostEqual(tl[5].q, 50)


    def test_q_1step(self):
        trajectory = Trajectory(self.trajectory_data[0][0])
        for t in self.trajectory_data:
            trajectory.add_transition(t[1], t[2], t[3], t[4] == None, t[4])
        greedy = EpsilonGreedyActionPolicy([0, 1, 2], 0.0)
        
        tl = trajectory.q_episode_backup(0.8, 1, self.estimator, greedy)
        self.assertAlmostEqual(tl[0].q, 13.4)
        self.assertAlmostEqual(tl[1].q, 20.4)
        self.assertAlmostEqual(tl[2].q, 28.6)
        self.assertAlmostEqual(tl[3].q, 28.6)
        self.assertAlmostEqual(tl[4].q, 53.8)
        self.assertAlmostEqual(tl[5].q, 50)
    
if __name__ == '__main__':
    unittest.main()
