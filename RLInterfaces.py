class IQEstimator:
    """
    Interface for class for estimating and revising Q(s, a) for reinforcment 
    learning problems
    """
    def estimate(self, state, action):
        """
        Q(state, action)

        Parameters
        ----------
        state : raw state 
            State to evaluate action for
        action : int
            Index of action to evaluate

        Returns
        -------
        float
            Estimated Q(state, action)
        """
        raise NotImplementedError('estimate not implemented')

    def batch_estimate(self, state, actions):
        """
        Compute Q(state, action) for multiple actions

        If state-transformation can be re-used for several actions, it might be
        faster to compute Q-values for all actions at once, rather than
        through multiple calls to estimate().

        Parameters
        ----------
        state : raw state
            State to evaluate action for
        action : [int]
            List of action indexes to evaluate

        Returns
        -------
        [(int, float)]
            List of (action, Q(s, a)) for all actions-list passed as an argument.
            No guarantee is given that the list is in the same order as the 
            actions-list passed as argument
        """
        raise NotImplementedError('batch_estimate not implemented')

    def episode_start(self, initial_state):
        """
        Notifies the QEstimator that a new trajectory (episode) will start.
        All calls to record_transition() between this call and episode_finished()
        will be a complete sequence of transitions belonging to the same trajectory

        Parameters
        ----------
        initial_state : raw state
            The first state in the trajectory
        """
        raise NotImplementedError()
    
    def record_transition(self, action, reward, state, terminal, lp_action):
        """
        Notifies the QEstimator of a transition belonging to the current
        trajectory. The "from"-state is implicitly the original state of
        the previous record_transition() (or the initial state, if no
        calls have been made to record_transition yet for this trajectory)

        Parameters
        ----------
        action : int
            Index of action performed
        reward : float
            Reward awarded for taking action from the last state
        state : raw state
            The resulting state of taking action in the previous state
        terminal : boolean
            True iff 'state' is a terminal state (in which case, 
            episode_finished will also be called later)
        lp_action : int | None
            The action that should be taken in 'state' according to the current 
            learning policy (next action for SARSA or None for Q-learning)
        """
        raise NotImplementedError()
    
    def episode_finished(self):
        """
        Notify the QEstimator that the current episode has finished.

        Can be used to update the QEstimator in the case of batch learning
        """
        pass

    def save(self, filename):
        """
        Saves the QEstimator to filename

        The QEstimator can then be restored by calling from_file(filename)
        
        Parameters
        ----------
        filename : string
            Filename to save to
        """
        raise NotImplementedError('save not implemented')

    def from_file(filename):
        """
        Loads a previously saved QEstimator from file

        Parameters
        ----------
        filename : string
            Filename to load from

        Returns
        -------
        IQEstimator
            The loaded QEstimator
        """
        raise NotImplementedError('from_file not implemented')

    def summary(self):
        """
        A string summary (may contain LaTeX) used to describe the QEstimator and
        its set of parameters
        
        Returns
        string
            The description string
        """
        return 'Implement IQEstimator.summary()!'


class IActionPolicy:
    """
    Interface for class to determine what action to take in a particular state
    (corresponding to \pi(s) in a reinforcement learning problem)
    """ 

    def get_action(self, state, q_estimator):
        """
        Gets the action that should be taken according to the ActionPolicy.

        ActionPolicy *may* mutate as a result of calling get_action

        Parameters
        ----------
        state : raw state
            The state to decide for
        q_estimator : IQEstimator
            QEstimator used to evaluate Q(state, a) for some actions a
            (usually the same QEstimator that is being used in the RL problem)

        Returns
        int
            Index of the action to be taken
        """
        raise NotImplementedError('get_action not implemented')
    
    def episode_finished(self):
        """
        Notifies the ActionPolicy that the current episode is finished

        The ActionPolicy may do some householding work, or simply ignore this
        """
        pass
    
    def save(self, filename):
        """
        Saves the ActionPolicy to filename

        The ActionPolicy can then be restored by calling from_file(filename)
        
        Parameters
        ----------
        filename : string
            Filename to save to
        """

        raise NotImplementedError('save not implemented')

    def from_file(filename):
        """
        Loads a previously saved ActionPolicy from file

        Parameters
        ----------
        filename : string
            Filename to load from

        Returns
        -------
        IActionPolicy
            The loaded ActionPolicy
        """

        raise NotImplementedError('from_file not implemented')
    
    def summary(self):
        """
        A string summary (may contain LaTeX) used to describe the ActionPolicy 
        and its set of parameters
        
        Returns
        string
            The description string
        """

        return 'Implement IActionPolicy.summary()!'

# Dont know yet if we will use this
class ITerminationPolicy:
    def new_episode(self):
        raise NotImplementedError('new_episode not implemented')

    def update_and_decide(self):
        raise NotImplementedError('update_and_decide not implemented')
