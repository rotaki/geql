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

    def reward(self, state, action, reward, state2, action2):
        """
        Notify the QEstimator of an observed SARSA'-tuple. 
        
        The QEstimator can choose to update itself immediately, or at a 
        later point (for instance when episode_finished() is called).

        The QEstimator is guaranteed that reward() will be called in the same
        order that states are observed

        Parameters
        ----------
        state : raw state
            State observed
        action : int
            Index of action performed in state
        reward : float
            Reward observed when performing action in state
        state2 : raw state
            State observed after performing action in state
        action2 : int
            Index of the action to be taken in state2 according to the learning
            algorithm. For example, for Q-learning, this will be argmax_a (Q(state2, a))
            and for SARSA, this will be Policy(state2)
         """
        raise NotImplementedError('reward not implemented')

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
