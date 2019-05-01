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

    def reward(self, state, action, reward, result_state):
        """
        Notify the QEstimator of an observed SARS'-tuple. 
        
        The QEstimator can choose to update itself immediately, or at a 
        later point (for instance when episode_finished() is called).

        Parameters
        ----------
        state : raw state
            State observed
        action : int
            Index of action performed in state
        reward : float
            Reward observed when performing action in state
        result_state : raw state
            State observed after performing action in state
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
    def action(self, state, q_estimator):
        """
        Decides what action to take in a given state (\pi(s))

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
        raise NotImplementedError('action not implemented')

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
