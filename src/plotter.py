from typing import List, Optional
import numpy as np

class Plotter:
    """
    Contains methods for visualising solver outputs.
    Does not contain solver logic.
    """
    
    @staticmethod
    def plot_history_3D(gamma: List[np.ndarray]):
        """
        Plots the full circulation or vortex strength history as a 3D visualisation.
        
        :param gamma: The circulation dimension (k x max_vortices x 2)
        """
        pass

    @staticmethod
    def plot_history_2D(gamma: List[np.ndarray], num_snapshots: int):
        """
        Plots selected 2D snapshots of the circulation or vortex strength distribution over time.

        :param gamma: The circulation dimension (k x max_vortices x 2)
        :param num_snapshots: Number of snapshots to select across the time history
        """
        pass

    @staticmethod
    def plot_lift_history(lift_history: List[float]):
        """
        Plots lift as a function of time.
        
        :param lift_history: Lift evaluated at each time step
        """
        pass

    @staticmethod
    def plot_lift_frequency_spectrum(lift_history: List[float]):
        """
        Performs a frequency-domain visualisation of the lift history (e.g. FFT).
        
        :param lift_history: Lift evaluated at each time step
        """
        pass
