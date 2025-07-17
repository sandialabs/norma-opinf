import abc
import numpy as np
from romtools.workflows.sampling import *
import sys

class NormaParameterSpace(ParameterSpace):

    @abc.abstractmethod
    def get_names(self):
        '''
        Returns a list of parameter names
        # e.g., ['sigma','beta',...]
        '''

    @abc.abstractmethod
    def get_dimensionality(self) -> int:
        '''
        Returns an integer for the size
        of the parameter domain
        '''

    @abc.abstractmethod
    def generate_samples(self, number_of_samples: int, seed=None) -> np.array:
        '''
        Generates samples from the parameter space

        Returns np.array of shape
        (number_of_samples, self.get_dimensionality())
        '''

    @abc.abstractmethod
    def update_norma_yaml(self,norma_yaml,parameter_sample):
        '''
        Updates the norma yaml file to reflect the values in parameter sample
        '''

