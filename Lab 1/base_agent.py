import math
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from world import Observation,CartpoleWorld

class RLAgent(ABC):
    def __init__(self, env:CartpoleWorld) -> None:
        self._env = env
        self._total_reward: float = 0
        
    @abstractmethod
    def get_optimal_action(self, s: Observation) -> int:
        pass
    
    def move(self, state: Observation) -> float:
        if (self._env.isEnd()):
            raise Exception("Episode already terminated")
        action = self.get_optimal_action(state)
        reward = self._env.update_world(action)
        # update reward
        self._total_reward += reward
        return reward
    
    def run_single_episode_training(self) -> int:
        pass
    
    @abstractmethod
    def run_single_episode_production(self) -> int:
        pass
    
    def wrap_observation(self, observation: np.ndarray) -> Observation:
        """Converts numpy array to Observation object

        Args:
            observation (np.ndarray): array to pass in from cartpole

        Returns:
            Observation: Object to return
        """
        return Observation(*observation)
    
    def discretise_observation(self, observation: np.ndarray) -> Observation:
        # Position round off to 0.1 precision
        # Velocity round off to whole
        # Angle round off to 0.01 precision
        # Velocity round off to whole
        observation[0] = round(observation[0],1)
        observation[1] = round(observation[1],0)
        observation[2] = round(observation[2],2)
        observation[3] = round(observation[3],0)
        return Observation(*observation)