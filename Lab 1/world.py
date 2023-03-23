from gym import make
import numpy as np
from gym.wrappers.record_video import RecordVideo
from typing import NamedTuple


"""
Observation tuple to store the 4 different states of the cartpole environment
"""
class Observation(NamedTuple):
    d: float
    x: float
    theta: float
    omega: float

class CartpoleWorld():
    def __init__(self, display: bool = False) -> None:
        if display:
            self.__env = make("CartPole-v1", render_mode="human")
        else:
            self.__env = make("CartPole-v1")
        self.__observation: np.ndarray
        self.__reward: float = 0
        self.__truncated: bool  = False
        self.__done: bool = False
        self.__observation, _ = self.__env.reset()
    
    def get_actionspace(self):
        return self.__env.action_space
        
    def get_observation(self) -> np.ndarray:
        return self.__observation
    
    def update_world(self,action) -> float:
        self.__observation, self.__reward, self.__truncated, self.__done, _ = self.__env.step(action)
        return self.__reward
    
    def isEnd(self) -> bool:
        # position range
        if not (-2.4 < self.__observation[0] < 2.4):
            return True
        # angle range
        if not (-.2095 < self.__observation[2] < .2095):
            return True
        return self.__done or self.__truncated
    
    def get_reward(self) -> float:
        return self.__reward
    
    def resetWorld(self) -> Observation:
        self.__observation, _ = self.__env.reset()
        self.__reward = 0
        self.__done = False
        self.__truncated  = False
        return Observation(*self.__observation)
    
    def close_display(self) -> None:
        self.__env.close()
        self.__env = make("CartPole-v1")
        self.__observation, _ = self.__env.reset()
        
    def set_to_display_mode(self) -> None:
        self.__env = make("CartPole-v1", render_mode="human")
        self.__observation, _ = self.__env.reset()
        
    def set_save_video(self,filename="rl-video") -> None:
        self.__env = make("CartPole-v1", render_mode="rgb_array_list")
        self.__env = RecordVideo(self.__env , video_folder="video", name_prefix = filename)