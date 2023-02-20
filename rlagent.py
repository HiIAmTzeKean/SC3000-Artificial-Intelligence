import math
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from gym import make
import numpy as np
from gym.wrappers.record_video import RecordVideo

class Observation():
    def __init__(self, observation: np.ndarray) -> None:
        self._state = tuple(observation)
    def get_state(self):
        return self._state
    def __hash__(self) -> int:
        return sum([hash(i+j) for i,j in enumerate(self._state)])
    def __eq__(self, __o: object) -> bool:
        return self._state == __o.get_state()

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
    def get_reward(self):
        return self.__reward
    def resetWorld(self):
        self.__observation, _ = self.__env.reset()
        self.__reward = 0
        self.__done = False
        self.__truncated  = False
    def set_display_mode(self):
        self.__env = make("CartPole-v1", render_mode="human")
        self.__observation, _ = self.__env.reset()
    def set_save_video(self):
        self.__env  = make("CartPole-v1", render_mode="rgb_array_list")
        self.__env  = RecordVideo(self.__env , video_folder="video", name_prefix = "rl-video")

class RLAgent(ABC):
    def __init__(self, env:CartpoleWorld) -> None:
        self._env = env
        self._total_reward: float = 0
        
    @abstractmethod
    def get_optimal_action(self, s: Observation):
        pass
    def move(self, state: Observation) -> float:
        if (self._env.isEnd()):
            raise Exception("Episode already terminated")
        action = self.get_optimal_action(state)
        reward = self._env.update_world(action)
        # update reward
        self._total_reward += reward
        return reward
    def discretise_observation(self, observation: np.ndarray) -> Observation:
        # Posititon round off to 0.01 precision
        # Velocity round off to whole
        # Angle round off to 0.001 precision
        # Velocity round off to whole
        observation[0] = round(observation[0],1)
        observation[1] = round(observation[1],0)
        observation[2] = round(observation[2],2)
        observation[3] = round(observation[3],0)
        return Observation(observation)

class QLearningAgent(RLAgent):
    def __init__(self, env:CartpoleWorld) -> None:
        super().__init__(env)
        self.__learning_rate = 0.6
        # defined for epsilon soft policy
        self.__epsilon = 0.1
        # dictionary of (state,action) -> quality
        self.__q_table : Dict[Tuple[Observation,int],float] = dict()
        self.__pi_table : Dict[Observation, int] = dict()
        # [left, right] action set
        self.__actions = [0,1]
        self.__discounted_reward = 0.9
    
    def get_optimal_action(self, s: Observation):
        # a* is the argmax_a Q(s,a)
        a_star: int = self.argmax_a_Q(s,self.__actions)
        epsilon_over_A: float = self.__epsilon / len(self.__actions)
        
        # apply epsilon soft policy here to encourage exploration
        if (np.random.randn() < 1 - self.__epsilon + epsilon_over_A):
            # pick optimal
            self.__pi_table[s] = a_star
        else:
            # pick random
            self.__pi_table[s] = self.get_random_action()
        return self.__pi_table[s]
    
    def run(self,  num_of_episode: int):
        cumulated_reward = 0
        for i in range(num_of_episode):
            cumulated_reward += self.run_single_episode()
        print(f"Mean reward is: {cumulated_reward/num_of_episode}")
    def run_single_episode(self):
        # clear history
        self._env.resetWorld()
        self._total_reward = 0
        
        s_prime = self._env.get_observation()
        s_prime = self.discretise_observation(s_prime)
        
        while (not self._env.isEnd()):
            s = s_prime
            R = self.move(s)
            s_prime = self._env.get_observation()
            s_prime = self.discretise_observation(s_prime)
            
            self.update_q_table(s,R,s_prime)
        print(f"Episode completed: reward {self._total_reward}")
        return self._total_reward

    def update_q_table(self,s: Observation, R: float, s_prime: Observation):
        Q_S_A = self.__q_table[(s,self.__pi_table[s])]
        Q_S_A = Q_S_A + self.__learning_rate * \
                (R + self.__discounted_reward*self.Q(s_prime, self.argmax_a_Q(s_prime,self.__actions)) - Q_S_A)
        
        self.__q_table[(s,self.__pi_table[s])] = Q_S_A

    def Q(self, state: Observation, action: int) -> float:
        if ((state,action) in self.__q_table):
            return self.__q_table[(state,action)]
        else:
            self.__q_table[(state,action)] = 0
            return 0
    
    def argmax_a_Q(self, state: Observation, action_set: List[int]) -> int:
        return max([(action,self.Q(state,action)) for action in action_set],key=lambda item:item[1])[0]
        
    def get_random_action(self) -> int:
        val = np.random.rand()
        if (float(val) % 1) >= 0.5:
            return math.ceil(val)
        else:
            return round(val)
    
    def print_q_table(self):
        print(self.__q_table)
    def get_q_table(self):
        return self.__q_table
    

world = CartpoleWorld()
agent = QLearningAgent(world)
agent.run(50)
world.set_save_video()
agent.run_single_episode()
a= agent.get_q_table()
print(len(a))