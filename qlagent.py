import math
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from gym import make
import numpy as np
from gym.wrappers.record_video import RecordVideo
import pickle
from typing import NamedTuple


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

        self.env = self.__env
        
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
        
    def set_to_display_mode(self) -> None:
        self.__env = make("CartPole-v1", render_mode="human")
        self.__observation, _ = self.__env.reset()
        
    def set_save_video(self) -> None:
        self.__env  = make("CartPole-v1", render_mode="rgb_array_list")
        self.__env  = RecordVideo(self.__env , video_folder="video", name_prefix = "rl-video")

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
    
    
    def run_training(self, num_of_episode: int) -> None:
        pass
    
    def run_single_episode_training(self) -> int:
        pass

    @abstractmethod
    def run_production(self, num_of_episode: int) -> None:
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

# define constants
MULTIPLICATIVE_DECAY=0

class QLearningAgent(RLAgent):
    """Q-learning agent

    Args:
        RLAgent (class): Abstract base class
    """
    def __init__(self, env:CartpoleWorld, to_load_pickle=True, to_save_pickle=True, max_epoch=100000, epsilon_decay=MULTIPLICATIVE_DECAY) -> None:
        super().__init__(env)
        
        # defines learning rate
        self.__learning_rate = 0.5
        self.__learning_rate_min = 0.01
        
        # defined for epsilon soft policy
        # initially set to a large number to encourage exploration
        # epsilon will decay as episodes increase
        self.__epsilon_decay = epsilon_decay
        self.__num_epoch = 0
        self.__max_epoch = max_epoch
        self.__epsilon = 0.9
        self._epsilon_min = 0.01
        
        # dictionary of (state,action) -> quality
        self.__q_table : Dict[Tuple[Observation,int],float] = dict()
        self.__pi_table : Dict[Observation, int] = dict()
        
        # Load pickle file to restore agent previous state
        if (to_load_pickle):
            self.load_pickle('QL_parameters.pkl')
        self.__to_save_pickle = to_save_pickle
        
        # [left, right] action set
        self.__actions = [0,1]
        self.__discounted_reward = 0.9
        
        # parameter for production
        self.__is_production = False
        
    
    def update_parameters(self) -> None:
        self.decay_epsilon()
        self.decay_learning_rate()
    
    def decay_epsilon(self) -> None:
        if self.__epsilon <= self._epsilon_min:
            return
        
        self.__num_epoch +=1
        if (self.__epsilon_decay==MULTIPLICATIVE_DECAY):
            # multiplicative decrease
            self.__epsilon *= 0.999999
        else:
            # exponential decrease
            # hyperparamters to tune
            A=0.5
            B=0.1
            C=0.1
            standardized_time=(self.__num_epoch-A*self.__max_epoch)/(B*self.__max_epoch)
            cosh=np.cosh(math.exp(-standardized_time))
            self.__epsilon=1.1-(1/cosh+(self.__num_epoch*C/self.__max_epoch))
    
    def decay_learning_rate(self) -> None:
        if self.__learning_rate <= self.__learning_rate_min:
            return
        self.__learning_rate *= 0.999999
    
    def run_training(self,  num_of_episode: int):
        """Overrides base class method

        Args:
            num_of_episode (int): Number of episode to run
        """
        self.__is_production = False
        cumulated_reward = 0
        for _ in range(num_of_episode):
            cumulated_reward += self.run_single_episode_training()
        
        print(f"Epsilon: {self.__epsilon}, Discounted reward: {self.__discounted_reward}, Learning rate: {self.__learning_rate}")
        print(f"Mean reward is: {cumulated_reward/num_of_episode} for {num_of_episode} episodes")
        if (self.__to_save_pickle):
            self.save_pickle("QL_parameters.pkl")
    
    def run_training_for_plot(self,  num_of_episode: int) -> float:
        cumulated_reward = 0
        for _ in range(num_of_episode):
            cumulated_reward += self.run_single_episode_training()
        return cumulated_reward/num_of_episode
    
    def run_production(self, num_of_episode: int):
        self.__is_production = True
        
        cumulated_reward = 0
        for _ in range(num_of_episode):
            cumulated_reward += self.run_single_episode_production()

        print(f"Mean reward is: {cumulated_reward/num_of_episode} for {num_of_episode} episodes")

    def run_single_episode_training(self) -> int:
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
            
        self.update_parameters()
        return self._total_reward
    
    def run_single_episode_production(self) -> int:
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
        return self._total_reward

    def get_optimal_action(self, s: Observation) -> int:
        """Gets optimal action for given state

        Args:
            s (Observation): State observed

        Returns:
            int: action to take, subjected to epsilon soft policy
        """
        # a* is the policy from the pi table
        if (self.__is_production):
            a_star: int = self.get_policy(s)
            return a_star
        
        epsilon_over_A: float = self.__epsilon / len(self.__actions)
        
        # apply epsilon soft policy here to encourage exploration
        if (np.random.randn() < 1 - self.__epsilon + epsilon_over_A):
            # pick optimal
            self.__pi_table[s] = self.argmax_a_Q(s,self.__actions)
        else:
            # pick random
            self.__pi_table[s] = self.get_random_action()
        return self.__pi_table[s]
    
    def update_q_table(self,s: Observation, R: float, s_prime: Observation):
        Q_S_A = self.Q(s,self.__pi_table[s])
        Q_S_A = Q_S_A + self.__learning_rate * \
                (R + self.__discounted_reward*self.max_Q(s_prime,self.__actions) - Q_S_A)
        
        self.__q_table[(s,self.__pi_table[s])] = Q_S_A

    def Q(self, state: Observation, action: int) -> float:
        if ((state,action) in self.__q_table):
            return self.__q_table[(state,action)]
        else:
            self.__q_table[(state,action)] = 0
            return 0
    
    def max_Q(self, state: Observation, action_set: List[int]) -> float:
        """Gets the max value of Q over all actions

        Args:
            state (Observation): state observed
            action_set (List[int]): list of possible actions

        Returns:
            float: max value of Q for state
        """
        return max([self.Q(state,action) for action in action_set])

    def argmax_a_Q(self, state: Observation, action_set: List[int]) -> int:
        """Returns action that maximises Q function

        Args:
            state (Observation): state observed
            action_set (List[int]): list of actions possible

        Returns:
            int: action
        """
        return max([(action,self.Q(state,action)) for action in action_set],key=lambda item:item[1])[0]
    
    def get_policy(self, state: Observation):
        if (state not in self.__pi_table):
            self.__pi_table[state] = self.get_random_action()
            for a in self.__actions: self.__q_table[(state,a)] = 0
        
        return self.__pi_table[state]
    
    def get_random_action(self) -> int:
        """Randomly generates an action.
        
        Returns:
            int: Action taken
        """
        val = np.random.rand()
        if (float(val) % 1) >= 0.5:
            return math.ceil(val)
        else:
            return round(val)
 
    def get_q_table(self):
        return self.__q_table
    
    def get_pi_table(self):
        return self.__pi_table
        
    def load_pickle(self, parameters_file: str):
        """Loads pickle file to agent table

        Args:
            parameters (str): pickle file location
        """
        if os.path.exists(parameters_file):
            with open(parameters_file, 'rb') as file:
                # Call load method to deserialze
                self.__pi_table,self.__q_table,self.__epsilon,self.__learning_rate = pickle.load(file)
        else:
            print("*** LOG: Pickle file not found")
            pass
        
    def save_pickle(self, parameters_file: str):
        """Saves q and pi table to pickle.

        Args:
            pi_table_file (str): location of file
            q_table_file (str): location of file
        """
        with open(parameters_file, 'wb') as file:
            # Call load method to deserialze
            pickle.dump([self.__pi_table,self.__q_table,self.__epsilon,self.__learning_rate], file)


 
import os
if __name__ == "__main__":
    world = CartpoleWorld()
    agent = QLearningAgent(world,to_load_pickle=False,to_save_pickle=False,epsilon_decay=1)
    # print(sorted(agent.get_q_table().items(), key = lambda x : x[1]))
    # world.set_display_mode()
    # for i in range(100):
    #     agent.run_production(1000)
    print(len(agent.get_q_table()))
    agent.run_production(100)
    print(len(agent.get_q_table()))