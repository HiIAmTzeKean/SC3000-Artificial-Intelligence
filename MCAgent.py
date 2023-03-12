import os
import pickle
import numpy as np
from qlagent import CartpoleWorld, RLAgent, Observation
from collections import defaultdict
import random

class MCAgent(RLAgent):
    def __init__(self, env: CartpoleWorld, load_pickle = False) -> None:
        #super().__init__(env)
        #for convenience
        self.env = env

        self.total_reward = 0
        self.max_reward = 0
        self.num_runs = 0

        self.actions = [0,1]

        self.epsilon = 0.9
        self.min_epsilon = 0.1
        self.gamma = 0.99

        # Format: dict[(state,action)] = [value]
        self.Q = defaultdict(int)
        self.returns = defaultdict(int)
        self.visits = defaultdict(int)
        self.history = []

        if load_pickle:
            self.load_pickle("MC_parameters.pkl")

    # Not Used
    def geometric_progression(self, n: int) -> float:
        # Calcuate rewards based on the formula S = a(1 - r^n) / (1-r)
        return (1 - (self.gamma ** n)) / (1 - self.gamma)


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * 0.99999, self.min_epsilon)
    

    def run_single_episode_training(self) -> int:
        reward = self.run_single_episode_production()
        self.update_q_table()
        return reward
    

    def run_single_episode_production(self) -> int:
        self.env.resetWorld()
        history = []
        total_reward = 0

        while (not self.env.isEnd()):
            ndarry: np.ndarray = self.env.get_observation()
            s = Observation(*self.discretise_observation(ndarry))

            action = self.get_optimal_action(s)
            reward = self.env.update_world(action)
            history.append((s, action, reward))
            total_reward += reward

        self.decay_epsilon()
        self.num_runs += 1
        self.history = history
        self.max_reward = max(self.max_reward, total_reward)
        self.total_reward += total_reward    
        return total_reward
    
    # Not Used
    def run_production(self, num_of_episode: int) -> None:
        return super().run_production(num_of_episode)
    
    # Not Used
    def run_single_episode(self) -> int:
        # clear history
        self.env.resetWorld()
        self.total_reward = 0

        history = []

        while (not self.env.isEnd()):
            ndarry: np.ndarray = self.env.get_observation()
            s = Observation(*self.discretise_observation(ndarry))

            action = self.get_optimal_action(s)
            reward = self.env.update_world(action)
            history.append((s, action, reward))
            # update reward
            self.total_reward += reward

        self.update_q_table(history)
        self.max_reward = max(self.max_reward, self.total_reward)
        return self.total_reward
    

    def update_q_table(self) -> None:
        history = self.history
        state_action_pair = defaultdict(int)
        G = 0

        # Adopts first visit by updating from the back
        # Hence, only the first occurence of each (state,action) pair will be recorded
        for i in range(len(history)-1, -1, -1):
            state, action, reward = history[i]
            G = self.gamma * G + reward
            state_action_pair[(state, action)] = G

        for key,val in state_action_pair.items():
            self.returns[key] += val
            self.visits[key] += 1
            self.Q[key] = self.returns[key] / self.visits[key]


    def get_optimal_action(self, s: Observation):
        
        # self.epsilon starts at 0.9, slowly decreases till a minimum of 0.1
        if random.random() <= self.epsilon:
            action = 1 if s[2] > 0 else 0
            # Chooses the physics action which a probability of 0.6, and the other action with a prob of 0.4
            return action if random.random() <= 0.70 else 1 - action
        
        return max(self.actions, key=lambda a: self.Q[(s,a)])
    
    # Not Used
    def run(self, num_episodes: int, display: bool = False) -> None:
        cumulated_reward = 0

        # Pre-training, maybe can replace with pre-load data here in the future
        # Run 1000 times first using the physics solution to populate values
        self.exploit_rate = 0.0
        for _ in range(1000):    
            self.run_single_episode()


        outer_range, inner_range = 10, 5000
        self.exploit_rate = 0.7

        for _ in range(outer_range):
            total = 0
            max_reward = 0
            for _ in range(inner_range):
                # Using a variable exploitation rate, dk if its a good idea
                #self.exploit_rate = i / inner_range
                reward = self.run_single_episode()
                total += reward
                max_reward = max(max_reward, reward)
                

            print(f"Mean reward is: {total / inner_range}")
            print(f"Max reward is: {max_reward}")

        # Actual run
        self.actual = True

        if display:
            self.env.set_to_display_mode()

        for _ in range(num_episodes):
            current_reward = self.run_single_episode()
            cumulated_reward += current_reward


    def load_pickle(self, parameters_file: str):
        """Loads pickle file to agent table

        Args:
            parameters (str): pickle file location
        """
        if os.path.exists(parameters_file):
            with open(parameters_file, 'rb') as file:
                # Call load method to deserialze
                Q, returns, visits, self.epsilon = pickle.load(file)
                self.Q = defaultdict(int, Q)
                self.returns = defaultdict(int, returns)
                self.visits = defaultdict(int, visits)
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
            pickle.dump([dict(self.Q), dict(self.returns), dict(self.visits), self.epsilon], file)

    def print_stats(self):
        print(f"avg_reward:{self.total_reward/self.num_runs}")


load_and_save = False

if __name__=="__main__":
    world = CartpoleWorld()
    agent = MCAgent(world, load_and_save)
    #world.set_to_display_mode()

    # training
    # for i in range(100000):
    #     agent.run_single_episode_training()

    #     if i % 1000 == 0:
    #         agent.print_stats()

    # actual
    world.set_to_display_mode()
    for _ in range(10):
        reward = agent.run_single_episode_production()
        print("current run:", reward)
        agent.print_stats()

    print("check epsilon value", agent.epsilon)
    print("check max Q(s,a) values:", max(agent.Q.values()))
    print("check max visits count:", max(agent.visits.values()))
    print("check max returns count:", max(agent.returns.values()))

    if load_and_save:
        agent.save_pickle("MC_parameters.pkl")
    