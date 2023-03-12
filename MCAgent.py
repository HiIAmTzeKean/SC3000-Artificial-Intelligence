import numpy as np
from qlagent import CartpoleWorld, Observation
from collections import defaultdict
import random
from typing import Dict, List, Tuple

class MCAgent():
    def __init__(self, env: CartpoleWorld) -> None:
        self.env = env
        self.total_reward = 0
        self.actions = [0,1]
        self.exploit_rate = 0
        self.gamma = 0.9
        self.actual = False

        # Format: dict[(state,action)] = [count, score]
        self.Q = defaultdict(lambda: [0, 0])

    def discretise_observation(self, observation: np.ndarray) -> Observation:
        # Position round off to 0.1 precision
        # Velocity round off to whole
        # Angle round off to 0.01 precision
        # Velocity round off to whole
        observation[0] = round(observation[0], 1)
        observation[1] = round(observation[1], 0)
        observation[2] = round(observation[2], 2)
        observation[3] = round(observation[3], 0)
        return Observation(*observation)
    
    def geometric_progression(self, n: int) -> float:
        # Calcuate rewards based on the formula S = a(1 - r^n) / 1 - r
        return (1 - (self.gamma ** n)) / (1 - self.gamma)


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
            history.append((s, action))
            # update reward
            self.total_reward += reward

        self.update_q_table(history)
        self.max_reward = max(self.max_reward, self.total_reward)
        return self.total_reward
    

    def update_q_table(self, history: List[Tuple[Observation, int]]) -> None:
        state_action_pair = defaultdict(int)
        # Adopts first visit by updating from the back
        # Hence, only the first occurence of each (state,action) pair will be recorded
        # n is the episode length starting from the back, which is also the reward 
        for n,val in enumerate(history[::-1], start=1):
            state_action_pair[val] = n

        for key,val in state_action_pair.items():
            count, score = self.Q[key]
            score = (count * score + val) / (count + 1)
            self.Q[key] = [count+1, score]


    def get_optimal_action(self, s: Observation):
        
        #Slowly increase the exploitation rate, such that the agent will choose the best action as time progresses
        if not self.actual and random.random() >= self.exploit_rate:
            #return random.randint(0, 1)
            
            action = 1 if s[2] > 0 else 0
            # Chooses the physics action which a probability of 0.6, and the other action with a prob of 0.4
            return action if random.random() <= 0.60 else 1 - action
        
        return max(self.actions, key=lambda a: self.Q[(s,a)])
    

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


if __name__=="__main__":
    world = CartpoleWorld()
    agent = MCAgent(world)
    #world.set_to_display_mode()

    agent.run(3, True)
    