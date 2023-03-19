from QLAgent import RLAgent, CartpoleWorld, Observation

class PhysicsAgent(RLAgent):
    
    def __init__(self, env:CartpoleWorld) -> None:
        super().__init__(env)

    def run_production(self, num_of_episode: int) -> None:
        cumulated_reward = 0
        for _ in range(num_of_episode):
            current_reward = self.run_single_episode_production()
            cumulated_reward += current_reward
        print(f"Mean reward is: {cumulated_reward/num_of_episode}")

    def run_single_episode_production(self) -> int:
        # clear history
        self._env.resetWorld()
        self._total_reward = 0
        while (not self._env.isEnd()):
            ndarry: Observation = self._env.get_observation()
            s = self.wrap_observation(ndarry)
            self.move(s)
        return self._total_reward
    
    
    def get_optimal_action(self, s: Observation) -> int:
        """Reference: https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
            Overrides abstract base class
        """
        theta, w = s[2:4]
        if abs(theta) < 0.03:
            return 0 if w < 0 else 1
        else:
            return 0 if theta < 0 else 1

if __name__=="__main__":
    world = CartpoleWorld()
    agent = PhysicsAgent(world)
    world.set_to_display_mode()
    agent.run_production(1)
