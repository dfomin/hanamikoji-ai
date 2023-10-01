import random
from typing import Optional

from gymnasium import Env, spaces
from hanamikoji.game import Game
from hanamikoji.observation import Observation
from hanamikoji.player import RandomPlayer
from hanamikoji.state import State


class DummyEnv(Env):
    game: Optional[State]

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(4)

        self.state = None
        self.player_index = random.randint(0, 1)
        self.internal_player = RandomPlayer()

    def step(self, action):
        # decode action
        action_index = 0
        action = Game.get_available_actions(self.state)[action_index]
        Game.apply_action(self.state, action)
        if self.state.is_finished():
            return (self.state.observation(),
                    1 if self.player_index == self.state.winner() else -1,
                    True,
                    False,
                    {})

        action_index = self.internal_player.choose_action(self.state.observation(),
                                                          Game.get_available_actions(self.state))
        action = Game.get_available_actions(self.state)[action_index]
        Game.apply_action(self.state, action)
        if self.state.is_finished():
            return (self.state.observation(),
                    1 if self.player_index == self.state.winner() else -1,
                    True,
                    False,
                    {})

        return self._encode_obs(self.state.observation()), 0, False, False, {}

    def reset(self, *args, **kwargs):
        self.state = State()
        return self._encode_obs(self.state.observation()), {}

    def _encode_obs(self, obs: Observation):
        return [x for x in self.state.actions[obs.current_player]]
