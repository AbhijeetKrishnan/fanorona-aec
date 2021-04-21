from gym import spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from .move import FanoronaMove
from .state import FanoronaState


def env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    Description:
        Implements the Fanorona board game following the 5x9 Fanoron Tsivy
        variation. A draw is declared if 50 half-moves have been exceeded 
        since the start of the game. Consecutive captures count as one move.

    References: 
        https://www.mindsports.nl/index.php/the-pit/528-fanorona
        https://en.wikipedia.org/wiki/Fanorona
    
    Reward:
        +1: win
         0: draw
        -1: loss, illegal move

    Starting State:
        Starting board setup for Fanorona (see https://en.wikipedia.org/wiki/Fanorona#/media/File:Fanorona-1.svg)

    Episode Termination:
        Game ends in a win, draw, loss or illegal move
    """

    metadata = {"render.modes": ["human", "svg"], "name": "fanorona_v0"}

    def __init__(self):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # The action space is a (5x9x8x3+1)-dimensional array. Each of the 5x9 positions identifies
        # the square from which to "pick up" the piece. 8 planes encode the possible directions
        # along which the piece will be moved (SW, S, SE, W, E, NW, N, NE). 3 planes encode the
        # capture type of the move (paika, approach, withdrawal). The last action denotes a manual
        # end turn.
        self.action_spaces = {
            agent: spaces.Discrete(45 * 8 * 3 + 1) for agent in self.possible_agents
        }

        # The main observation space is a 5x9 space representing the board. It has 7 channels
        # representing -
        #   Channel 1: whose turn to play (all 0s for white, all 1s for black)
        #   Channel 2: move counter counting up to 44 moves. Represented by a single channel where
        #              the n^th element in the flattened channel is set if there has been n moves.
        #   Channel 3: positions used (1 in the squares whose positions have been used in a
        #              capturing sequence)
        #   Channel 4: last capture position (1 in the position to which captured piece was moved
        #              else 0)
        #   Channel 5: last direction used (all 1s in the nth row if the nth direction was last
        #              used else 0. Direction index is determined by a canonical order)
        #   Channel 6: all 1s to help neural networks find board edges in padded convolutions
        #   Channel 7: white piece positions (1 if a piece exists in the corresponding index)
        #   Channel 8: black piece positions
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(5, 9, 8), dtype=np.int32
                    ),  # ideally should be np.bool
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(45 * 8 * 3 + 1,), dtype=np.int32
                    ),  # ideally should be np.int8
                }
            )
            for name in self.possible_agents
        }
        self.board_state = FanoronaState()

    def render(self, mode: str = "human"):
        if mode == "human":
            print(str(self.board_state))
        elif mode == "svg":
            print(self.board_state.to_svg())

    def observe(self, agent: str):
        observation = self.board_state.get_observation(
            self.possible_agents.index(agent)
        )
        legal_moves = (
            self.board_state.legal_moves if agent == self.agent_selection else []
        )

        action_mask = np.zeros(45 * 8 * 3 + 1, np.int32)
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def close(self):
        pass

    def state(self):
        return self.board_state

    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.board_state.reset()

    def step(self, action: int):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        chosen_move = FanoronaMove.action_to_move(action)
        self.board_state.push(chosen_move)
        if self.board_state.is_game_over():
            result = self.board_state.get_result()
            for i, name in enumerate(self.agents):
                self.dones[name] = True
                result_coeff = 1 if i == 0 else -1
                self.rewards[name] = result * result_coeff
                self.infos[name] = {"legal_moves": []}

        self._accumulate_rewards()
