from gymnasium import spaces
import gymnasium.spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from .move import FanoronaMove
from .state import FanoronaState


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
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

    metadata = {"render_mode": ["human", "svg"], "name": "fanorona_v1"}

    def __init__(self, render_mode="human"):
        self.possible_agents = [f"player_{str(r)}" for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # The action space is a (5x9x8x3+1)-dimensional array. Each of the 5x9 positions identifies
        # the square from which to "pick up" the piece. 8 planes encode the possible directions
        # along which the piece will be moved (SW, S, SE, W, E, NW, N, NE). 3 planes encode the
        # capture type of the move (paika, approach, withdrawal). The last action denotes a manual
        # end turn.
        self._action_spaces = {
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
        self._observation_spaces = {
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
        self.render_mode = render_mode

    def state(self):
        return self.board_state

    def render(self):
        if self.render_mode == "human":
            print(str(self.board_state))
        elif self.render_mode == "svg":
            print(self.board_state.to_svg())

    def observe(self, agent: str):
        observation = self.board_state.get_observation(
            self.possible_agents.index(agent)
        )
        legal_moves = (
            self.board_state.legal_moves if agent == self.agent_selection else []
        )

        action_mask = np.zeros(45 * 8 * 3 + 1, np.int8)
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def close(self):
        pass

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.board_state.reset()

    def step(self, action: int):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        if self.board_state.is_game_over():
            result = self.board_state.get_result()
        else:
            chosen_move = FanoronaMove.action_to_move(action)
            self.board_state.push(chosen_move)
            result = None

        if self._agent_selector.is_last():
            self.terminations = {
                agent: (result == 1 or result == -1) for agent in self.agents
            }
            self.truncations = {agent: (result == 0) for agent in self.agents}
            if result is not None:
                (
                    self.rewards[self.agents[0]],
                    self.rewards[self.agents[1]],
                ) = result * 1, result * (-1)
                self.infos[self.agents[0]], self.infos[self.agents[1]] = {
                    "legal_moves": []
                }, {"legal_moves": []}
            else:
                self.rewards[self.agents[0]], self.rewards[self.agents[1]] = 0, 0

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.board_state
        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
