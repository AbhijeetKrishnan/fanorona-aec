import functools
from typing import Any, Dict, List, Literal, Tuple, TypeAlias, TypedDict

import gymnasium.spaces
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from .fanorona_move import END_TURN_ACTION, ActionType, FanoronaMove
from .fanorona_state import AgentId, FanoronaState
from .utils import Piece, MOVE_LIMIT

RenderMode: TypeAlias = str

DRAW_REWARD = 0
WIN_REWARD = 1


class Metadata(TypedDict):
    render_modes: List[RenderMode]
    name: str
    is_parallelizable: bool
    render_fps: int


class Observation(TypedDict):
    observation: np.ndarray[
        Tuple[Literal[5], Literal[9], Literal[7]], np.dtype[np.int8]
    ]
    action_mask: np.ndarray[Literal[1081], np.dtype[np.int8]]  # 5 * 9 * 8 * 3 + 1


def env(render_mode: RenderMode | None = None) -> AECEnv:
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = FanoronaEnv(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class FanoronaEnv(AECEnv):  # type: ignore
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

    metadata: Metadata = {
        "render_modes": ["human", "fen", "svg"],
        "name": "fanorona_v3",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(self, render_mode: RenderMode | None = "human"):
        super().__init__()

        self.possible_agents = ["black", "white"]
        self.board_state = FanoronaState()
        self.render_mode = render_mode

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> None:
        self.agents = self.possible_agents[:]
        self.timestep = 0

        self.board_state.reset()
        self.agent_selection = "black"

        infos = self._get_infos()

        # needed for passing api_test and performance_benchmark
        self.terminations: Dict[AgentId, bool] = {agent: False for agent in self.agents}
        self.truncations: Dict[AgentId, bool] = {agent: False for agent in self.agents}
        self.rewards: Dict[AgentId, float] = {agent: 0 for agent in self.agents}
        self._cumulative_rewards: Dict[AgentId, float] = {
            agent: 0 for agent in self.agents
        }
        self.infos = infos

        if self.render_mode == "human":
            self.render()

    def step(self, action: ActionType) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)

        # push the move
        chosen_move = FanoronaMove.from_action(action)
        self.board_state.push(chosen_move)

        # check termination conditions
        self.terminations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        if self.board_state.done:
            if self.board_state.winner is None:
                result = DRAW_REWARD
            elif self.board_state.winner == Piece.BLACK:
                result = WIN_REWARD
            else:
                result = -WIN_REWARD
            self.terminations = {agent: True for agent in self.agents}
            self.rewards = {"black": result, "white": -result}

        # check truncation conditions
        self.truncations = {agent: False for agent in self.agents}
        if self.timestep >= MOVE_LIMIT:
            self.truncations = {agent: True for agent in self.agents}
            self.rewards = {agent: 0 for agent in self.agents}
        self.timestep += 1

        # get infos
        self.infos = self._get_infos()

        self._accumulate_rewards()

        # update agent selection
        if self.board_state.turn_to_play == Piece.BLACK:
            self.agent_selection = "black"
        else:
            self.agent_selection = "white"

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: AgentId) -> Observation:
        observation = self.board_state.get_observation(agent)
        legal_moves = (
            self.board_state.legal_moves if agent == self.agent_selection else []
        )

        action_mask = np.zeros(END_TURN_ACTION + 1, np.int8)
        action_mask[legal_moves] = 1

        return {"observation": observation, "action_mask": action_mask}

    def _get_infos(self) -> Dict[AgentId, Dict[str, Any]]:
        infos: Dict[AgentId, Dict[str, Any]] = {agent: {} for agent in self.agents}
        infos[self.agent_selection] = {"legal_moves": self.board_state.legal_moves}
        return infos

    def render(self) -> None:
        match self.render_mode:
            case "fen":
                print(str(self.board_state))
            case "svg":
                print(self.board_state.to_svg())
            case "human" | _:
                print(self.board_state.as_rich_board())

    def state(self) -> FanoronaState:
        return self.board_state

    def close(self) -> None:
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentId) -> gymnasium.spaces.Dict:
        """
        The main observation space is a 5x9 space representing the board. It has 7 channels
        representing -
          Channel 1: whose turn to play (all 0s for white, all 1s for black)
        #   Channel 2: move counter counting up to 44 moves. Represented by a single channel where
        #              the n^th element in the flattened channel is set if there has been n moves.
          Channel 2: positions used (1 in the squares whose positions have been used in a
                     capturing sequence)
          Channel 3: last capture position (1 in the position to which captured piece was moved
                     else 0)
          Channel 4: last direction used (all 1s in the nth row if the nth direction was last
                     used else 0. Direction index is determined by a canonical order)
          Channel 5: all 1s to help neural networks find board edges in padded convolutions
          Channel 6: white piece positions (1 if a piece exists in the corresponding index)
          Channel 7: black piece positions
        """
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=1, shape=(5, 9, 7), dtype=np.int8
                ),  # ideally should be np.bool
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(45 * 8 * 3 + 1,), dtype=np.int32
                ),  # ideally should be np.int8
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentId) -> gymnasium.spaces.Discrete:
        """
        The action space is a (5x9x8x3+1)-dimensional array. Each of the 5x9 positions identifies
        the square from which to "pick up" the piece. 8 planes encode the possible directions
        along which the piece will be moved (SW, S, SE, W, E, NW, N, NE). 3 planes encode the
        capture type of the move (paika, approach, withdrawal). The last action denotes a manual
        end turn.
        """
        return spaces.Discrete(END_TURN_ACTION + 1)
