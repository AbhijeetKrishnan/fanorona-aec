import functools
from typing import Any, Dict, List, Literal, Tuple, TypeAlias, TypedDict

import gymnasium.spaces
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .fanorona_move import END_TURN_ACTION, ActionType, FanoronaMove
from .fanorona_state import AgentId, FanoronaState
from .utils import Piece

RenderMode: TypeAlias = str


class Metadata(TypedDict):
    render_modes: List[RenderMode]
    name: str
    is_parallelizable: bool
    render_fps: int


class Observation(TypedDict):
    observation: np.ndarray[
        Tuple[Literal[5], Literal[9], Literal[8]], np.dtype[np.int8]
    ]
    action_mask: np.ndarray[Literal[1081], np.dtype[np.int8]]  # 5 * 9 * 8 * 3 + 1


def env(render_mode: RenderMode | None = None) -> AECEnv:
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):  # type: ignore
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
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: RenderMode | None = "human"):
        super().__init__()

        self.board_state = FanoronaState()

        self.agents: List[AgentId] = [f"player_{str(r)}" for r in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        # The action space is a (5x9x8x3+1)-dimensional array. Each of the 5x9 positions identifies
        # the square from which to "pick up" the piece. 8 planes encode the possible directions
        # along which the piece will be moved (SW, S, SE, W, E, NW, N, NE). 3 planes encode the
        # capture type of the move (paika, approach, withdrawal). The last action denotes a manual
        # end turn.
        self.action_spaces: Dict[AgentId, gymnasium.spaces.Discrete] = {
            agent: spaces.Discrete(END_TURN_ACTION + 1)
            for agent in self.possible_agents
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
        self.observation_spaces: Dict[AgentId, spaces.Dict] = {
            name: spaces.Dict(
                {
                    "observation": spaces.MultiBinary((5, 9, 8)),
                    "action_mask": spaces.MultiBinary(END_TURN_ACTION + 1),
                }
            )
            for name in self.possible_agents
        }

        self.rewards: Dict[AgentId, int] = {agent: 0 for agent in self.agents}
        self.infos: Dict[AgentId, Dict[str, Any] | None] = {
            agent: None for agent in self.agents
        }
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.agent_selection: AgentId | None = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentId) -> gymnasium.spaces.Dict:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentId) -> gymnasium.spaces.Discrete:
        return self.action_spaces[agent]

    def render(self) -> None:
        match self.render_mode:
            case "fen":
                print(str(self.board_state))
            case "svg":
                print(self.board_state.to_svg())
            case "human" | _:
                print(self.board_state.as_rich_board())

    def observe(self, agent: AgentId) -> Observation:
        observation = self.board_state.get_observation(agent)
        legal_moves = (
            self.board_state.legal_moves if agent == self.agent_selection else []
        )

        action_mask = np.zeros(END_TURN_ACTION + 1, np.int8)
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def close(self) -> None:
        pass

    def state(self) -> FanoronaState:
        return self.board_state

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> None:
        self.agents = self.possible_agents[:]

        self.board_state = FanoronaState()
        self.board_state.reset()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._state: Dict[AgentId, ActionType | None] = {
            agent: None for agent in self.agents
        }
        self.observations: Dict[AgentId, FanoronaState | None] = {
            agent: None for agent in self.agents
        }
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        for _, obs_space in self.observation_spaces.items():
            obs_space.seed(seed)

        for _, action_space in self.action_spaces.items():
            action_space.seed(seed)

        if self.render_mode == "human":
            self.render()

    def step(self, action: ActionType) -> None:
        assert self.agent_selection is not None
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection

        self._cumulative_rewards[current_agent] = 0

        self._state[self.agent_selection] = action

        chosen_move = FanoronaMove.from_action(action)
        legal_moves = list(self.board_state.legal_moves)
        # assert chosen_move in legal_moves
        self.board_state.push(chosen_move)
        game_over = self.board_state.done

        if game_over:
            result = 1 if self.board_state.winner == Piece.WHITE else -1
            (
                self.rewards[self.agents[0]],
                self.rewards[self.agents[1]],
            ) = (
                result,
                -result,
            )
        else:
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = 0, 0

        self.terminations = {agent: game_over for agent in self.agents}
        # no cause for external episode end
        self.truncations = {agent: False for agent in self.agents}

        # observe the current state
        self.observations[current_agent] = self.board_state
        self.infos[current_agent] = {"legal_moves": legal_moves}

        # selects the next agent
        self.agent_selection = self._agent_selector.next()
        # Adds rewards to _cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
