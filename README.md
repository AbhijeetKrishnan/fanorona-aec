# Fanorona Gym Environment

This is an implementation of the Fanorona board game as an OpenAI Gym environment. The rules have
been sourced from [here](https://www.mindsports.nl/index.php/the-pit/528-fanorona). An additional
rule where games exceeding 50 half-moves end in a draw has been implemented, since the original
rules do not have explicit draw conditions.

## Installation

```bash
git clone https://github.com/AbhijeetKrishnan/gym-fanorona.git
cd gym-fanorona
pip install -e .
```

## Usage

### Setting up a basic environment

In a Python shell, run the following:

```python
import gym
import gym_fanorona
env = gym.make('fanorona-v0')
```

### Running a game between two (random) agents

In a Python shell, run the following:

```python
import gym
import gym_fanorona
from gym_fanorona.agents.random_agent import RandomAgent

white, black = RandomAgent(), RandomAgent()
env = gym.make('fanorona-v0', white_player=white, black_player=black)
moves = env.play_game()
```

## Testing

We use [pytest](http://doc.pytest.org/) for tests. You can run them via:

```bash
pytest
```

## TODO

- [x] Refactor state into a state object with attributes
- [x] Refactor action into an action object with attributes, and a convenient string representation to interconvert between
- [x] Write a visual interface for playing Fanorona
- [x] Add multiple tests for each method used
- [x] Refactor coordinate conversion methods into a single method with flags
- [x] Refactor validity checking into list of smaller rule checks combined in a larger method, so that is_valid() and is_capture_valid() can reuse code (possibly as a list of methods which must all return True)
- [x] Organize imports to remove unused imports
- [x] Test environment with a complete game
- [ ] Build interface to easily communicate moves and visualize output
- [ ] Would dynamically changing action space be better than current implementation?
- [ ] Use` capture_type` to indicate `end_turn` action as well (reduces number of states from 2430 to 1620)
- [ ] Turn `FanoronaEnv.get_valid_moves()` into a generator
- [ ] Replace types in FanoronaState with numpy types from numpy.typing OR consider not using numpy at all
- [x] Automatically delete .svg files generated from tests
- [ ] Clarify reward-giving mechanism to agents (at each step, and at end of game; who stores the cumulative reward?)
