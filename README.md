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

In a Python shell, run the following:

```python
import gym
import gym_fanorona
env = gym.make('fanorona-v0')
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
- [ ] Organize imports to remove unused imports
- [ ] Test environment with a complete game
- [ ] Build interface to easily communicate moves and visualize output