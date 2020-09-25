# Fanorona Gym Environment

This is an implementation of the Fanorona board game as an OpenAI Gym environment. The rules have
been sourced from [here](https://www.mindsports.nl/index.php/the-pit/528-fanorona). An additional
rule where games exceeding 50 half-moves end in a draw has been implemented, since the original
rules do not have explicit draw conditions.

## Installation

```
git clone https://github.com/AbhijeetKrishnan/gym-fanorona.git
cd gym-fanorona
pip install -e .
```

## Usage

In a Python shell, run the following:
```
import gym
import gym_fanorona
env = gym.make('fanorona-v0')
```

## Testing

We use [pytest](http://doc.pytest.org/) for tests. You can run them via:
```
pytest
```

## TODO

[ ] Refactor state into a state object with attributes
[ ] Write a visual interface for playing Fanorona
[ ] Add multiple tests for each method used
[ ] Refactor coordinate conversion methods into a single method with flags
