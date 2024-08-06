# Fanorona AEC Environment

[![PyPI v3.0.2](https://img.shields.io/pypi/v/fanorona-aec)](https://pypi.org/project/fanorona-aec/3.0.2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


```
  A B C D E F G H I
5 ●─●─●─●─●─●─●─●─●
  │╲│╱│╲│╱│╲│╱│╲│╱│
4 ●─●─●─●─●─●─●─●─●
  │╱│╲│╱│╲│╱│╲│╱│╲│
3 ●─○─●─.─○─.─○─●─○
  │╲│╱│╲│╱│╲│╱│╲│╱│
2 ○─○─○─○─○─○─○─○─○
  │╱│╲│╱│╲│╱│╲│╱│╲│
1 ○─○─○─○─○─○─○─○─○

W to play
Last capture: E3 E
Visited: D3, E3
Half-moves: 0
```

This is an implementation of the Fanorona board game as a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) [AEC](https://arxiv.org/abs/2009.13051) game.
The rules have been sourced from [here](https://www.mindsports.nl/index.php/the-pit/528-fanorona).
An additional rule where games exceeding $44$ moves end in a draw has been implemented, since the
original rules do not have explicit draw conditions.

## Installation

### Using pip (recommended)

```bash
python -m pip install fanorona-aec
```

### Local

```bash
git clone https://github.com/AbhijeetKrishnan/fanorona-aec.git
cd fanorona-aec
python -m pip install .
```

## Usage

### Setting up a basic environment

In a Python shell, run the following:

```python
from fanorona_aec import fanorona_v3
env = fanorona_v3.env()
```

See [`demo.py`](./demo.py) for a script that implements a simple random policy to interact with the environment.

## Testing

We use [pytest](http://doc.pytest.org/) for tests. You can run them via:

```bash
git clone https://github.com/AbhijeetKrishnan/fanorona-aec.git
cd fanorona-aec
python -m pip install .[dev]
pytest
```
