# Fanorona AEC Environment

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is an implementation of the Fanorona board game as a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) [AEC](https://arxiv.org/abs/2009.13051) game.
The rules have been sourced from [here](https://www.mindsports.nl/index.php/the-pit/528-fanorona).
An additional rule where games exceeding 45 moves end in a draw has been implemented, since the
original rules do not have explicit draw conditions.

## Installation

```bash
git clone https://github.com/AbhijeetKrishnan/fanorona-aec.git
cd fanorona-aec
pip install -e .
```

## Usage

### Setting up a basic environment

In a Python shell, run the following:

```python
import pettingzoo
import fanorona_aec
env = fanorona_v0.env()
```

## Testing

We use [pytest](http://doc.pytest.org/) for tests. You can run them via:

```bash
pytest
```
