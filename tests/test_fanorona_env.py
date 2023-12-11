import pettingzoo
import pettingzoo.test
import pytest

from fanorona_aec import fanorona_v3

TEST_STATES = [
    "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - 0",  # start state
    # capturing seq after D2->E3 approach
    "WWWWWWWWW/WWW1WWWWW/BWBWWBWBW/BBBBB1BBB/BBBBBB1BB B - - 1",
    "9/9/3W1B3/9/9 W - - 49",  # random endgame state
]


@pytest.fixture(scope="function")
def env():
    env = fanorona_v3.env()
    env.reset()
    yield env
    env.close()


def test_api(env):
    "Test the env using PettingZoo's API test function"
    pettingzoo.test.api_test(env, num_cycles=10, verbose_progress=False)


def test_reset(env):
    "Verify that reset() executes without error"
    env.reset()


def test_reset_starting(env):
    "Verify that reset() sets the board state to the correct starting position"
    assert (
        str(env.state())
        == "WWWWWWWWW/WWWWWWWWW/BWBW1BWBW/BBBBBBBBB/BBBBBBBBB W - - - 0"
    )  # starting position


@pytest.mark.skip(reason="render_test does not support render.mode='svg'")
def test_render(env):
    "Verify that render() executes without error for human-readable output"
    pettingzoo.test.render_test(env)


def test_render_human(env):
    "Verify that render() executes without error for human-readable output"
    env.render_mode = "human"
    env.render()


def test_render_svg(env):
    "Verify that render() executes without error for svg"
    env.render_mode = "svg"
    env.render()


@pytest.mark.skip(
    reason="possibly subsumed by api_test unless I have important corner cases"
)
def test_step(env):
    "Test step() with a variety of moves from different initial states"
    pass


def test_performance_benchmark(env):
    "Run PettingZoo performance benchmark on the env"
    pettingzoo.test.performance_benchmark(env)
