import pytest

from agedi.diffusion import Diffusion


def test_init():
    diffusion = Diffusion()
    assert diffusion is not None
