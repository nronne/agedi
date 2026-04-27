import math
import torch
import pytest

from agedi.diffusion.sdes.noise_schedules import Linear, Exponential, Cosine, NoiseSchedule, DiscreteExponential


# ── Linear ──────────────────────────────────────────────────────────────────

class TestLinear:
    def setup_method(self):
        self.sched = Linear(min=0.1, max=1.0)

    def test_f_at_boundaries(self):
        assert self.sched.f(0.0) == pytest.approx(0.1)
        assert self.sched.f(1.0) == pytest.approx(1.0)

    def test_f_midpoint(self):
        assert self.sched.f(0.5) == pytest.approx(0.55)

    def test_fprime_constant(self):
        assert self.sched.fprime(0.0) == pytest.approx(0.9)
        assert self.sched.fprime(0.5) == pytest.approx(0.9)
        assert self.sched.fprime(1.0) == pytest.approx(0.9)

    def test_fint_at_zero(self):
        assert self.sched.fint(0.0) == pytest.approx(0.0)

    def test_fint_at_one(self):
        expected = 0.1 * 1.0 + 0.5 * 0.9 * 1.0**2
        assert self.sched.fint(1.0) == pytest.approx(expected)

    def test_df2dt(self):
        t = 0.5
        expected = 2 * self.sched.f(t) * self.sched.fprime(t)
        assert self.sched.df2dt(t) == pytest.approx(expected)


# ── Exponential ─────────────────────────────────────────────────────────────

class TestExponential:
    def setup_method(self):
        self.sched = Exponential(min=0.01, max=1.0)

    def test_f_at_boundaries(self):
        assert self.sched.f(0.0) == pytest.approx(0.01)
        assert self.sched.f(1.0) == pytest.approx(1.0)

    def test_f_midpoint_is_geometric_mean(self):
        expected = math.sqrt(0.01 * 1.0)
        assert self.sched.f(0.5) == pytest.approx(expected, rel=1e-5)

    def test_fprime_positive(self):
        t = torch.tensor(0.5)
        val = self.sched.fprime(t)
        assert val > 0

    def test_fint_at_zero(self):
        assert self.sched.fint(0.0) == pytest.approx(0.0)

    def test_fint_sign(self):
        t = torch.tensor(0.5)
        val = self.sched.fint(t)
        assert val > 0


# ── Cosine ──────────────────────────────────────────────────────────────────

class TestCosine:
    def setup_method(self):
        self.sched = Cosine(min=0.0, max=1.0)

    def test_f_at_zero(self):
        t = torch.tensor(0.0)
        assert self.sched.f(t).item() == pytest.approx(0.0, abs=1e-6)

    def test_f_at_one(self):
        t = torch.tensor(1.0)
        assert self.sched.f(t).item() == pytest.approx(1.0, abs=1e-6)

    def test_fprime_positive_midrange(self):
        t = torch.tensor(0.5)
        assert self.sched.fprime(t) > 0

    def test_fint_increases(self):
        t0 = torch.tensor(0.1)
        t1 = torch.tensor(0.9)
        assert self.sched.fint(t1) > self.sched.fint(t0)

    def test_df2dt_via_base_class(self):
        t = torch.tensor(0.3)
        expected = 2 * self.sched.f(t) * self.sched.fprime(t)
        assert self.sched.df2dt(t).item() == pytest.approx(expected.item(), rel=1e-5)


# ── DiscreteExponential ──────────────────────────────────────────────────────

class TestDiscreteExponential:
    def setup_method(self):
        self.sched = DiscreteExponential(min=0.01, max=3.0)

    def test_total_noise_equals_f(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(self.sched.total_noise(t), self.sched.f(t))

    def test_rate_noise_equals_fprime(self):
        t = torch.tensor([0.2, 0.7])
        assert torch.allclose(self.sched.rate_noise(t), self.sched.fprime(t))

    def test_beta_min_beta_max_aliases(self):
        """beta_min / beta_max keyword arguments are backward-compatible aliases."""
        sched = DiscreteExponential(beta_min=0.01, beta_max=3.0)
        assert sched.min == pytest.approx(0.01)
        assert sched.max == pytest.approx(3.0)

    def test_positional_and_keyword_equivalent(self):
        s_pos = DiscreteExponential(0.01, 3.0)
        s_kw = DiscreteExponential(min=0.01, max=3.0)
        t = torch.tensor(0.5)
        assert s_pos.total_noise(t).item() == pytest.approx(s_kw.total_noise(t).item())

    def test_total_noise_at_zero(self):
        assert self.sched.total_noise(torch.tensor(0.0)).item() == pytest.approx(0.01)

    def test_total_noise_at_one(self):
        assert self.sched.total_noise(torch.tensor(1.0)).item() == pytest.approx(3.0)

