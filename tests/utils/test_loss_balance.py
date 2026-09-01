import pytest

from agedi.utils.loss_balance import format_loss_balance, normalize_loss_balance


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, None),
        ("50:50", (0.5, 0.5)),
        ("80:20", (0.8, 0.2)),
        ("80/20", (0.8, 0.2)),
        ("0.8-0.2", (0.8, 0.2)),
        ("80%:20%", (0.8, 0.2)),
        ("8:2", (0.8, 0.2)),            # normalised by the sum
        ("0.2", (0.8, 0.2)),            # bare number = regressor fraction
        (0.2, (0.8, 0.2)),
        (0, (1.0, 0.0)),
        ((0.8, 0.2), (0.8, 0.2)),
        ([4, 1], (0.8, 0.2)),
        ({"diffusion": 0.8, "regressor": 0.2}, (0.8, 0.2)),
    ],
)
def test_normalize_loss_balance(spec, expected):
    result = normalize_loss_balance(spec)

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)
        assert sum(result) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "spec",
    [
        "80:20:10",         # three values
        "eighty:twenty",    # non-numeric
        "abc",
        (0.5,),             # wrong length
        (0.5, 0.2, 0.3),
        (-1.0, 2.0),        # negative
        (0.0, 0.0),         # sums to zero
        {"diffusion": 1.0, "typo": 0.0},
        object(),
    ],
)
def test_normalize_loss_balance_rejects_invalid(spec):
    with pytest.raises(ValueError):
        normalize_loss_balance(spec)


def test_format_loss_balance():
    assert format_loss_balance(None) == "disabled"
    assert format_loss_balance((0.8, 0.2)) == "80% / 20% (diffusion / regressor)"
