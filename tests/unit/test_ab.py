"""The A/B harness.

What is tested here is mostly that the instrument does not lie: the blinding
actually blinds, position bias is detectable rather than assumed away, and the
summary refuses to claim significance it does not have.
"""

from __future__ import annotations

import random
from pathlib import Path

from aura.personalization.ab import ABLog, Comparison, present, resolve


class _Coin:
    """A deterministic stand-in for random.Random."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0) if self._values else 0.9


# -- blinding ---------------------------------------------------------------


def test_the_adapted_answer_can_be_shown_either_way_round() -> None:
    first, shown_a, shown_b = present("q", "ADAPTED", "BASELINE", _Coin(0.1))  # type: ignore[arg-type]
    assert (shown_a, shown_b) == ("ADAPTED", "BASELINE")
    assert first.adapted_shown_as == "A"

    second, shown_a, shown_b = present("q", "ADAPTED", "BASELINE", _Coin(0.9))  # type: ignore[arg-type]
    assert (shown_a, shown_b) == ("BASELINE", "ADAPTED")
    assert second.adapted_shown_as == "B"


def test_ordering_is_actually_randomised() -> None:
    """A 'blind' test that always shows the adapted answer first is not blind."""
    rng = random.Random(1234)
    sides = {present("q", "a", "b", rng)[0].adapted_shown_as for _ in range(40)}
    assert sides == {"A", "B"}


def test_choosing_a_side_maps_back_to_the_right_arm() -> None:
    comparison, _, _ = present("q", "ADAPTED", "BASELINE", _Coin(0.9))  # type: ignore[arg-type]
    assert comparison.adapted_shown_as == "B"

    assert resolve(comparison, "B").picked == "adapted"
    assert resolve(comparison, "A").picked == "baseline"


def test_a_tie_is_recordable() -> None:
    """Forcing a choice between two equally good answers manufactures signal."""
    comparison, _, _ = present("q", "a", "b", _Coin(0.1))  # type: ignore[arg-type]
    assert resolve(comparison, "tie").picked == "tie"


# -- the tally --------------------------------------------------------------


def _log(tmp_path: Path, *comparisons: Comparison) -> ABLog:
    log = ABLog(tmp_path / "ab.jsonl")
    for c in comparisons:
        log.record(c)
    return log


def _judged(picked: str, shown_as: str = "A") -> Comparison:
    return Comparison(
        prompt="q", adapted="a", baseline="b", adapted_shown_as=shown_as, picked=picked
    )


def test_the_tally_counts_each_arm(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        _judged("adapted"),
        _judged("adapted"),
        _judged("baseline"),
        _judged("tie"),
    )
    tally = log.tally()
    assert (tally.adapted, tally.baseline, tally.ties) == (2, 1, 1)
    assert tally.judged == 4


def test_unjudged_comparisons_are_not_counted(tmp_path: Path) -> None:
    log = _log(tmp_path, Comparison("q", "a", "b", "A"), _judged("adapted"))
    assert log.tally().judged == 1


def test_results_survive_a_restart(tmp_path: Path) -> None:
    """The raw data has to outlive the summary, or nobody can disagree with it."""
    _log(tmp_path, _judged("adapted"), _judged("baseline"))
    assert ABLog(tmp_path / "ab.jsonl").tally().judged == 2


def test_a_corrupt_line_loses_one_judgement_not_the_file(tmp_path: Path) -> None:
    path = tmp_path / "ab.jsonl"
    _log(tmp_path, _judged("adapted"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    _log(tmp_path, _judged("baseline"))

    assert ABLog(path).tally().judged == 2


# -- refusing to overclaim --------------------------------------------------


def test_a_small_sample_is_reported_as_meaningless(tmp_path: Path) -> None:
    """The point of the module docstring, enforced."""
    log = _log(tmp_path, *[_judged("adapted") for _ in range(8)])
    summary = log.tally().summary()
    assert "too few" in summary.lower()


def test_a_narrow_lead_is_not_called_a_result(tmp_path: Path) -> None:
    """13-11 out of 24 is a coin, and must not read as a win."""
    log = _log(
        tmp_path,
        *[_judged("adapted") for _ in range(13)],
        *[_judged("baseline") for _ in range(11)],
    )
    assert "within what chance would produce" in log.tally().summary()


def test_a_clear_lead_is_reported_plainly(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        *[_judged("adapted", shown_as="A") for _ in range(20)],
        *[_judged("baseline", shown_as="A") for _ in range(4)],
    )
    summary = log.tally().summary()
    assert "adapted 20" in summary
    assert "within what chance" not in summary


def test_position_bias_is_surfaced_not_assumed_away(tmp_path: Path) -> None:
    """If the user always picks whichever came first, the tally is measuring
    position, not quality -- and should say so."""
    log = _log(
        tmp_path,
        # Adapted shown first and picked, every time.
        *[_judged("adapted", shown_as="A") for _ in range(21)],
        *[_judged("baseline", shown_as="B") for _ in range(4)],
    )
    tally = log.tally()
    assert tally.picked_first_shown == 25
    assert "position bias" in tally.summary()


def test_no_data_says_so(tmp_path: Path) -> None:
    assert "No comparisons" in ABLog(tmp_path / "none.jsonl").tally().summary()
