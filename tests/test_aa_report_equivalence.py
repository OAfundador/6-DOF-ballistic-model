"""The anti-air report must match the frozen engine character for character.

``tests/reference/legacy_motor.py`` is an unmodified copy of the author's
working engine: ``Motor.py`` plus the proximity-fuze event and the article's
verification ``main``.  It is what produced the anti-air numbers quoted in the
work.  This test runs that file's own ``main``, runs the equivalent through the
refactored package, and compares the two printed reports line by line.

This is slower than the rest of the suite (two full trajectory integrations,
roughly a minute).  Skip it with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

import legacy_motor as frozen_aa_engine  # noqa: E402
from compat import patch_legacy_coefficients  # noqa: E402

patch_legacy_coefficients(frozen_aa_engine)

from sixdof import (  # noqa: E402
    BallisticSimulator,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.aa import (  # noqa: E402
    evaluate_engagement,
    print_damage_report,
    print_trajectory_summary,
    ProximityFuze,
    shahed_136,
    vt_fcl_mk49,
)

#: The scenario hard-coded in the frozen engine's article main.
SCENARIO = dict(
    elevation_deg=39.6,
    azimuth_deg=-1.35,
    target_center=(16673.0, 200.0, 0.7),
    fuze_radius_m=24.38,
    fuze_arm_time_s=0.5,
    max_time_s=100.0,
)


def report_body(text: str) -> List[str]:
    """The comparable part of a report: from the trajectory summary onwards."""
    index = text.index("RESUMO DA TRAJETORIA")
    return [line.rstrip() for line in text[index:].splitlines() if line.strip()]


@pytest.fixture(scope="module")
def frozen_report() -> List[str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        frozen_aa_engine.run_article_verification_main()
    return report_body(buffer.getvalue())


@pytest.fixture(scope="module")
def refactored_report() -> List[str]:
    target = shahed_136(center=SCENARIO["target_center"])
    warhead = vt_fcl_mk49()
    fuze = ProximityFuze(
        target_center=target.center,
        radius_m=SCENARIO["fuze_radius_m"],
        arm_time_s=SCENARIO["fuze_arm_time_s"],
    )
    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(name='Projetil Naval 5"/38 AA VT(FCL)'),
        weapon=naval_5in38_gun(
            elevation_deg=SCENARIO["elevation_deg"],
            azimuth_deg=SCENARIO["azimuth_deg"],
        ),
        environment=standard_atmosphere(),
        aero_coeffs=naval_5in38_coefficients(),
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        trajectory = simulator.simulate(
            max_time=SCENARIO["max_time_s"], fuze=fuze, verbose=False
        )
        print_trajectory_summary(trajectory)
        burst, damage = evaluate_engagement(trajectory, target, warhead, fuze)
        print_damage_report(burst, damage)
    return report_body(buffer.getvalue())


@pytest.mark.slow
def test_reports_have_the_same_number_of_lines(frozen_report, refactored_report):
    assert len(frozen_report) == len(refactored_report)


@pytest.mark.slow
def test_reports_are_character_for_character_identical(frozen_report, refactored_report):
    """Every printed line matches, which pins every rounded value in the report."""
    differences = [
        (index, old, new)
        for index, (old, new) in enumerate(zip(frozen_report, refactored_report))
        if old != new
    ]
    assert not differences, "linhas divergentes:\n" + "\n".join(
        f"  [{i}] original  : {old}\n  [{i}] refatorado: {new}"
        for i, old, new in differences
    )


@pytest.mark.slow
def test_report_carries_the_expected_burst(refactored_report):
    """Guard the headline numbers explicitly, so a silent shift is visible."""
    text = "\n".join(refactored_report)
    assert "Motivo da parada      : fuze" in text
    assert "Distancia burst-alvo         : 24.380 m" in text
    assert "Fragmentos totais no modelo  : 2113" in text
    assert "Prob. destruicao Bernoulli   : 58.849674%" in text
