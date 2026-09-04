"""Monte Carlo dispersion: perturb the laying angles, score the hits.

For each aim point the campaign fires ``n_shots`` rounds whose elevation and
azimuth are perturbed by independent zero-mean normals, integrates each one to
the water line, and asks which hulls the impact point falls inside.  The
outputs are a hit rate per target and the circular error probable (CEP) of the
impact scatter.

The perturbations for the whole campaign are drawn up front from a single
seeded generator, so a run is reproducible end to end and can be resumed at a
point boundary without changing any later draw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from ..simulator import BallisticSimulator
    from ..vessel import Vessel


@dataclass(frozen=True)
class DispersionSettings:
    """How the laying angles are perturbed.

    Attributes
    ----------
    n_shots:
        Rounds fired per aim point.
    sigma_elevation_deg, sigma_azimuth_deg:
        Standard deviations of the laying errors, in degrees.
    alpha0_deg, beta0_deg:
        Initial projectile attitude, held fixed across the campaign.
    w_j0, w_k0:
        Initial transverse rates, in rad/s.
    max_time:
        Integration horizon per shot, in s.
    seed:
        Seed of the perturbation generator.
    """

    n_shots: int = 1000
    sigma_elevation_deg: float = 0.1
    sigma_azimuth_deg: float = 0.05
    alpha0_deg: float = 0.0
    beta0_deg: float = 0.0
    w_j0: float = 5.0
    w_k0: float = 5.0
    max_time: float = 100.0
    seed: int = 16184331


@dataclass
class AimPoint:
    """One nominal firing solution to be sampled.

    Attributes
    ----------
    elevation_deg, azimuth_deg:
        Nominal laying angles.
    nominal_range_m, nominal_drift_m:
        Where the unperturbed round is expected to land; errors are measured
        against this point.
    label:
        Optional identifier carried into the results table.
    """

    elevation_deg: float
    azimuth_deg: float
    nominal_range_m: float
    nominal_drift_m: float
    label: Optional[str] = None

    @classmethod
    def from_row(cls, row: Mapping, label: Optional[str] = None) -> "AimPoint":
        """Build from a row of the selected-points table."""
        return cls(
            elevation_deg=float(row["Elevacao_deg"]),
            azimuth_deg=float(row["Azimute_otimo_deg"]),
            nominal_range_m=float(row["Alcance_x_m"]),
            nominal_drift_m=float(row["Desvio_z_resultante_m"]),
            label=label,
        )


@dataclass
class PointResult:
    """Statistics for one aim point."""

    aim_point: AimPoint
    point_number: int
    n_shots: int
    n_valid: int
    errors_x_m: np.ndarray
    errors_z_m: np.ndarray
    miss_distances_m: np.ndarray
    flight_times_s: np.ndarray
    hits: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def hit_rate(self, target_name: str) -> float:
        """Fraction of valid shots that landed inside ``target_name``, in percent."""
        if self.n_valid == 0:
            return 0.0
        return 100.0 * self.hits.get(target_name, 0) / self.n_valid

    def cep(self, percentile: float = 50.0) -> float:
        """Radius containing ``percentile`` percent of the impacts, in m."""
        if len(self.miss_distances_m) == 0:
            return float("nan")
        if percentile == 50.0:
            return float(np.median(self.miss_distances_m))
        return float(np.percentile(self.miss_distances_m, percentile))

    def as_record(self) -> Dict[str, object]:
        """Flatten into the row layout of the campaign's output workbook."""
        record: Dict[str, object] = {
            "Ponto_numero": self.point_number,
            "Elevacao_deg": self.aim_point.elevation_deg,
            "Azimute_deg": self.aim_point.azimuth_deg,
            "Alcance_m": self.aim_point.nominal_range_m,
            "Desvio_Z_nominal_m": self.aim_point.nominal_drift_m,
            "N_simulacoes": self.n_shots,
            "N_validas": self.n_valid,
            "Erro_X_medio_m": float(self.errors_x_m.mean()),
            "Erro_X_std_m": float(self.errors_x_m.std()),
            "Erro_X_min_m": float(self.errors_x_m.min()),
            "Erro_X_max_m": float(self.errors_x_m.max()),
            "Erro_Z_medio_m": float(self.errors_z_m.mean()),
            "Erro_Z_std_m": float(self.errors_z_m.std()),
            "Erro_Z_min_m": float(self.errors_z_m.min()),
            "Erro_Z_max_m": float(self.errors_z_m.max()),
            "CEP50_m": self.cep(50.0),
            "CEP90_m": self.cep(90.0),
            "CEP95_m": self.cep(95.0),
            "Tempo_voo_medio_s": float(self.flight_times_s.mean()),
            "Tempo_simulacao_s": self.wall_time_s,
        }
        for name, count in self.hits.items():
            record[f"Acertos_{name}"] = count
            record[f"Taxa_acerto_{name}_pct"] = self.hit_rate(name)
        return record


class MonteCarloCampaign:
    """Run the dispersion study over a list of aim points.

    Parameters
    ----------
    simulator:
        A configured :class:`~sixdof.simulator.BallisticSimulator`.
    build_targets:
        Called as ``build_targets((range_m, drift_m))`` and returns a mapping
        of target name to :class:`~sixdof.vessel.Vessel`, all centred on the
        nominal impact point.  :func:`sixdof.presets.surface_target_fleet` fits
        this signature.
    settings:
        Perturbation and integration settings.

    Examples
    --------
    >>> campaign = MonteCarloCampaign(simulator, surface_target_fleet)  # doctest: +SKIP
    >>> results = campaign.run(aim_points)                              # doctest: +SKIP
    >>> campaign.to_frame(results).head()                               # doctest: +SKIP
    """

    def __init__(
        self,
        simulator: "BallisticSimulator",
        build_targets: Callable[[Sequence[float]], Mapping[str, "Vessel"]],
        settings: DispersionSettings = DispersionSettings(),
    ) -> None:
        self.simulator = simulator
        self.build_targets = build_targets
        self.settings = settings

    # ------------------------------------------------------------------
    def draw_perturbations(self, n_points: int) -> tuple:
        """Pre-draw every laying error for the campaign.

        Returns
        -------
        tuple
            ``(delta_elevation, delta_azimuth)``, each of length
            ``n_points * settings.n_shots``.

        Notes
        -----
        Uses the legacy global seeding (``numpy.random.seed`` then
        ``numpy.random.normal``) so a re-run reproduces the numbers published in
        the thesis exactly.
        """
        settings = self.settings
        total = n_points * settings.n_shots
        np.random.seed(settings.seed)
        delta_elevation = np.random.normal(0, settings.sigma_elevation_deg, total)
        delta_azimuth = np.random.normal(0, settings.sigma_azimuth_deg, total)
        return delta_elevation, delta_azimuth

    def run(
        self,
        aim_points: Sequence[AimPoint],
        progress_every: int = 50,
        verbose: bool = True,
        on_point_complete: Optional[Callable[[int, PointResult], None]] = None,
        *,
        campaign_size: Optional[int] = None,
        first_point_index: int = 0,
    ) -> List[PointResult]:
        """Fire the campaign, or a slice of it.

        Parameters
        ----------
        aim_points:
            Nominal firing solutions, in the order they should be simulated.
        progress_every:
            Print a progress line every N shots within a point; 0 disables it.
        on_point_complete:
            Called as ``(point_number, result)`` after each aim point, which is
            where a caller can checkpoint results to disk.
        campaign_size:
            Number of aim points in the **whole** campaign, when ``aim_points``
            is only part of it.  Defaults to ``len(aim_points)``.
        first_point_index:
            Zero-based position of ``aim_points[0]`` within the whole campaign.

        Notes
        -----
        The last two arguments exist so that a partial run draws the same
        perturbations it would have drawn inside the full campaign.  They
        matter more than they look: the legacy generator draws *all* the
        elevation errors first and *then* all the azimuth errors, so the
        azimuth stream starts at a position that depends on the total number of
        shots.  Drawing for 1 point instead of 163 therefore yields the correct
        elevations but entirely different azimuths.  Pass ``campaign_size`` (and
        ``first_point_index`` when skipping ahead) whenever a partial run must
        line up with the published results.
        """
        settings = self.settings
        total_points = len(aim_points) if campaign_size is None else campaign_size
        if total_points < first_point_index + len(aim_points):
            raise ValueError(
                "campaign_size must cover first_point_index plus the aim points given"
            )
        delta_elevation, delta_azimuth = self.draw_perturbations(total_points)

        results: List[PointResult] = []
        shot_counter = first_point_index * settings.n_shots

        for point_index, aim_point in enumerate(aim_points):
            point_number = first_point_index + point_index + 1
            if verbose:
                print(f"\n{'-'*80}")
                print(
                    f"PONTO {point_number}/{total_points} | "
                    f"Elevação: {aim_point.elevation_deg:.1f}° | "
                    f"Alcance: {aim_point.nominal_range_m:.0f} m"
                )
                print(f"{'-'*80}")

            targets = dict(
                self.build_targets((aim_point.nominal_range_m, aim_point.nominal_drift_m))
            )
            hits = {name: 0 for name in targets}
            errors_x: List[float] = []
            errors_z: List[float] = []
            miss_distances: List[float] = []
            flight_times: List[float] = []

            started = time.time()

            for shot in range(settings.n_shots):
                elevation = aim_point.elevation_deg + delta_elevation[shot_counter]
                azimuth = aim_point.azimuth_deg + delta_azimuth[shot_counter]
                shot_counter += 1

                self.simulator.weapon.set_firing_angles(
                    elevation_deg=elevation, azimuth_deg=azimuth
                )

                try:
                    trajectory = self.simulator.simulate(
                        max_time=settings.max_time,
                        alpha0_deg=settings.alpha0_deg,
                        beta0_deg=settings.beta0_deg,
                        w_j0=settings.w_j0,
                        w_k0=settings.w_k0,
                        verbose=False,
                    )
                except Exception:  # noqa: BLE001 - a failed shot is simply dropped
                    continue

                impact = trajectory.impact_point
                for name, vessel in targets.items():
                    if vessel.check_impact(
                        impact, time=trajectory.flight_time, check_height=False
                    ):
                        hits[name] += 1

                error_x = impact[0] - aim_point.nominal_range_m
                error_z = impact[2] - aim_point.nominal_drift_m
                errors_x.append(error_x)
                errors_z.append(error_z)
                miss_distances.append(float(np.sqrt(error_x**2 + error_z**2)))
                flight_times.append(trajectory.flight_time)

                if verbose and progress_every and (shot + 1) % progress_every == 0:
                    print(f"    [{shot+1}/{settings.n_shots}] válidas: {len(errors_x)}")

            result = PointResult(
                aim_point=aim_point,
                point_number=point_number,
                n_shots=settings.n_shots,
                n_valid=len(errors_x),
                errors_x_m=np.array(errors_x),
                errors_z_m=np.array(errors_z),
                miss_distances_m=np.array(miss_distances),
                flight_times_s=np.array(flight_times),
                hits=hits,
                wall_time_s=time.time() - started,
            )
            results.append(result)

            if verbose and result.n_valid:
                print(f"\n  ✓ Ponto concluído em {result.wall_time_s:.1f} s")
                print(f"    CEP50: {result.cep(50):.2f} m | CEP90: {result.cep(90):.2f} m")
                for name in targets:
                    print(
                        f"      • {name}: {result.hit_rate(name):.1f}% "
                        f"({result.hits[name]}/{result.n_valid})"
                    )

            if on_point_complete is not None:
                on_point_complete(point_number, result)

        return results

    @staticmethod
    def to_frame(results: Sequence[PointResult]) -> pd.DataFrame:
        """Assemble the campaign results into the output table."""
        records = [result.as_record() for result in results if result.n_valid]
        return pd.DataFrame.from_records(records)


__all__ = [
    "MonteCarloCampaign",
    "DispersionSettings",
    "AimPoint",
    "PointResult",
]
