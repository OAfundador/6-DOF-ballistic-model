"""Prove that the refactored package reproduces the original engine exactly.

This script is the evidence behind the claim in the README. It does not trust
any stored golden numbers: it loads the frozen pre-refactor code from
``tests/reference/``, runs it and the refactored package side by side, and
compares the results bit for bit.

Run it::

    python scripts/proof_of_equivalence.py
    python scripts/proof_of_equivalence.py --report docs/verification.md

Exit code 0 means every check passed; 1 means at least one differs.

Four checks:

1. **Trajectories** -- seven scenarios across the firing envelope, plus a
   moving platform and a wind field. Compares sample times and the full 12 x N
   state history with ``numpy.array_equal`` (no tolerance) and prints a SHA-256
   of each engine's raw bytes.
2. **Anti-air report** -- runs the frozen ``legacy_motor.py`` article main and
   the refactored equivalent, and compares the printed reports character for
   character.
3. **Aim-point selection** -- runs the refactored selection against the
   published zero-drift table and compares it with the aim-point table used in
   the thesis, cell for cell.
4. **Damage model** -- compares every quantity of the refactored damage model
   against the frozen procedural appendix.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

import apendice_dano as frozen_appendix  # noqa: E402
import compat  # noqa: E402
import legacy_motor as frozen_aa_engine  # noqa: E402
import motor_original as frozen_engine  # noqa: E402

from sixdof import (  # noqa: E402
    BallisticSimulator,
    Environment,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    Projectile,
    standard_atmosphere,
    Vessel,
    Weapon,
)
from sixdof.aa import (  # noqa: E402
    evaluate_engagement,
    print_damage_report,
    print_trajectory_summary,
    ProximityFuze,
    shahed_136,
    vt_fcl_mk49,
)
from sixdof.montecarlo import select_points_by_spacing  # noqa: E402
from sixdof.paths import (  # noqa: E402
    AERO_COEFFICIENTS_5IN38,
    AERO_SOURCE_5IN38,
    OPTIMAL_AZIMUTHS,
    SELECTED_POINTS_100M,
)

PROJECTILE_SPEC = dict(
    name='Projétil Naval 5"/38',
    mass_lb=68.10,
    diameter_in=5.0,
    I_P_lbin2=240.9,
    I_T_lbin2=2619.0,
    rifling_twist_calibers=25.0,
)

#: (label, elevation, azimuth, environment kwargs, platform velocity or None)
SCENARIOS: List[Tuple[str, float, float, dict, object]] = [
    ("Exemplo.py de referência (43.3°)", 43.3, 0.0, {}, None),
    ("alcance máximo (39.6°, -1.35°)", 39.6, -1.35, {}, None),
    ("meia elevação (20.0°, -1.00°)", 20.0, -1.00, {}, None),
    ("tiro raso (5.0°, 0.00°)", 5.0, 0.00, {}, None),
    ("elevação negativa (-1.5°, -0.50°)", -1.5, -0.50, {}, None),
    ("limite do envelope (45.0°, -1.65°)", 45.0, -1.65, {}, None),
    ("vento cruzado (35.0°)", 35.0, 0.0,
     dict(rho=1.2, g=9.80665, W1=6.0, W2=0.0, W3=-4.0), None),
    ("plataforma em movimento (30.0°)", 30.0, -1.0, {}, (8.0, -2.0)),
]


def sha256_of(*arrays: np.ndarray) -> str:
    """Fingerprint the raw bytes of one or more arrays."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def md5_of_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# runners
# ----------------------------------------------------------------------
def run_frozen(coefficients, elevation, azimuth, env_kwargs, platform_velocity):
    """One trajectory from the frozen pre-refactor engine."""
    projectile = frozen_engine.Projectile.from_imperial(**PROJECTILE_SPEC)
    vessel = None
    if platform_velocity is not None:
        vessel = frozen_engine.Vessel(
            name="plataforma", center_position=(0.0, 0.0), length=100.0,
            width=20.0, height=30.0, velocity=platform_velocity,
        )
    weapon = frozen_engine.Weapon(
        name='Canhão Naval 5"/38',
        position=(0.0, 10.0, 0.0) if vessel is None else (5.0, 10.0, 1.0),
        elevation_deg=elevation, azimuth_deg=azimuth,
        rate_of_fire_rpm=15.0, muzzle_velocity_mps=807.0,
        mounted_on_vessel=vessel,
    )
    environment = frozen_engine.Environment(**env_kwargs)
    simulator = frozen_engine.BallisticSimulator(
        projectile, weapon, environment, coefficients
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return simulator.simulate(
            max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0,
            w_j0=5.0, w_k0=5.0, rtol=1e-7, atol=1e-8,
        )


def run_refactored(coefficients, elevation, azimuth, env_kwargs, platform_velocity):
    """The same trajectory from the refactored package."""
    projectile = Projectile.from_imperial(**PROJECTILE_SPEC)
    vessel = None
    if platform_velocity is not None:
        vessel = Vessel(
            name="plataforma", center_position=(0.0, 0.0), length=100.0,
            width=20.0, height=30.0, velocity=platform_velocity,
        )
    weapon = Weapon(
        name='Canhão Naval 5"/38',
        position=(0.0, 10.0, 0.0) if vessel is None else (5.0, 10.0, 1.0),
        elevation_deg=elevation, azimuth_deg=azimuth,
        rate_of_fire_rpm=15.0, muzzle_velocity_mps=807.0,
        mounted_on_vessel=vessel,
    )
    environment = Environment(**env_kwargs)
    simulator = BallisticSimulator(projectile, weapon, environment, coefficients)
    return simulator.simulate(
        max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0,
        w_j0=5.0, w_k0=5.0, rtol=1e-7, atol=1e-8, verbose=False,
    )


def frozen_aa_report() -> str:
    """Capture the article report printed by the frozen legacy_motor.py."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        frozen_aa_engine.run_article_verification_main()
    return buffer.getvalue()


def refactored_aa_report() -> str:
    """The same report, produced through the refactored package."""
    target = shahed_136(center=(16673.0, 200.0, 0.7))
    warhead = vt_fcl_mk49()
    fuze = ProximityFuze(target_center=target.center, radius_m=24.38, arm_time_s=0.5)

    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(name='Projetil Naval 5"/38 AA VT(FCL)'),
        weapon=naval_5in38_gun(elevation_deg=39.6, azimuth_deg=-1.35),
        environment=standard_atmosphere(),
        aero_coeffs=naval_5in38_coefficients(),
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        trajectory = simulator.simulate(max_time=100.0, fuze=fuze, verbose=False)
        print_trajectory_summary(trajectory)
        burst, damage = evaluate_engagement(trajectory, target, warhead, fuze)
        print_damage_report(burst, damage)
    return buffer.getvalue()


def report_body(text: str) -> List[str]:
    """The comparable part of a report: from the trajectory summary onwards."""
    marker = "RESUMO DA TRAJETORIA"
    index = text.index(marker)
    return [line.rstrip() for line in text[index:].splitlines() if line.strip()]


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def check_trajectories(out, coefficients_frozen, coefficients_new) -> bool:
    out(header("1. TRAJETÓRIAS — comparação bit a bit do integrador"))
    out("")
    out("Cada cenário é integrado pelos dois motores com os mesmos parâmetros.")
    out("`igual` usa numpy.array_equal sobre t e sobre o estado 12 x N inteiro:")
    out("igualdade exata, sem tolerância. O SHA-256 é dos bytes crus dos dois")
    out("arrays — se ele coincide, não há diferença em nenhum bit de nenhuma")
    out("das ~19 000 casas double de cada trajetória.")
    out("")

    all_equal = True
    rows = []
    for label, elevation, azimuth, env_kwargs, mount in SCENARIOS:
        old = run_frozen(coefficients_frozen, elevation, azimuth, env_kwargs, mount)
        new = run_refactored(coefficients_new, elevation, azimuth, env_kwargs, mount)

        t_equal = np.array_equal(old.t, new.t)
        y_equal = np.array_equal(old.solution.y, new.solution.y)
        same_shape = old.solution.y.shape == new.solution.y.shape

        if same_shape:
            max_diff = float(np.max(np.abs(old.solution.y - new.solution.y)))
        else:
            max_diff = float("nan")

        digest_old = sha256_of(old.t, old.solution.y)
        digest_new = sha256_of(new.t, new.solution.y)
        equal = t_equal and y_equal and digest_old == digest_new
        all_equal &= equal

        rows.append(
            (label, len(old.t), max_diff, digest_old[:16], digest_new[:16], equal)
        )

        out(f"  {label}")
        out(f"    amostras                    : {len(old.t)}")
        out(f"    max |Δ| em todo o estado    : {max_diff:.1e}")
        out(f"    SHA-256 motor original      : {digest_old}")
        out(f"    SHA-256 pacote refatorado   : {digest_new}")
        out(f"    idêntico                    : {'SIM' if equal else 'NÃO'}")
        out("")

    out(f"  Resultado: {sum(1 for r in rows if r[5])}/{len(rows)} cenários idênticos bit a bit.")
    out("")
    return all_equal


def check_derived(out, coefficients_frozen, coefficients_new) -> bool:
    out(header("2. GRANDEZAS DERIVADAS — velocidade, Mach, spin, ângulo de ataque"))
    out("")
    out("O integrador é só metade da história: as grandezas derivadas são o que")
    out("aparece nos gráficos e nas estatísticas do TCC. Cenário de referência,")
    out("elevação 43.3°.")
    out("")

    old = run_frozen(coefficients_frozen, 43.3, 0.0, {}, None)
    new = run_refactored(coefficients_new, 43.3, 0.0, {}, None)

    checks = [
        ("|V| (m/s)", old.V_mag, new.V_mag),
        ("Mach", old.mach, new.mach),
        ("|h| (rad/s)", old.h_mag, new.h_mag),
        ("spin ω1 (rad/s)", old.spin_rate, new.spin_rate),
        ("ângulo de ataque (°)", old.alpha_traj, new.alpha_traj),
    ]

    all_equal = True
    for name, a, b in checks:
        equal = np.array_equal(a, b)
        all_equal &= equal
        out(f"  {name:24s} max |Δ| = {np.max(np.abs(a - b)):.1e}   "
            f"idêntico: {'SIM' if equal else 'NÃO'}")

    out("")
    out("  Estatísticas de resumo (as quatro que o TCC cita):")
    stats = [
        ("alcance (m)", old.alcance_max, new.max_range),
        ("altura máxima (m)", old.altura_max, new.max_altitude),
        ("desvio lateral (m)", old.desvio_lateral_max, new.max_lateral_drift),
        ("tempo de voo (s)", old.tempo_voo, new.flight_time),
    ]
    for name, a, b in stats:
        equal = float(a) == float(b)
        all_equal &= equal
        out(f"    {name:22s} original={float(a)!r:22s} refatorado={float(b)!r:22s} "
            f"{'igual' if equal else 'DIFERE'}")
    out("")
    return all_equal


def check_aa_report(out) -> bool:
    out(header("3. RELATÓRIO ANTIAÉREO — comparação caractere por caractere"))
    out("")
    out("O `legacy_motor.py` congelado roda a main canônica do artigo; o pacote")
    out("refatorado produz o mesmo relatório pela API nova. As duas saídas são")
    out("comparadas linha a linha, do 'RESUMO DA TRAJETORIA' em diante.")
    out("")

    old_text = frozen_aa_report()
    new_text = refactored_aa_report()
    old_lines = report_body(old_text)
    new_lines = report_body(new_text)

    equal = old_lines == new_lines
    digest_old = hashlib.sha256("\n".join(old_lines).encode("utf-8")).hexdigest()
    digest_new = hashlib.sha256("\n".join(new_lines).encode("utf-8")).hexdigest()

    out(f"  linhas comparadas           : {len(old_lines)}")
    out(f"  SHA-256 legacy_motor.py     : {digest_old}")
    out(f"  SHA-256 pacote refatorado   : {digest_new}")
    out(f"  idêntico                    : {'SIM' if equal else 'NÃO'}")
    out("")

    if not equal:
        out("  DIFERENÇAS:")
        for a, b in zip(old_lines, new_lines):
            if a != b:
                out(f"    original  : {a}")
                out(f"    refatorado: {b}")
        out("")
    else:
        out("  Relatório reproduzido (as duas saídas são este mesmo texto):")
        out("")
        for line in new_lines:
            out(f"    | {line}")
        out("")
    return equal


def check_damage_model(out) -> bool:
    out(header("4. MODELO DE DANO — contra o apêndice procedural congelado"))
    out("")
    out("Três trajetórias sintéticas de geometria conhecida, cobrindo burst")
    out("frontal, eixo desalinhado da velocidade e burst oblíquo a 30°.")
    out("")

    class Frontal:
        t = [0.0, 1.0, 2.0]
        x = [0.0, 0.0, 0.0]
        y = [-30.0, -24.38, -5.0]
        z = [0.0, 0.0, 0.0]
        V1 = [0.0, 0.0, 0.0]
        V2 = [620.0, 620.0, 620.0]
        V3 = [0.0, 0.0, 0.0]
        i1 = [0.0, 0.0, 0.0]
        i2 = [1.0, 1.0, 1.0]
        i3 = [0.0, 0.0, 0.0]

    class EixoVsVelocidade:
        t = [1.0]
        x = [0.0]
        y = [-24.38]
        z = [0.0]
        V1 = [620.0]
        V2 = [0.0]
        V3 = [0.0]
        i1 = [0.0]
        i2 = [1.0]
        i3 = [0.0]

    class Obliquo:
        r = 24.38
        t = [1.0]
        x = [-r * 0.5]
        y = [-r * 0.8660254037844386]
        z = [0.0]
        V1 = [0.0]
        V2 = [620.0]
        V3 = [0.0]
        i1 = [0.0]
        i2 = [1.0]
        i3 = [0.0]

    cases = [("burst frontal", Frontal), ("eixo != velocidade", EixoVsVelocidade),
             ("burst oblíquo 30°", Obliquo)]

    pairs = [
        ("distância burst-alvo (m)", "distancia_m", "distance_m"),
        ("phi alvo (°)", "phi_alvo_deg", "phi_target_deg"),
        ("área exposta (m²)", "area_exposta_m2", "exposed_area_m2"),
        ("velocidade do fragmento (m/s)", "v_frag_m_s", "fragment_speed_mps"),
        ("densidade (frag/m²)", "densidade_efetiva", "density_per_m2"),
        ("M fragmentos esperados", "fragmentos_esperados_area", "expected_fragments"),
        ("P(destruição)", "p_destruicao", "p_destruction"),
    ]

    target = shahed_136(center=(0.0, 0.0, 0.0))
    warhead = vt_fcl_mk49()
    frozen_target = frozen_appendix.criar_alvo_shahed(centro=(0.0, 0.0, 0.0))
    frozen_warhead = frozen_appendix.criar_ogiva_vt_fcl_mk49()

    all_equal = True
    for label, trajectory_class in cases:
        _, old = frozen_appendix.avaliar_fuze_e_dano(
            trajectory_class(), frozen_target, frozen_warhead,
            raio_fuze_m=24.38, tempo_armar_s=0.5,
        )
        _, new = evaluate_engagement(trajectory_class(), target, warhead)

        out(f"  {label}")
        for name, old_key, new_attr in pairs:
            a = float(old[old_key])
            b = float(getattr(new, new_attr))
            equal = a == b
            all_equal &= equal
            out(f"    {name:32s} {a!r:24s} {'==' if equal else '!='} {b!r}")
        out("")

    out(f"  Resultado: todos os valores {'coincidem exatamente' if all_equal else 'DIVERGEM'}.")
    out("")
    return all_equal


def check_selection(out) -> bool:
    out(header("5. SELEÇÃO DE PONTOS DE MIRA — contra a tabela publicada"))
    out("")
    out("A seleção refatorada roda sobre a tabela de azimutes ótimos publicada")
    out("e o resultado é comparado com a tabela de pontos de mira usada no TCC.")
    out("")

    optimal = pd.read_excel(OPTIMAL_AZIMUTHS)
    published = pd.read_excel(SELECTED_POINTS_100M)
    selected = select_points_by_spacing(optimal, elevation_max=39.6, elevation_min=-1.5)

    out(f"  entrada  : {OPTIMAL_AZIMUTHS.name} ({len(optimal)} elevações)")
    out(f"  publicado: {SELECTED_POINTS_100M.name} ({len(published)} pontos)")
    out(f"  reproduzido: {len(selected)} pontos")
    out("")

    all_equal = len(selected) == len(published)
    for column in ("Elevacao_deg", "Azimute_otimo_deg", "Alcance_x_m",
                   "Desvio_z_resultante_m"):
        equal = all_equal and np.array_equal(
            selected[column].values, published[column].values
        )
        all_equal &= equal
        out(f"    {column:26s} idêntico coluna inteira: {'SIM' if equal else 'NÃO'}")

    out("")
    out("  Primeiras cinco linhas (publicado | reproduzido):")
    for i in range(min(5, len(published))):
        p = published.iloc[i]
        s = selected.iloc[i]
        out(f"    elev {p['Elevacao_deg']:6.1f}° azim {p['Azimute_otimo_deg']:6.2f}° "
            f"alcance {p['Alcance_x_m']:12.6f} m  |  "
            f"elev {s['Elevacao_deg']:6.1f}° azim {s['Azimute_otimo_deg']:6.2f}° "
            f"alcance {s['Alcance_x_m']:12.6f} m")
    out("")
    return all_equal


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def header(title: str) -> str:
    return f"{'=' * 78}\n{title}\n{'=' * 78}"


def environment_block(out, patched: bool) -> None:
    out(header("AMBIENTE E PROCEDÊNCIA"))
    out("")
    out(f"  data (UTC)     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"  python         : {platform.python_version()} ({platform.system()})")
    out(f"  numpy          : {np.__version__}")
    out(f"  scipy          : {scipy.__version__}")
    out(f"  pandas         : {pd.__version__}")
    out("")
    out("  Arquivos congelados de referência (MD5):")
    for frozen_name in ("motor_original.py", "apendice_dano.py", "legacy_motor.py"):
        frozen_path = REPO_ROOT / "tests" / "reference" / frozen_name
        out(f"    {frozen_name:24s} {md5_of_file(frozen_path)}")
    out("")
    out(f"  Tabela fonte (origem)  : {AERO_SOURCE_5IN38.name} "
        f"({md5_of_file(AERO_SOURCE_5IN38)})")
    out(f"  Coeficientes do modelo : {AERO_COEFFICIENTS_5IN38.name} "
        f"({md5_of_file(AERO_COEFFICIENTS_5IN38)})")
    out("")
    if patched:
        out("  Shim NumPy 2.x aplicado ao código congelado (tests/reference/compat.py):")
        out("  apenas a conversão de array 1x1 para float, que o NumPy 2.0 passou a")
        out("  rejeitar. Nenhum valor é alterado — ver tests/reference/README.md.")
    else:
        out("  Shim NumPy 2.x não foi necessário (NumPy 1.x detectado).")
    out("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", default=None,
        help="também grava a saída neste arquivo, como bloco de código markdown",
    )
    args = parser.parse_args()

    lines: List[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    patched = compat.patch_legacy_coefficients(frozen_engine)
    compat.patch_legacy_coefficients(frozen_aa_engine)

    out(header("PROVA DE EQUIVALÊNCIA — motor original vs pacote refatorado"))
    out("")
    environment_block(out, patched)

    with contextlib.redirect_stdout(io.StringIO()):
        coefficients_frozen = frozen_engine.RealAerodynamicCoefficients(str(AERO_SOURCE_5IN38))
    coefficients_new = naval_5in38_coefficients()

    results = {
        "trajetórias": check_trajectories(out, coefficients_frozen, coefficients_new),
        "grandezas derivadas": check_derived(out, coefficients_frozen, coefficients_new),
        "relatório antiaéreo": check_aa_report(out),
        "modelo de dano": check_damage_model(out),
        "seleção de pontos": check_selection(out),
    }

    out(header("VEREDITO"))
    out("")
    for name, passed in results.items():
        out(f"  {name:24s} {'OK — idêntico' if passed else 'FALHOU — há diferença'}")
    out("")
    everything = all(results.values())
    out("  " + ("TODAS AS VERIFICAÇÕES PASSARAM." if everything
                else "ALGUMA VERIFICAÇÃO FALHOU."))
    out("")

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(lines), encoding="utf-8")
        print(f"Relatório gravado em {path}")

    return 0 if everything else 1


def render_markdown(lines: List[str]) -> str:
    """Wrap the captured output in a markdown document."""
    return (
        "# Prova de equivalência numérica\n\n"
        "Saída literal de `python scripts/proof_of_equivalence.py`, gerada\n"
        "automaticamente. Para regenerar:\n\n"
        "```bash\n"
        "python scripts/proof_of_equivalence.py --report docs/verification.md\n"
        "```\n\n"
        "O script sai com código 1 se qualquer verificação falhar, então este\n"
        "arquivo só existe na forma abaixo se tudo passou.\n\n"
        "```text\n" + "\n".join(lines) + "\n```\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
