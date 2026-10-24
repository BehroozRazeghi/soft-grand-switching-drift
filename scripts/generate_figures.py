from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["font.size"] = 7.2
mpl.rcParams["axes.labelsize"] = 7.8
mpl.rcParams["legend.fontsize"] = 6.4
mpl.rcParams["xtick.labelsize"] = 6.8
mpl.rcParams["ytick.labelsize"] = 6.8

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_FIGSIZE = (3.45, 1.90)
COMBINED_FIGSIZE = (7.16, 3.35)
OUTPUT_STEMS = (
    "fig_switching_pathwise",
    "fig_refresh_tracking",
    "fig_refresh_minimizer",
    "fig_tracked_decoding",
    "fig_dynamic_evaluation",
)
PDF_METADATA = {
    "CreationDate": None,
    "ModDate": None,
    "Creator": "Soft GRAND under Channel Switching and Drift",
    "Producer": "matplotlib",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="") as f:
        return list(csv.DictReader(f))


def _ci_from_se(se: float, reps: int) -> float:
    if reps <= 1:
        return 0.0
    return float(student_t.ppf(0.975, reps - 1) * se)


def _style_boxplot(bp, linestyle: str) -> None:
    for key in ("boxes", "whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_linestyle(linestyle)
            artist.set_linewidth(0.9)
    for flier in bp["fliers"]:
        flier.set_marker("x")
        flier.set_markersize(3.0)
        flier.set_markerfacecolor("none")


def _draw_switching_pathwise(ax, pathwise: list[dict[str, str]]) -> None:
    s_values = sorted({int(r["S"]) for r in pathwise})
    width = 0.18
    policies = (
        ("switching_excess_mean", -width, "-", "o"),
        ("initial_state_static_excess_mean", width, "--", "s"),
    )
    for key, offset, linestyle, marker in policies:
        values = []
        positions = []
        for switches in s_values:
            values.append([
                1e3 * float(r[key])
                for r in pathwise
                if int(r["S"]) == switches
            ])
            positions.append(switches + offset)
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.28,
            patch_artist=False,
            showfliers=True,
            showmeans=True,
            meanline=False,
            meanprops={
                "marker": marker,
                "markerfacecolor": "none",
                "markeredgecolor": "black",
                "markersize": 3.0,
            },
            manage_ticks=False,
        )
        _style_boxplot(bp, linestyle)
    ax.set_xticks(s_values)
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$10^3\widehat{\Delta P}_{\mathrm{e}}^{\pi}(s^n)$")
    ax.grid(True, alpha=0.22)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color="black", linestyle="-", marker="o",
                markerfacecolor="none", linewidth=0.9, markersize=3.0,
                label=r"$Q_{n,1}^{\mathrm{sw}}$",
            ),
            Line2D(
                [0], [0], color="black", linestyle="--", marker="s",
                markerfacecolor="none", linewidth=0.9, markersize=3.0,
                label=r"$Q_n^{\mathrm{init}}$",
            ),
        ],
        loc="upper left",
        frameon=True,
    )
    ax.tick_params(axis="both", pad=1.5)


def _draw_refresh_tracking(ax, scaling: list[dict[str, str]]) -> None:
    periods = sorted({int(r["period"]) for r in scaling}, reverse=True)
    selected = [p for p in (120, 60, 30) if p in periods]
    if len(selected) < min(3, len(periods)):
        selected = (
            [periods[0], periods[len(periods) // 2], periods[-1]]
            if len(periods) >= 3
            else periods
        )
    for period, marker, linestyle in zip(
        selected, ("o", "s", "^"), ("-", "--", "-.")
    ):
        rows = sorted(
            [r for r in scaling if int(r["period"]) == period],
            key=lambda r: int(r["q"]),
        )
        q = np.array([int(r["q"]) for r in rows])
        y = np.array([float(r["tracking_mean"]) for r in rows])
        se = np.array([float(r["tracking_se"]) for r in rows])
        reps = int(rows[0]["reps"])
        ci = np.array([_ci_from_se(value, reps) for value in se])
        ax.errorbar(
            q, y, yerr=ci, marker=marker, linestyle=linestyle,
            capsize=1.8, linewidth=0.9, markersize=3.7,
            label=rf"$T={period}$",
        )
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\widehat{\bar e}_{\vartheta}(q;T)$")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=True)
    ax.tick_params(axis="both", pad=1.5)


def _draw_refresh_minimizer(ax, summary: list[dict[str, str]]) -> None:
    variation = np.array([float(r["v"]) for r in summary])
    q_empirical = np.array([float(r["q_empirical"]) for r in summary])
    q_star = np.array([float(r["q_star"]) for r in summary])
    q_low = np.array([
        float(r["q_empirical_bootstrap_low95"]) for r in summary
    ])
    q_high = np.array([
        float(r["q_empirical_bootstrap_high95"]) for r in summary
    ])
    x = 100.0 * variation
    order = np.argsort(x)
    lower = q_empirical - q_low
    upper = q_high - q_empirical
    ax.errorbar(
        x[order],
        q_empirical[order],
        yerr=np.vstack([lower[order], upper[order]]),
        marker="o",
        linestyle="-",
        capsize=2.0,
        linewidth=0.9,
        markersize=3.8,
        label=r"$q_{\mathrm{grid}}^\star$",
    )
    ax.plot(
        x[order],
        q_star[order],
        marker="s",
        linestyle="--",
        linewidth=0.95,
        markersize=3.8,
        label=r"$q^\star$",
    )
    ax.set_xlabel(r"$10^2 v$")
    ax.set_ylabel(r"$q$")
    if len(x) <= 12:
        ax.set_xticks(x[order])
        ax.set_xticklabels([f"{value:.1f}" for value in x[order]])
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", frameon=True)
    ax.tick_params(axis="both", pad=1.5)


def _draw_tracked_decoding(ax, pilot: list[dict[str, str]]) -> None:
    available_periods = sorted({int(r["period"]) for r in pilot})
    selected = [p for p in (120, 40) if p in available_periods]
    if len(selected) < 2 and available_periods:
        selected = [max(available_periods), min(available_periods)]
    curve_handles = []
    for period, marker, linestyle in zip(selected[:2], ("o", "s"), ("-", "--")):
        rows = sorted(
            [r for r in pilot if int(r["period"]) == period],
            key=lambda r: int(r["q"]),
        )
        q = np.array([int(r["q"]) for r in rows])
        excess = 1e3 * np.array([float(r["tracked_excess_mean"]) for r in rows])
        ci = 1e3 * np.array([float(r["tracked_excess_ci95"]) for r in rows])
        plotted = ax.errorbar(
            q, excess, yerr=ci, marker=marker, linestyle=linestyle,
            capsize=1.8, linewidth=0.9, markersize=3.7,
        )
        color = plotted[0].get_color()
        initial = 1e3 * float(rows[0]["true_initial_held_excess_mean"])
        initial_ci = 1e3 * float(rows[0]["true_initial_held_excess_ci95"])
        ax.axhline(initial, linestyle=":", linewidth=0.85, color=color, alpha=0.9)
        ax.axhspan(
            initial - initial_ci,
            initial + initial_ci,
            color=color,
            alpha=0.075,
            linewidth=0,
        )
        q_star = float(rows[0]["q_star"])
        nearest = int(np.argmin(np.abs(q - q_star)))
        q_near = int(q[nearest])
        y_near = float(excess[nearest])
        ax.axvline(q_star, linestyle=(0, (2, 2)), linewidth=0.75, color=color, alpha=0.75)
        ax.plot(q_near, y_near, marker="*", linestyle="none", color=color, markersize=6.0)
        if period == max(selected[:2]):
            xytext = (3, 6)
            valign = "bottom"
        else:
            xytext = (3, -7)
            valign = "top"
        ax.annotate(
            rf"$q^\star={q_star:.2f},\ {y_near:.2f}$",
            xy=(q_near, y_near), xytext=xytext, textcoords="offset points",
            fontsize=5.9, ha="left", va=valign, color=color,
        )
        curve_handles.append(
            Line2D(
                [0], [0], color=color, marker=marker, linestyle=linestyle,
                linewidth=0.9, markersize=3.7, label=rf"$T={period}$",
            )
        )
    curve_handles.append(
        Line2D(
            [0], [0], color="black", linestyle=":", linewidth=0.9,
            label=r"$Q_{b,n}^{\mathrm{init}}$",
        )
    )
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$10^3\widehat{\Delta P}_{\mathrm{e}}^{\mathrm{tr}}(q;T)$")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22)
    ax.legend(handles=curve_handles, loc="best", frameon=True)
    ax.tick_params(axis="both", pad=1.5)


def _save_single(draw, data, stem: str) -> None:
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    draw(ax, data)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _save_combined(pathwise, scaling, summary, pilot) -> None:
    fig, axes = plt.subplots(2, 2, figsize=COMBINED_FIGSIZE)
    drawers = (
        (_draw_switching_pathwise, pathwise),
        (_draw_refresh_tracking, scaling),
        (_draw_refresh_minimizer, summary),
        (_draw_tracked_decoding, pilot),
    )
    for label, ax, (draw, data) in zip("abcd", axes.ravel(), drawers):
        draw(ax, data)
        ax.text(
            -0.135, 1.035, f"({label})", transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", ha="left", va="bottom",
        )
    fig.tight_layout(pad=0.3, w_pad=0.85, h_pad=0.55)
    stem = "fig_dynamic_evaluation"
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def generate() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pathwise = read_csv("switching_channel_pathwise.csv")
    scaling = read_csv("refresh_scaling.csv")
    summary = read_csv("refresh_scaling_summary.csv")
    pilot = read_csv("pilot_drift.csv")
    q_star_by_period = {int(r["period"]): float(r["q_star"]) for r in summary}
    for row in pilot:
        period = int(row["period"])
        if period in q_star_by_period:
            row["q_star"] = q_star_by_period[period]

    _save_single(_draw_switching_pathwise, pathwise, "fig_switching_pathwise")
    _save_single(_draw_refresh_tracking, scaling, "fig_refresh_tracking")
    _save_single(_draw_refresh_minimizer, summary, "fig_refresh_minimizer")
    _save_single(_draw_tracked_decoding, pilot, "fig_tracked_decoding")
    _save_combined(pathwise, scaling, summary, pilot)

    print("Generated four standalone figures and the reproducible four-panel figure:")
    for stem in OUTPUT_STEMS:
        print(f"  {FIG_DIR / (stem + '.pdf')}")
        print(f"  {FIG_DIR / (stem + '.png')}")


if __name__ == "__main__":
    generate()
