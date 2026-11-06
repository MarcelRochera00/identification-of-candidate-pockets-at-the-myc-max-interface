#!/usr/bin/env python3

"""
Pocket Characterization
==============================================
Parses, cleans, and visualizes per-frame MDpocket descriptor outputs 
across multiple sub-pocket directories (SP1, SP2, etc.).

For each pocket, this script:
    1. Parses the descriptor file (e.g. SP1_descriptors.txt).
    2. Calculates the physical time axis (time_ns) based on snapshot indices.
    3. Exports per-pocket cleaned CSV data and summary statistics.

After processing all sub-pockets, it compiles and generates:
    - Combined multi-panel figures for primary descriptors (Volume, 
      Hydrophobicity, Hydrophobic Density, Polarity), overlaying raw 
      per-frame traces, rolling means, and overall averages.
    - A grand summary grid figure comparing all descriptors across all pockets.
    - A compact summary table displaying mean values and detection frequencies.
    - A long-format combined CSV summary (pocket_summary_statistics.csv).
    - A comprehensive text report (pocket_descriptor_report.txt) covering 
      metrics both including and excluding volume-0 (undetected) frames.
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ============================================================
# >>> CONFIG - the only section you should need to touch <<<
# ============================================================

# Folder that directly contains the SP1, SP2, ... SPn pocket subfolders.
ROOT_DIR = Path("Results/Pockets")

# Glob pattern used to discover pocket subfolders under ROOT_DIR.
POCKET_DIR_PATTERN = "SP*"

# Suffix of the descriptor filename (as written by mdpocket) inside each
# pocket folder. The filename actually looked up is "<pocket_name>" +
# DESCRIPTOR_SUFFIX (e.g. "SP1_descriptors.txt" inside the SP1 folder).
DESCRIPTOR_SUFFIX = "_descriptors.txt"

# Name of the folder (created under ROOT_DIR) that all outputs go into.
OUTPUT_DIRNAME = "sub_pockets_analysis"

# Trajectory timing: time_ns = snapshot_index * TIME_PER_SNAPSHOT_NS.
TIME_PER_SNAPSHOT_NS = 0.05

# Rolling-mean smoothing window, in nanoseconds (converted to a frame
# count using TIME_PER_SNAPSHOT_NS).
ROLLING_WINDOW_NS = 7.5

# If True, frames where the pocket was NOT detected (pock_volume == 0) are
# excluded from the figures, the CSV statistics and the means table.
# (The text report ALWAYS shows both versions, regardless of this flag, and
# the raw per-frame CSV always keeps every parsed row.)
EXCLUDE_UNDETECTED_FRAMES_FROM_STATS = False

# Figure DPI for saved plots.
FIGURE_DPI = 600

# Figure 4.6 (means table): if True, descriptor means are computed over the
# frames where the pocket was detected (pock_volume > 0), so they aren't diluted
# by closed frames (that is what the detection-frequency column reports).
# If False, all frames are used.
TABLE_MEANS_DETECTED_ONLY = True

# Per-frame (raw) trace drawn behind the rolling mean in the combined figures.
RAW_COLOR = "#9A9A9A"
RAW_LINEWIDTH = 0.5
RAW_ALPHA = 0.6

# ============================================================
# End of CONFIG
# ============================================================

EXPECTED_COLUMNS = [
    "snapshot", "pock_volume", "pock_asa", "pock_pol_asa", "pock_apol_asa",
    "pock_asa22", "pock_pol_asa22", "pock_apol_asa22", "nb_AS", "mean_as_ray",
    "mean_as_solv_acc", "apol_as_prop", "mean_loc_hyd_dens",
    "hydrophobicity_score", "volume_score", "polarity_score", "charge_score",
    "prop_polar_atm", "as_density", "as_max_dst",
]


class DescriptorPlot(NamedTuple):
    column: str
    title: str
    ylabel: str
    color: str
    filename_stub: str


PLOTS = [
    DescriptorPlot("pock_volume", "Pocket Volume", "Volume (\u00c5\u00b3)",
                    "#2E86AB", "4.1_combined_pocket_volume"),
    DescriptorPlot("hydrophobicity_score", "Hydrophobicity Score", "Hydrophobicity score",
                    "#A23B72", "4.2_combined_hydrophobicity_score"),
    DescriptorPlot("mean_loc_hyd_dens", "Mean Local Hydrophobic Density", "Density",
                    "#F18F01", "4.3_combined_mean_loc_hyd_dens"),
    DescriptorPlot("polarity_score", "Polarity Score", "Polarity score",
                    "#3B7A57", "4.4_combined_polarity_score"),
]


class PocketResult(NamedTuple):
    """Everything produced for one successfully-processed pocket."""
    name: str
    df: pd.DataFrame
    stats: pd.DataFrame


class Diagnostics:
    """Accumulates the parser diagnostics for one pocket."""

    def __init__(self, pocket_name: str):
        self.pocket_name = pocket_name
        self.total_lines = 0
        self.n_parsed_rows = 0
        self.n_short_rows = 0
        self.malformed: list[tuple[int, int, str]] = []
        self.n_zero_volume = 0
        self.n_total_rows = 0
        self.mean_volume_incl = float("nan")
        self.mean_volume_excl = float("nan")

    @property
    def pct_undetected(self) -> float:
        if self.n_total_rows == 0:
            return float("nan")
        return 100 * self.n_zero_volume / self.n_total_rows

    def print_report(self) -> None:
        print(f"[info] Total lines in file (incl. header): {self.total_lines}")
        print(f"[info] Successfully parsed data rows: {self.n_parsed_rows} "
              f"({self.n_short_rows} of these were short 'pocket not detected' rows, "
              f"kept as pock_volume=0)")
        if self.malformed:
            print(f"[warning] {len(self.malformed)} line(s) could not be parsed and were "
                  f"SKIPPED (not silently averaged in, not silently coerced):")
            for lineno, nfields, content in self.malformed[:10]:
                print(f"    line {lineno}: expected fields mismatch, got {nfields} -> {content!r}")
            if len(self.malformed) > 10:
                print(f"    ... and {len(self.malformed) - 10} more.")
        if self.n_total_rows:
            print(f"[info] Frames with pock_volume == 0 (pocket not detected): "
                  f"{self.n_zero_volume} / {self.n_total_rows} ({self.pct_undetected:.1f}%)")
            print(f"[info] Mean pock_volume INCLUDING zero/undetected frames: {self.mean_volume_incl:.2f}")
            print(f"[info] Mean pock_volume EXCLUDING zero/undetected frames: {self.mean_volume_excl:.2f}")


def natural_sp_sort_key(path: Path):
    """Sort pocket folders numerically (SP2 before SP10)."""
    match = re.search(r"(\d+)\s*$", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def discover_pocket_dirs(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root}")
    dirs = [p for p in root.glob(pattern) if p.is_dir()]
    return sorted(dirs, key=natural_sp_sort_key)


def find_descriptor_file(folder: Path, pocket_name: str, suffix: str) -> Path | None:
    """Locate "<pocket_name><suffix>" inside a pocket folder. Returns None
    (rather than raising) so the batch can continue."""
    expected_name = f"{pocket_name}{suffix}"
    candidate = folder / expected_name
    if candidate.exists():
        return candidate
    matches = list(folder.rglob(expected_name))
    return matches[0] if matches else None


def load_descriptors(filepath: Path, pocket_name: str) -> tuple[pd.DataFrame, Diagnostics]:
    """
    Manually parse an mdpocket descriptors file line by line. Every non-blank
    line after the header is either parsed into a row, or recorded as
    malformed. Short rows (pocket not detected that frame) are kept, with
    pock_volume forced to 0.0 and every other column set to NaN.
    """
    diag = Diagnostics(pocket_name)

    with open(filepath, "r") as fh:
        raw_lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    if not raw_lines:
        raise ValueError(f"{filepath} is empty.")
    diag.total_lines = len(raw_lines)

    header_tokens = raw_lines[0].split()
    if header_tokens[0] == "#":
        header_tokens = header_tokens[1:]
    header = header_tokens
    n_cols = len(header)

    if header != EXPECTED_COLUMNS:
        print(f"[warning] [{pocket_name}] Header does not exactly match the expected mdpocket "
              f"column order. Proceeding with the header as found in the file.")

    rows = []
    for i, line in enumerate(raw_lines[1:], start=2):
        fields = line.split()

        if len(fields) == n_cols:
            try:
                row = [float(f) for f in fields]
            except ValueError:
                diag.malformed.append((i, len(fields), line))
                continue
            rows.append(row)

        elif len(fields) < n_cols:
            try:
                partial = [float(f) for f in fields]
            except ValueError:
                diag.malformed.append((i, len(fields), line))
                continue
            row = partial + [float("nan")] * (n_cols - len(partial))
            if len(partial) < 2:
                row[1] = 0.0
            rows.append(row)
            diag.n_short_rows += 1

        else:
            diag.malformed.append((i, len(fields), line))

    diag.n_parsed_rows = len(rows)

    df = pd.DataFrame(rows, columns=header)
    if "snapshot" in df.columns:
        df = df.sort_values("snapshot").reset_index(drop=True)

    if "pock_volume" in df.columns:
        diag.n_total_rows = len(df)
        diag.n_zero_volume = int((df["pock_volume"] == 0).sum())
        diag.mean_volume_incl = df["pock_volume"].mean()
        if diag.n_zero_volume < diag.n_total_rows:
            diag.mean_volume_excl = df.loc[df["pock_volume"] > 0, "pock_volume"].mean()

    return df, diag


def add_time_column(df: pd.DataFrame, time_per_snapshot_ns: float) -> pd.DataFrame:
    df = df.copy()
    df.insert(1, "time_ns", df["snapshot"] * time_per_snapshot_ns)
    return df


def summarize(df: pd.DataFrame, exclude_undetected: bool) -> pd.DataFrame:
    """Per-descriptor mean/std/min/max/median for one pocket."""
    data = df
    if exclude_undetected and "pock_volume" in df.columns:
        data = df.loc[df["pock_volume"] > 0]
    numeric_cols = [c for c in data.columns if c not in ("snapshot", "time_ns")]
    stats = data[numeric_cols].agg(["mean", "std", "min", "max", "median"]).T
    stats.index.name = "descriptor"
    return stats.round(4)


def rolling_window_frames(rolling_window_ns: float, time_per_snapshot_ns: float) -> int:
    return max(1, round(rolling_window_ns / time_per_snapshot_ns))


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.3, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_trajectory(ax: plt.Axes, time_ns, y, color: str, window_frames: int) -> None:
    """Per-frame values (thin grey line) with the rolling mean on top."""
    ax.plot(time_ns, y, linewidth=RAW_LINEWIDTH, color=RAW_COLOR, alpha=RAW_ALPHA, zorder=1)
    rolled = pd.Series(y).rolling(window_frames, center=True, min_periods=1).mean()
    ax.plot(time_ns, rolled, linewidth=1.8, color=color, zorder=3, solid_capstyle="round")


def add_figure_legend(fig: plt.Figure, rolling_color: str, window_frames: int) -> None:
    handles = [
        Line2D([0], [0], color=RAW_COLOR, lw=1.5, label="per-frame value"),
        Line2D([0], [0], color=rolling_color, lw=2.0, label=f"rolling mean ({window_frames} frames)"),
        Line2D([0], [0], color="black", lw=1.3, ls="--", alpha=0.6, label="mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)


def grid_shape(n: int) -> tuple[int, int]:
    """Up to 3 columns per row (2x3 for 6 pockets)."""
    ncols = min(3, n) if n > 0 else 1
    nrows = math.ceil(n / ncols) if ncols else 1
    return nrows, ncols


def plot_combined_descriptor(results: list[PocketResult], spec: DescriptorPlot,
                              outpath: Path, exclude_undetected: bool,
                              window_frames: int) -> None:
    """One figure per descriptor, one panel per pocket, identical x/y limits
    across panels. Per-frame values in grey + rolling mean + mean line."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
    })

    series_by_pocket = {}
    for res in results:
        data = res.df
        if exclude_undetected and "pock_volume" in data.columns:
            data = data.loc[data["pock_volume"] > 0]
        if spec.column in data.columns:
            series_by_pocket[res.name] = (data["time_ns"].to_numpy(), data[spec.column].to_numpy())

    if not series_by_pocket:
        print(f"[warning] Descriptor '{spec.column}' not found in any pocket - skipping "
              f"combined figure {outpath.name}.")
        return

    all_x = pd.concat([pd.Series(t) for t, _ in series_by_pocket.values()])
    all_y = pd.concat([pd.Series(y) for _, y in series_by_pocket.values()])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min(skipna=True)), float(all_y.max(skipna=True))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    nrows, ncols = grid_shape(len(results))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, res in zip(axes, results):
        if res.name not in series_by_pocket:
            ax.set_title(f"{res.name}\n(data not found)", color="gray")
            ax.axis("off")
            continue

        time_ns, y = series_by_pocket[res.name]
        draw_trajectory(ax, time_ns, y, spec.color, window_frames)
        mean_val = float(pd.Series(y).mean())
        ax.axhline(mean_val, linestyle="--", linewidth=1.1, color="black", alpha=0.6, zorder=2)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(res.name)
        style_axis(ax)

    for ax in axes[len(results):]:
        ax.axis("off")

    for ax in axes[: len(results)][-ncols:]:
        ax.set_xlabel("Time (ns)")
    for row_start in range(0, len(axes), ncols):
        axes[row_start].set_ylabel(spec.ylabel)

    suffix = " (excl. undetected frames)" if exclude_undetected else ""
    fig.suptitle(f"{spec.title} Across Sub-Pockets{suffix}", fontsize=15, fontweight="bold")
    add_figure_legend(fig, spec.color, window_frames)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_all_descriptors_grid(results: list[PocketResult], specs: list[DescriptorPlot],
                               outpath: Path, exclude_undetected: bool,
                               window_frames: int) -> None:
    """One figure: rows = descriptors, columns = pockets. Y-limits shared per
    row, x-limits shared across the figure."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
    })

    filtered = {}
    for res in results:
        data = res.df
        if exclude_undetected and "pock_volume" in data.columns:
            data = data.loc[data["pock_volume"] > 0]
        filtered[res.name] = data

    all_x = pd.concat([data["time_ns"] for data in filtered.values()])
    x_min, x_max = float(all_x.min()), float(all_x.max())

    nrows, ncols = len(specs), len(results)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows),
                              sharex=True, squeeze=False)

    for i, spec in enumerate(specs):
        row_series = {name: data[spec.column] for name, data in filtered.items()
                       if spec.column in data.columns}
        if row_series:
            all_y = pd.concat(row_series.values())
            y_min, y_max = float(all_y.min(skipna=True)), float(all_y.max(skipna=True))
            y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        else:
            y_min = y_max = y_pad = 0.0

        for j, res in enumerate(results):
            ax = axes[i, j]
            data = filtered.get(res.name)
            if data is None or spec.column not in data.columns or data[spec.column].isna().all():
                ax.set_title("(no data)" if i == 0 else "", color="gray", fontsize=9)
                ax.axis("off")
                continue

            time_ns = data["time_ns"].to_numpy()
            y = data[spec.column].to_numpy()
            draw_trajectory(ax, time_ns, y, spec.color, window_frames)
            mean_val = float(pd.Series(y).mean())
            ax.axhline(mean_val, linestyle="--", linewidth=1.0, color="black", alpha=0.6, zorder=2)

            ax.set_xlim(x_min, x_max)
            if row_series:
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            style_axis(ax)

            if i == 0:
                ax.set_title(res.name)
            if j == 0:
                ax.set_ylabel(spec.ylabel)
            if i == nrows - 1:
                ax.set_xlabel("Time (ns)")

    suffix = " (excl. undetected frames)" if exclude_undetected else ""
    fig.suptitle(f"All Descriptors Across All Sub-Pockets{suffix}", fontsize=16, fontweight="bold")
    # rolling-mean colour differs per row, so the legend uses a neutral colour
    add_figure_legend(fig, "#333333", window_frames)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_pocket_means_table(results: list[PocketResult], specs: list[DescriptorPlot],
                             outpath: Path) -> None:
    """Figure 4.6: compact grid table, one row per sub-pocket: mean of each
    plotted descriptor (header in that descriptor's colour, body lightly
    tinted) + detection frequency (% of frames with pock_volume > 0).
    Computed straight from the per-frame data, independent of
    EXCLUDE_UNDETECTED_FRAMES_FROM_STATS."""
    slate, name_fill, line = "#2E4057", "#EEF1F5", "#B8BFCA"

    def header_label(spec: DescriptorPlot) -> str:
        title = textwrap.fill(spec.title, 20)
        unit = re.search(r"\(([^)]+)\)", spec.ylabel)
        return f"{title} ({unit.group(1)})" if unit else title

    def mean_of(df: pd.DataFrame, col: str) -> float:
        data = df.loc[df["pock_volume"] > 0] if TABLE_MEANS_DETECTED_ONLY else df
        return float(data[col].mean()) if col in data.columns else float("nan")

    name_w, val_w, det_w = 1.0, 1.8, 1.7
    head_h, row_h = 0.62, 0.36
    n_rows = len(results)
    W = name_w + len(specs) * val_w + det_w
    H = head_h + n_rows * row_h
    pad = 0.03

    fig = plt.figure(figsize=(W + 2 * pad, H + 2 * pad))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(H + pad, -pad)
    ax.axis("off")

    def cell(x, y, w, h, fill):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=line, linewidth=0.8))

    # columns: (x, width, header fill, body fill, header text)
    columns = [(0.0, name_w, slate, name_fill, "Sub-pocket")]
    x = name_w
    for spec in specs:
        columns.append((x, val_w, spec.color, to_rgba(spec.color, 0.12), header_label(spec)))
        x += val_w
    columns.append((x, det_w, slate, to_rgba(slate, 0.08), "Detection\nfrequency (%)"))

    for cx, cw, hfill, _, label in columns:
        cell(cx, 0, cw, head_h, hfill)
        ax.text(cx + cw / 2, head_h / 2, label, color="white", fontsize=10,
                fontweight="bold", ha="center", va="center", linespacing=1.2)

    for i, res in enumerate(results):
        ry = head_h + i * row_h
        yc = ry + row_h / 2
        n_total = len(res.df)
        frac = 100 * int((res.df["pock_volume"] > 0).sum()) / n_total if n_total else float("nan")
        values = [res.name]
        values += [f"{v:.2f}" if pd.notna(v) else "n/a"
                   for v in (mean_of(res.df, s.column) for s in specs)]
        values.append(f"{frac:.1f}" if pd.notna(frac) else "n/a")

        for (cx, cw, _, bfill, _), text in zip(columns, values):
            cell(cx, ry, cw, row_h, bfill)
            ax.text(cx + cw / 2, yc, text, fontsize=10.5, ha="center", va="center",
                    color=slate if cx == 0 else "#1F2328",
                    fontweight="bold" if cx == 0 else "normal")

    fig.savefig(outpath, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def _label_with_unit(spec: DescriptorPlot) -> str:
    m = re.search(r"\(([^)]+)\)", spec.ylabel)
    return f"{spec.title} ({m.group(1)})" if m else spec.title


def _fmt(x: float) -> str:
    return f"{x:.3f}" if pd.notna(x) else "n/a"


def _stats_table(data: pd.DataFrame, specs: list[DescriptorPlot]) -> list[str]:
    """Aligned text table (Mean / Std / Min / Max) for the plotted descriptors."""
    label_w = max(len(_label_with_unit(s)) for s in specs) + 2
    lines = [f"    {'Descriptor':<{label_w}}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}"]
    lines.append("    " + "-" * (label_w + 48))
    for spec in specs:
        label = _label_with_unit(spec)
        if spec.column not in data.columns or data[spec.column].dropna().empty:
            lines.append(f"    {label:<{label_w}}{'n/a':>12}{'n/a':>12}{'n/a':>12}{'n/a':>12}")
            continue
        s = data[spec.column]
        lines.append(f"    {label:<{label_w}}{_fmt(s.mean()):>12}{_fmt(s.std()):>12}"
                     f"{_fmt(s.min()):>12}{_fmt(s.max()):>12}")
    return lines


def write_readable_report(results: list[PocketResult], specs: list[DescriptorPlot],
                           outpath: Path) -> None:
    """Readable per-pocket report of the plotted descriptors, computed both
    INCLUDING and EXCLUDING the frames where pock_volume == 0 (pocket not
    detected). Independent of EXCLUDE_UNDETECTED_FRAMES_FROM_STATS.

    In the 'including' block, undetected frames count as 0 for every
    descriptor (not only volume), so the mean reflects the pocket being
    closed in those frames."""
    bar = "=" * 76
    lines = [
        "POCKET DESCRIPTOR REPORT",
        bar,
        "For each pocket, the 4 descriptors are summarised twice:",
        "  (A) INCLUDING frames where the pocket was not detected (pock_volume = 0;",
        "      these frames count as 0 for every descriptor)",
        "  (B) EXCLUDING those frames (detected frames only)",
        f"Time per snapshot: {TIME_PER_SNAPSHOT_NS} ns",
        "",
    ]

    for res in results:
        df = res.df
        n_total = len(df)
        undetected = df["pock_volume"] == 0
        n_zero = int(undetected.sum())
        n_det = n_total - n_zero
        pct = 100 * n_zero / n_total if n_total else float("nan")

        cols = [s.column for s in specs if s.column in df.columns]
        incl = df.copy()
        incl.loc[undetected, cols] = incl.loc[undetected, cols].fillna(0.0)
        excl = df.loc[~undetected]

        lines += [bar, res.name, bar,
                  f"  Frames: {n_total} total | {n_det} detected | "
                  f"{n_zero} with volume 0 ({pct:.1f}%)",
                  "",
                  f"  (A) Including volume-0 frames  (N = {n_total})"]
        lines += _stats_table(incl, specs)
        lines += ["", f"  (B) Excluding volume-0 frames  (N = {n_det})"]
        if n_det:
            lines += _stats_table(excl, specs)
        else:
            lines.append("    (pocket never detected - no frames left)")
        lines.append("")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_combined_summary(results: list[PocketResult]) -> pd.DataFrame:
    """Long-format table: one row per (pocket, descriptor)."""
    records = []
    for res in results:
        for descriptor, row in res.stats.iterrows():
            mean = row["mean"]
            std = row["std"]
            cv = (std / mean) if mean not in (0, float("nan")) and pd.notna(mean) else float("nan")
            records.append({
                "Pocket": res.name,
                "Descriptor": descriptor,
                "Mean": mean,
                "Median": row["median"],
                "Std": std,
                "Min": row["min"],
                "Max": row["max"],
                "Coefficient_of_Variation": cv,
            })
    return pd.DataFrame.from_records(records)


def process_pocket(pocket_dir: Path, output_root: Path) -> PocketResult | None:
    """Locate + parse the descriptor file, add the time column, write
    descriptors.csv / summary_statistics.csv. All figures are produced
    afterwards in the combined outputs."""
    pocket_name = pocket_dir.name
    print(f"\n{'=' * 70}\n[info] Processing {pocket_name}\n{'=' * 70}")

    descriptor_file = find_descriptor_file(pocket_dir, pocket_name, DESCRIPTOR_SUFFIX)
    if descriptor_file is None:
        print(f"[warning] No '{pocket_name}{DESCRIPTOR_SUFFIX}' found under {pocket_dir} - "
              f"skipping {pocket_name}.")
        return None

    print(f"[info] Reading: {descriptor_file}")
    df, diag = load_descriptors(descriptor_file, pocket_name)
    diag.print_report()

    df = add_time_column(df, TIME_PER_SNAPSHOT_NS)
    stats = summarize(df, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS)

    pocket_outdir = output_root / pocket_name
    pocket_outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(pocket_outdir / "descriptors.csv", index=False)
    stats.to_csv(pocket_outdir / "summary_statistics.csv")

    print(f"[info] Wrote: {pocket_outdir / 'descriptors.csv'}")
    print(f"[info] Wrote: {pocket_outdir / 'summary_statistics.csv'}")

    return PocketResult(name=pocket_name, df=df, stats=stats)


def main() -> None:
    output_root = ROOT_DIR / OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)

    pocket_dirs = discover_pocket_dirs(ROOT_DIR, POCKET_DIR_PATTERN)
    print(f"[info] Found {len(pocket_dirs)} pocket folder(s) under {ROOT_DIR} "
          f"matching '{POCKET_DIR_PATTERN}': {[p.name for p in pocket_dirs]}")

    results: list[PocketResult] = []
    missing_pockets: list[str] = []
    for pocket_dir in pocket_dirs:
        result = process_pocket(pocket_dir, output_root)
        if result is None:
            missing_pockets.append(pocket_dir.name)
        else:
            results.append(result)

    if not results:
        print("\n[error] No pockets were successfully processed - nothing to combine. "
              "Check ROOT_DIR and DESCRIPTOR_SUFFIX at the top of the script.")
        return

    print(f"\n{'=' * 70}\n[info] Building combined outputs across {len(results)} pocket(s)\n{'=' * 70}")

    combined_stats = compute_combined_summary(results)
    combined_csv = output_root / "pocket_summary_statistics.csv"
    combined_stats.to_csv(combined_csv, index=False)
    print(f"[info] Wrote: {combined_csv}")

    report_txt = output_root / "pocket_descriptor_report.txt"
    write_readable_report(results, PLOTS, report_txt)
    print(f"[info] Wrote: {report_txt}")

    window_frames = rolling_window_frames(ROLLING_WINDOW_NS, TIME_PER_SNAPSHOT_NS)
    for spec in PLOTS:
        outpath = output_root / f"{spec.filename_stub}.png"
        plot_combined_descriptor(results, spec, outpath, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS,
                                  window_frames)
        print(f"[info] Wrote: {outpath}")

    grand_outpath = output_root / "4.5_combined_all_descriptors.png"
    plot_all_descriptors_grid(results, PLOTS, grand_outpath, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS,
                               window_frames)
    print(f"[info] Wrote: {grand_outpath}")

    means_table_outpath = output_root / "4.6_pocket_means_table.png"
    plot_pocket_means_table(results, PLOTS, means_table_outpath)
    print(f"[info] Wrote: {means_table_outpath}")

    print(f"\n{'=' * 70}\n[done] Batch summary\n{'=' * 70}")
    print(f"  Pockets processed : {len(results)}  ({', '.join(r.name for r in results)})")
    if missing_pockets:
        print(f"  Pockets skipped (missing '<pocket>{DESCRIPTOR_SUFFIX}'): "
              f"{len(missing_pockets)}  ({', '.join(missing_pockets)})")
    else:
        print("  Pockets skipped   : none")
    print(f"  All outputs under : {output_root}")


if __name__ == "__main__":
    main()