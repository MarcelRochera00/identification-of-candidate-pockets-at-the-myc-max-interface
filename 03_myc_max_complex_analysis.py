#!/usr/bin/env python3

"""
MYC-MAX Complex Analysis Pipeline
=================================
Performs multi-replica structural analysis on protein-protein complexes, 
including interface contacts, MM-GBSA binding free energies, hotspot 
decomposition, and inter-chain salt bridge occupancies.

Pipeline Overview:
    1. Evaluates per-frame interface contact counts across replicas.
    2. Runs gmx_MMPBSA calculations to compute binding free energy and 
       per-residue energy contributions (hotspots).
    3. Calculates inter-chain salt bridge frequencies and occupancy thresholds.
    4. Outputs structured summary reports and machine-readable CSVs under Results/Complex/.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import subprocess
import datetime
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import distance_array

# =============================================================================
# SELECT ANALYSES TO RUN
# =============================================================================
print("\nSelect analyses to run (leave blank to run all):")
print("  1. Interface Contacts")
print("  2. MM-GBSA + Hotspot Decomposition")
print("  3. Salt Bridge Occupancy")
print("Enter numbers separated by commas (e.g. 1,3) or press Enter for all: ")
_selection = input().strip()

try:
    RUN = {1, 2, 3} if _selection == "" else set(int(x.strip()) for x in _selection.split(","))
except ValueError:
    print("  Invalid input — running all analyses.")
    RUN = {1, 2, 3}

print(f"  → Running: {', '.join(str(x) for x in sorted(RUN))}\n")

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PSF_FILE    = os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.psf")
CRYSTAL_PSF = os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.psf")
CRYSTAL_PDB = os.path.join(PROJECT_ROOT, "crystal_proteins.pdb")

TRAJS = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.0_proteins_only.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.0_proteins_only.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.0_proteins_only.xtc"),
]

EQUIL = [500, 1000, 1500]

TPR_FILES = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.1_production.tpr"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.1_production.tpr"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.1_production.tpr"),
]

FULL_TRAJS = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.0_production.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.0_production.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.0_production.xtc"),
]

TOPOL_FILES = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/topol.top"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/topol.top"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/topol.top"),
]

SEL_ALL      = "protein and name CA"
SEL_MYC      = "segid PROA and name CA"
SEL_MAX      = "segid PROB and name CA"
CONTACT_DIST = 4.5
OUT          = os.path.join(PROJECT_ROOT, "Results/Complex")
BASE_PATH    = os.path.join(PROJECT_ROOT, "MYC-MAX")
INDEX_FILE   = os.path.abspath(os.path.join(BASE_PATH, "MYC-MAX-1000/charmm-gui/gromacs/index-new.ndx"))

# Hotspot criterion for the decomposition plots and summary: residues whose
# per-residue binding contribution is <= this value (kcal/mol). Keep in sync
# with HOTSPOT_THRESHOLD in 06_multi_method_validation.py.
HOTSPOT_THRESHOLD = -2.0

# Skip the (multi-hour) gmx_MMPBSA run for a replica when its
# FINAL_RESULTS_MMPBSA.csv and DECOMP_RESULTS.csv already exist, and only
# re-parse / re-plot them. Set to False to force recomputation.
REUSE_EXISTING_MMPBSA = True

# Salt bridge parameters
SALT_DIST      = 4.0   # Å — standard geometric cutoff for salt bridges
SALT_OCCUPANCY = 0.30  # only report bridges present in >30% of frames

# =============================================================================

os.makedirs(OUT, exist_ok=True)
COLORS = ["#8ecae6", "#f4a340","#c8a2d9"]

def frames_to_ns(n_frames, ps_per_frame=50):
    return np.arange(n_frames) * ps_per_frame / 1000


# =============================================================================
# PARSE HELPERS
# =============================================================================

def parse_mmpbsa_csv(csv_file):
    """Parse gmx_MMPBSA CSV — extract TOTAL column from Delta Energy Terms section."""
    deltas = []
    in_delta = False
    header_next = False
    total_idx = None

    with open(csv_file) as f:
        for line in f:
            line = line.strip()
            if "Delta Energy Terms" in line:
                in_delta = True
                header_next = True
                continue
            if in_delta and header_next:
                cols = line.split(",")
                try:
                    total_idx = cols.index("TOTAL")
                except ValueError:
                    in_delta = False
                    continue
                header_next = False
                continue
            if in_delta and line == "":
                in_delta = False
                continue
            if in_delta and total_idx is not None:
                parts = line.split(",")
                try:
                    deltas.append(float(parts[total_idx]))
                except (ValueError, IndexError):
                    continue

    return np.array(deltas)


# Energy terms of the decomposition, in the column order written by gmx_MMPBSA.
DECOMP_TERMS = ["Internal", "vdW", "Electrostatic", "Polar", "NonPolar", "TOTAL"]


def parse_decomp_terms(csv_file):
    """
    Parse gmx_MMPBSA decomposition CSV, keeping every energy term.
    Format: Frame #,Residue,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL
    Residue label format: R:A:ARG:913 — chain A = MYC, chain B = MAX.

    The file holds 12 tables with identical headers: Complex / Receptor /
    Ligand / DELTAS, each split into Total / Sidechain / Backbone
    contributions. Only DELTAS → Total Decomposition Contribution (TDC) is
    the per-residue contribution to binding, so every other table is skipped.

    Returns {residue: array of the six DECOMP_TERMS, averaged over frames}.
    """
    residue_terms = {}
    block = None   # "Complex" | "Receptor" | "Ligand" | "DELTAS"
    sub   = None   # "Total" | "Sidechain" | "Backbone"

    with open(csv_file) as f:
        for line in f:
            line = line.strip().strip('"')
            if not line or line.startswith("Frame #,Residue"):
                continue
            if not line[0].isdigit():
                # Block / sub-block title line
                if line.endswith(":"):
                    block = line[:-1]
                    sub = None
                elif "Contribution" in line:
                    sub = line.split()[0]
                continue
            if block != "DELTAS" or sub != "Total":
                continue

            parts = line.split(",")
            if len(parts) < 8:
                continue

            try:
                int(parts[0])
            except ValueError:
                continue

            res_raw = parts[1].strip()
            try:
                terms = [float(x) for x in parts[2:8]]
            except (ValueError, IndexError):
                continue

            residue_terms.setdefault(res_raw, []).append(terms)

    return {k: np.mean(v, axis=0) for k, v in residue_terms.items()}


def parse_decomp_csv(csv_file):
    """Mean per-residue TOTAL contribution to binding (kcal/mol), per residue."""
    return {k: v[-1] for k, v in parse_decomp_terms(csv_file).items()}


def label_residue(res_raw):
    """
    Parse R:A:ARG:913 format into a clean label and protein name.
    Chain A = MYC (PROA), Chain B = MAX (PROB).
    """
    parts = res_raw.split(":")
    if len(parts) >= 4:
        chain   = parts[1].upper()
        resname = parts[2]
        resid   = parts[3]
        protein = "MYC" if chain == "A" else "MAX" if chain == "B" else "?"
        label   = f"{protein}:{resname}{resid}"
    else:
        protein = "?"
        label   = res_raw
    return label, protein

def display_residue(res_raw):
    """R:A:LEU:967 -> 'LEU967' for figure axis labels."""
    label, _ = label_residue(res_raw)
    return label.split(":")[1]


# =============================================================================
# 1. Interface Contacts
# =============================================================================
if 1 in RUN:
    print("\n[1/3] Calculating interface contacts...")
    try:
        fig, ax = plt.subplots(figsize=(13, 5))
        all_contacts = []

        for i, (traj, eq) in enumerate(zip(TRAJS, EQUIL)):
            try:
                u    = mda.Universe(PSF_FILE, traj)
                myc  = u.select_atoms("segid PROA and not name H*")
                max_ = u.select_atoms("segid PROB and not name H*")

                contacts = []
                for ts in u.trajectory:
                    dists      = distance_array(myc.positions, max_.positions)
                    n_contacts = int((dists < CONTACT_DIST).sum())
                    contacts.append(n_contacts)

                time     = frames_to_ns(len(contacts))
                contacts = np.array(contacts)
                all_contacts.append(contacts)

                ax.plot(time, contacts, color=COLORS[i], linewidth=0.8, alpha=0.8,
                        label=f"Replica {i+1}  mean: {contacts.mean():.0f}")
                print(f"  Replica {i+1} — mean contacts: {contacts.mean():.1f} ± {contacts.std():.1f}")
            except Exception as e:
                print(f"  ✗ Replica {i+1} contacts failed: {e}")
                continue

        if all_contacts:
            min_len       = min(len(c) for c in all_contacts)
            mean_contacts = np.mean([c[:min_len] for c in all_contacts], axis=0)
            ax.plot(frames_to_ns(min_len), mean_contacts, color="#2b2d42", linewidth=1.5,
                    linestyle="--", label="Mean across replicas", zorder=5)

        ax.set_xlabel("Time (ns)", fontsize=11)
        ax.set_ylabel(f"Interface contacts (< {CONTACT_DIST} Å)", fontsize=11)
        #ax.set_title("MYC-MAX Interface Contacts Over Time (3 replicas)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/3.1_interface_contacts.png", dpi=150)
        plt.close()
        print(f"  → Saved {OUT}/3.1_interface_contacts.png")

        # 3.2_contacts_summary.png generation removed per request.
        # (Mean/std per replica is still printed above.)
    except Exception as e:
        print(f"  ✗ Contacts section failed: {e}")


# =============================================================================
# 2. MM-GBSA + Hotspot Decomposition
# =============================================================================
if 2 in RUN:
    print("\n[2/3] Running MM-GBSA + Hotspot Decomposition...")

    # KEY FIXES vs previous version:
    #   igb=2    — GBHCT model; validated for charged PPIs. igb=5 (OBC2) was
    #              over-estimating electrostatics at the MYC-MAX interface.
    #   idecomp=1 — avoids 1-4 pair double-counting that inflates per-residue
    #              energies with idecomp=2.
    #   interval=50 — reduces correlated-frame artefacts. 10 was too dense.

    MMPBSA_INPUT = """&general
startframe=1,
interval=50,
/

&gb
igb=2,
saltcon=0.150,
/

&decomp
idecomp=1,
dec_verbose=1,
/
"""

    mmpbsa_results    = []
    mmpbsa_pb_results = []   # PB cross-validation
    decomp_results    = []
    decomp_terms      = []   # per replica: {residue: array of DECOMP_TERMS}

    for i, (tpr, xtc, topol, eq) in enumerate(zip(TPR_FILES, TRAJS, TOPOL_FILES, EQUIL)):
        print(f"\n  Replica {i+1} (eq {eq}ps) started at {datetime.datetime.now().strftime('%H:%M:%S')}")
        sys.stdout.flush()

        rep_dir = os.path.abspath(f"{OUT}/mmpbsa_rep{i+1}")
        os.makedirs(rep_dir, exist_ok=True)

        inp_file = f"{rep_dir}/mmpbsa.in"
        with open(inp_file, "w") as f:
            f.write(MMPBSA_INPUT)

        cmd = [
            "mpirun", "-np", "8", "--allow-run-as-root",
            "gmx_MMPBSA", "MPI",
            "-O",
            "-i",  inp_file,
            "-cs", os.path.abspath(tpr),
            "-ct", os.path.abspath(xtc),
            "-ci", INDEX_FILE,
            "-cg", "16", "17",
            "-cp", os.path.abspath(topol),
            "-o",  f"{rep_dir}/FINAL_RESULTS_MMPBSA.dat",
            "-eo", f"{rep_dir}/FINAL_RESULTS_MMPBSA.csv",
            "-deo", f"{rep_dir}/DECOMP_RESULTS.csv",
        ]

        reuse = REUSE_EXISTING_MMPBSA and all(
            os.path.exists(f"{rep_dir}/{name}")
            for name in ("FINAL_RESULTS_MMPBSA.csv", "DECOMP_RESULTS.csv")
        )

        try:
            if reuse:
                print(f"  ↺ Reusing existing gmx_MMPBSA output in {rep_dir} (no recomputation)")
                run_stdout = run_stderr = ""
            else:
                print(f"  Command: {' '.join(cmd)}")
                sys.stdout.flush()

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=25200,
                                        env=os.environ.copy())
                print(f"  Finished at {datetime.datetime.now().strftime('%H:%M:%S')} "
                      f"(return code {result.returncode})")
                run_stdout = result.stdout or ""
                run_stderr = result.stderr or ""

                # --- Always dump the full stdout/stderr to disk, win or lose,
                # so the actual gmx_MMPBSA/mpirun output is available to read
                # directly instead of relying on truncated console prints.
                stdout_log = f"{rep_dir}/mmpbsa_stdout.log"
                stderr_log = f"{rep_dir}/mmpbsa_stderr.log"
                with open(stdout_log, "w") as f:
                    f.write(run_stdout)
                with open(stderr_log, "w") as f:
                    f.write(run_stderr)
                print(f"  Full logs saved: {stdout_log}")
                print(f"                   {stderr_log}")

            # --- GB binding energy ---
            csv_file = f"{rep_dir}/FINAL_RESULTS_MMPBSA.csv"
            if os.path.exists(csv_file):
                data = parse_mmpbsa_csv(csv_file)
                if len(data) > 0:
                    mmpbsa_results.append(data)
                    mean_val = data.mean()
                    print(f"  ✓ Replica {i+1} — ΔG (GB): {mean_val:.2f} ± {data.std():.2f} kcal/mol", end="")
                    # Sanity check: expected range for a PPI is roughly -20 to -50 kcal/mol
                    if mean_val < -80:
                        print(f"  ⚠ WARNING: value < -80 suggests GB is still overcounting. "
                              f"Check PB result and consider raising interval further.")
                    elif mean_val > 0:
                        print(f"  ⚠ WARNING: positive ΔG — check index groups and topology.")
                    else:
                        print()
                else:
                    print(f"  ✗ Replica {i+1} — CSV exists but no Delta data found")
                    print(f"  --- stdout tail ---\n{run_stdout[-2000:]}")
                    print(f"  --- stderr tail ---\n{run_stderr[-2000:]}")
            else:
                print(f"  ✗ Replica {i+1} — no CSV generated")
                print(f"  --- stdout tail ---\n{run_stdout[-2000:]}")
                print(f"  --- stderr tail ---\n{run_stderr[-2000:]}")

            # --- Per-residue decomposition ---
            decomp_file = f"{rep_dir}/DECOMP_RESULTS.csv"
            if os.path.exists(decomp_file):
                terms  = parse_decomp_terms(decomp_file)
                decomp = {k: v[-1] for k, v in terms.items()}
                if decomp:
                    decomp_results.append(decomp)
                    decomp_terms.append(terms)
                    top = sorted(decomp.items(), key=lambda x: x[1])[:5]
                    print(f"  ✓ Decomp top 5 hotspots: " +
                          ", ".join(f"{label_residue(r)[0]}={v:.2f}" for r, v in top))
                else:
                    print(f"  ✗ Replica {i+1} — decomp CSV empty or unparseable")
            else:
                print(f"  ✗ Replica {i+1} — no decomp CSV generated")

        except FileNotFoundError:
            print(f"  ✗ Replica {i+1} — gmx_MMPBSA not found, is gromacs env active?")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Replica {i+1} — timed out after 7 hours")
            csv_file = f"{rep_dir}/FINAL_RESULTS_MMPBSA.csv"
            if os.path.exists(csv_file):
                data = parse_mmpbsa_csv(csv_file)
                if len(data) > 0:
                    mmpbsa_results.append(data)
                    print(f"  ~ Partial results — ΔG (GB): {data.mean():.2f} ± {data.std():.2f} kcal/mol")
            decomp_file = f"{rep_dir}/DECOMP_RESULTS.csv"
            if os.path.exists(decomp_file):
                terms  = parse_decomp_terms(decomp_file)
                decomp = {k: v[-1] for k, v in terms.items()}
                if decomp:
                    decomp_results.append(decomp)
                    decomp_terms.append(terms)
        except Exception as e:
            print(f"  ✗ Replica {i+1} — unexpected error: {e}")

        for f in glob.glob(os.path.join(PROJECT_ROOT, "_GMXMMPBSA_*")):
            try:
                os.remove(f)
            except:
                pass

        sys.stdout.flush()

    # -------------------------------------------------------------------------
    # Overall ΔG binding per replica — 3.3_mmgbsa.png generation removed per
    # request. Still report the overall mean/std to the console.
    # -------------------------------------------------------------------------
    if mmpbsa_results:
        overall_mean = np.mean([r.mean() for r in mmpbsa_results])
        overall_std  = np.std([r.mean()  for r in mmpbsa_results])
        print(f"\n  Overall ΔG (GB): {overall_mean:.2f} ± {overall_std:.2f} kcal/mol")
    else:
        print("  ✗ No MM-GBSA results")

    # -------------------------------------------------------------------------
    # PLOT 2 — Per-replica hotspot bar charts
    # -------------------------------------------------------------------------
    for i, decomp in enumerate(decomp_results):
        if not decomp:
            continue

        sorted_res = sorted(decomp.items(), key=lambda x: x[1])
        top        = [(r, v) for r, v in sorted_res if v <= HOTSPOT_THRESHOLD]
        labels     = [r for r, v in top]
        values     = [v for r, v in top]
        colors_bar = []
        for lbl in labels:
            _, prot = label_residue(lbl)
            colors_bar.append("#e74c3c" if prot == "MYC" else "#3498db" if prot == "MAX" else "#95a5a6")

        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.3)))
        ax.barh(range(len(labels)), values, color=colors_bar, edgecolor="k", linewidth=0.5)
        ax.axvline(HOTSPOT_THRESHOLD, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([display_residue(l) for l in labels], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("ΔG contribution (kcal/mol)", fontsize=11)
        ax.set_title(f"MM-GBSA Hotspot Residues — Replica {i+1} (eq {EQUIL[i]}ps)\n"
                     f"{len(labels)} residues with ΔG ≤ {HOTSPOT_THRESHOLD} kcal/mol",
                     fontsize=12, fontweight="bold")
        ax.legend(handles=[mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                            mpatches.Patch(color="#3498db", label="MAX (PROB)"),
                            mpatches.Patch(color="red", label=f"Threshold ({HOTSPOT_THRESHOLD} kcal/mol)",
                                           linestyle="--", fill=False)], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out_path = f"{OUT}/3.4_hotspots_rep{i+1}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  → Saved {out_path}")

    # -------------------------------------------------------------------------
    # PLOT 3 — Consensus hotspots across ALL replicas
    # -------------------------------------------------------------------------
    if len(decomp_results) > 1:
        all_residues = set()
        for d in decomp_results:
            all_residues.update(d.keys())

        consensus = {}
        for res in all_residues:
            vals = [d[res] for d in decomp_results if res in d]
            if len(vals) == len(decomp_results):
                consensus[res] = (np.mean(vals), np.std(vals))

        if consensus:
            sorted_c = sorted(consensus.items(), key=lambda x: x[1][0])
            top_c    = [(r, v) for r, v in sorted_c if v[0] <= HOTSPOT_THRESHOLD]
            labels_c = [r for r, _ in top_c]
            means_c  = [v[0] for _, v in top_c]
            stds_c   = [v[1] for _, v in top_c]
            colors_c = []
            for lbl in labels_c:
                _, prot = label_residue(lbl)
                colors_c.append("#e74c3c" if prot == "MYC" else "#3498db" if prot == "MAX" else "#95a5a6")

            fig, ax = plt.subplots(figsize=(12, max(6, len(labels_c) * 0.3)))
            ax.barh(range(len(labels_c)), means_c, xerr=stds_c,
                    color=colors_c, edgecolor="k", linewidth=0.5,
                    error_kw={"elinewidth": 1.2, "capsize": 3})
            ax.axvline(HOTSPOT_THRESHOLD, color="red", linestyle="--", linewidth=1)
            ax.set_yticks(range(len(labels_c)))
            ax.set_yticklabels([display_residue(l) for l in labels_c], fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Mean ΔG contribution (kcal/mol)", fontsize=11)
            # Title removed per request.
            ax.legend(handles=[mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                                mpatches.Patch(color="#3498db", label="MAX (PROB)"),
                                mpatches.Patch(color="red", label=f"Threshold ({HOTSPOT_THRESHOLD} kcal/mol)",
                                               linestyle="--", fill=False)], fontsize=9)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUT}/3.5_hotspots_consensus.png", dpi=150)
            plt.close()
            print(f"  → Saved {OUT}/3.5_hotspots_consensus.png")

            summary_path = f"{OUT}/3.0_hotspots_summary.txt"
            with open(summary_path, "w") as f:
                f.write("Consensus MM-GBSA Hotspot Residues — MYC-MAX\n")
                f.write(f"Replicas: {len(decomp_results)} | {len(top_c)} residues with mean "
                        f"ΔG <= {HOTSPOT_THRESHOLD} kcal/mol (of {len(sorted_c)} present in all replicas)\n")
                f.write(f"Settings: igb=2, idecomp=1, interval=50\n")
                f.write("="*60 + "\n")
                f.write(f"{'Residue':<25} {'Mean (kcal/mol)':>18} {'Std':>10} {'Protein':>10}\n")
                f.write("-"*60 + "\n")
                for res, (mean_v, std_v) in top_c:
                    _, prot = label_residue(res)
                    f.write(f"{res:<25} {mean_v:>18.2f} {std_v:>10.2f} {prot:>10}\n")
            print(f"  → Saved {summary_path}")

            # --- EXPORT MACHINE-READABLE CSV FOR CROSS-VALIDATION ---
            csv_hotspot_path = f"{OUT}/mmpbsa_hotspots.csv"
            with open(csv_hotspot_path, "w") as f:
                f.write("Chain,Number,Residue,Energy_Mean,Energy_Std,Protein\n")
                for res, (mean_v, std_v) in sorted_c:
                    # Parse res label R:A:ARG:913
                    parts = res.split(":")
                    if len(parts) == 4:
                        _, chain, resname, resnum = parts
                        _, prot = label_residue(res)
                        f.write(f"{chain},{resnum},{resname},{mean_v:.4f},{std_v:.4f},{prot}\n")
            print(f"  → Saved {csv_hotspot_path} (machine-readable)")

            # --- EXPORT FULL PER-RESIDUE DECOMPOSITION (all energy terms) ---
            # One row per consensus residue, with the per-replica TOTAL, the
            # mean/SD across replicas and each energy term averaged over
            # replicas. This file is the source for the Supporting Information
            # table and for the per-term aggregates reported below.
            if len(decomp_terms) == len(decomp_results):
                csv_decomp_path = f"{OUT}/mmpbsa_decomposition.csv"
                n_rep = len(decomp_terms)
                term_cols = ["vdW", "Electrostatic", "Polar", "NonPolar"]
                with open(csv_decomp_path, "w") as f:
                    f.write("Rank,Chain,Number,Residue,Protein," +
                            ",".join(f"Rep{i+1}_Total" for i in range(n_rep)) +
                            ",Total_Mean,Total_Std," + ",".join(term_cols) + "\n")
                    for rank, (res, (mean_v, std_v)) in enumerate(sorted_c, 1):
                        parts = res.split(":")
                        if len(parts) != 4:
                            continue
                        _, chain, resname, resnum = parts
                        _, prot = label_residue(res)
                        per_rep = [d[res] for d in decomp_terms]
                        # DECOMP_TERMS = Internal, vdW, Electrostatic, Polar, NonPolar, TOTAL
                        tmean = np.mean(per_rep, axis=0)
                        f.write(f"{rank},{chain},{resnum},{resname},{prot}," +
                                ",".join(f"{r[-1]:.4f}" for r in per_rep) +
                                f",{mean_v:.4f},{std_v:.4f}," +
                                ",".join(f"{tmean[j]:.4f}" for j in (1, 2, 3, 4)) + "\n")
                print(f"  → Saved {csv_decomp_path} (per-residue energy terms)")

                # --- Per-term aggregates, appended to the hotspot summary ---
                sums = {}
                for label, subset in (
                    (f"Hotspots (<= {HOTSPOT_THRESHOLD})", [r for r, v in sorted_c if v[0] <= HOTSPOT_THRESHOLD]),
                    ("All consensus residues", [r for r, _ in sorted_c]),
                ):
                    arr = np.array([np.mean([d[r] for d in decomp_terms], axis=0) for r in subset])
                    sums[label] = (len(subset), arr.sum(axis=0))
                with open(summary_path, "a") as f:
                    f.write("\n" + "="*72 + "\n")
                    f.write("Energy term decomposition (kcal/mol, summed over residues,\n")
                    f.write("each residue averaged over frames and then over replicas)\n")
                    f.write("-"*72 + "\n")
                    f.write(f"{'Group':<28}{'n':>5}{'vdW':>9}{'Elec':>9}{'Polar':>9}"
                            f"{'NonPol':>9}{'TOTAL':>9}\n")
                    for label, (n, v) in sums.items():
                        f.write(f"{label:<28}{n:>5}{v[1]:>9.1f}{v[2]:>9.1f}{v[3]:>9.1f}"
                                f"{v[4]:>9.1f}{v[5]:>9.1f}\n")
                print(f"  → Appended energy-term aggregates to {summary_path}")
        else:
            print("  ✗ No residues found consistently across all replicas for consensus plot")


# =============================================================================
# 3. Salt Bridge Occupancy
# =============================================================================
if 3 in RUN:
    print("\n[3/3] Calculating inter-chain salt bridge occupancy...")

    # Positive residues: Arg (NH1/NH2), Lys (NZ), His (ND1/NE2)
    # Negative residues: Asp (OD1/OD2), Glu (OE1/OE2)
    # Inter-chain pairs only: MYC donors <-> MAX acceptors and vice versa.
    PAIRS_TO_CHECK = [
        (
            "segid PROA and (resname ARG LYS HIS) and (name NH1 NH2 NZ ND1 NE2)",
            "segid PROB and (resname ASP GLU) and (name OD1 OD2 OE1 OE2)",
            "MYC→MAX",
        ),
        (
            "segid PROB and (resname ARG LYS HIS) and (name NH1 NH2 NZ ND1 NE2)",
            "segid PROA and (resname ASP GLU) and (name OD1 OD2 OE1 OE2)",
            "MAX→MYC",
        ),
    ]

    # all_bridges: {bridge_label: {"occupancies": [...], "direction": str}}
    all_bridges = {}

    for i, (traj, eq) in enumerate(zip(TRAJS, EQUIL)):
        try:
            u = mda.Universe(PSF_FILE, traj)
            # bridge_counts key: (don_resid, don_segid, don_resname,
            #                     acc_resid, acc_segid, acc_resname, direction)
            # Counts frames (not atom hits) — deduplicated within each frame.
            bridge_counts = {}
            n_frames = 0

            for ts in u.trajectory:
                n_frames += 1
                # seen_this_frame ensures each RESIDUE PAIR is counted at most
                # once per frame regardless of how many atom pairs satisfy the
                # distance cutoff (fixes occupancy > 1.0 bug).
                seen_this_frame = set()

                for sel_don, sel_acc, direction in PAIRS_TO_CHECK:
                    donors    = u.select_atoms(sel_don)
                    acceptors = u.select_atoms(sel_acc)
                    if len(donors) == 0 or len(acceptors) == 0:
                        continue

                    dists = distance_array(donors.positions, acceptors.positions)
                    hits  = np.argwhere(dists < SALT_DIST)

                    for di, ai in hits:
                        d_atom = donors[di]
                        a_atom = acceptors[ai]
                        key = (
                            d_atom.resid, d_atom.segid, d_atom.resname,
                            a_atom.resid, a_atom.segid, a_atom.resname,
                            direction,
                        )
                        if key not in seen_this_frame:
                            seen_this_frame.add(key)
                            bridge_counts[key] = bridge_counts.get(key, 0) + 1

            # Convert to fractional occupancy — now guaranteed in [0, 1]
            n_stable = 0
            for key, count in bridge_counts.items():
                occ = count / n_frames          # always ≤ 1.0
                if occ >= SALT_OCCUPANCY:
                    don_chain = "MYC" if key[1] == "PROA" else "MAX"
                    acc_chain = "MYC" if key[4] == "PROA" else "MAX"
                    direction  = key[6]
                    # Clean label — direction encoded by colour, not text
                    bridge_label = (
                        f"{key[2]}{key[0]}({don_chain})"
                        f"--{key[5]}{key[3]}({acc_chain})"
                    )
                    if bridge_label not in all_bridges:
                        all_bridges[bridge_label] = {
                            "occupancies": [],
                            "direction":   direction,
                        }
                    all_bridges[bridge_label]["occupancies"].append(occ)
                    n_stable += 1

            print(f"  Rep {i+1} (eq {eq}ps) — {n_stable} salt bridges above "
                  f"{SALT_OCCUPANCY*100:.0f}% occupancy threshold")

        except Exception as e:
            print(f"  ✗ Rep {i+1} salt bridge analysis failed: {e}")

    # Consensus: present in ALL replicas above threshold
    consensus_bridges = {
        k: v for k, v in all_bridges.items()
        if len(v["occupancies"]) == len(TRAJS)
    }
    # Partial consensus: seen in 2 out of 3 replicas
    partial_bridges = {
        k: v for k, v in all_bridges.items()
        if 1 < len(v["occupancies"]) < len(TRAJS)
    }

    print(f"\n  Consensus salt bridges (all {len(TRAJS)} replicas): {len(consensus_bridges)}")
    print(f"  Partial consensus (2/{len(TRAJS)} replicas):        {len(partial_bridges)}")

    # ---- Save text summary ----
    sb_summary_path = f"{OUT}/3.0_salt_bridges_summary.txt"
    with open(sb_summary_path, "w") as f:
        
        for bridge, data in sorted(consensus_bridges.items(),
                                   key=lambda x: -np.mean(x[1]["occupancies"])):
            occs = data["occupancies"]
            f.write(f"{bridge:<45} {data['direction']:<12} "
                    f"{np.mean(occs):>10.3f} {np.std(occs):>8.3f}\n")

        f.write("\n=== PARTIAL CONSENSUS (2 of 3 replicas) ===\n")
        f.write(f"{'Bridge':<45} {'Direction':<12} {'Mean occ':>10} {'Std':>8}\n")
        f.write("-"*70 + "\n")
        for bridge, data in sorted(partial_bridges.items(),
                                   key=lambda x: -np.mean(x[1]["occupancies"])):
            occs = data["occupancies"]
            f.write(f"{bridge:<45} {data['direction']:<12} "
                    f"{np.mean(occs):>10.3f} {np.std(occs):>8.3f}\n")

    print(f"  → Saved {sb_summary_path}")

    # --- EXPORT MACHINE-READABLE CSV FOR CROSS-VALIDATION ---
    csv_sb_path = f"{OUT}/salt_bridges.csv"
    with open(csv_sb_path, "w") as f:
        f.write("Residue_A_Name,Residue_A_Number,Residue_B_Name,Residue_B_Number,Mean_Occupancy,Consensus\n")
        # Combine both for the CSV, mark consensus
        for bridge, data in consensus_bridges.items():
            # Parse label: GLU957(MYC)--LYS256(MAX)
            # This is a bit brittle, better to use a regex or split
            import re
            m = re.match(r"([A-Z]{3})(\d+)\(([A-Z]+)\)--([A-Z]{3})(\d+)\(([A-Z]+)\)", bridge)
            if m:
                res1, num1, prot1, res2, num2, prot2 = m.groups()
                # Order as MYC then MAX for consistency
                if prot1 == "MYC":
                    f.write(f"{res1},{num1},{res2},{num2},{np.mean(data['occupancies']):.3f},3/3\n")
                else:
                    f.write(f"{res2},{num2},{res1},{num1},{np.mean(data['occupancies']):.3f},3/3\n")
        for bridge, data in partial_bridges.items():
            import re
            m = re.match(r"([A-Z]{3})(\d+)\(([A-Z]+)\)--([A-Z]{3})(\d+)\(([A-Z]+)\)", bridge)
            if m:
                res1, num1, prot1, res2, num2, prot2 = m.groups()
                if prot1 == "MYC":
                    f.write(f"{res1},{num1},{res2},{num2},{np.mean(data['occupancies']):.3f},2/3\n")
                else:
                    f.write(f"{res2},{num2},{res1},{num1},{np.mean(data['occupancies']):.3f},2/3\n")
    print(f"  → Saved {csv_sb_path} (machine-readable)")

    def _plot_salt_bridges(bridges_dict, title, outpath, alpha=1.0):
        """Helper to plot a salt bridge occupancy bar chart.
        If title is falsy (empty string / None), no title is drawn."""
        if not bridges_dict:
            return
        sorted_b  = sorted(bridges_dict.items(),
                            key=lambda x: np.mean(x[1]["occupancies"]))
        labels_b  = [k for k, _ in sorted_b]
        means_b   = [np.mean(v["occupancies"]) for _, v in sorted_b]
        stds_b    = [np.std(v["occupancies"])  for _, v in sorted_b]

        fig, ax = plt.subplots(figsize=(11, max(4, len(labels_b) * 0.55)))
        ax.barh(range(len(labels_b)), means_b, xerr=stds_b,
                color="#3498db", edgecolor="k", linewidth=0.5, alpha=alpha,
                error_kw={"elinewidth": 1.2, "capsize": 3})
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b, fontsize=9)
        ax.set_xlabel("Mean occupancy (fraction of frames)", fontsize=11)
        # Hard cap at 1.0 — occupancy is a fraction, never exceeds 1
        ax.set_xlim(0, 1.05)
        ax.axvline(SALT_OCCUPANCY, color="red", linestyle="--", linewidth=0.9)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(handles=[
            mpatches.Patch(color="red", label=f"Threshold ({SALT_OCCUPANCY*100:.0f}%)",
                           linestyle="--", fill=False),
        ], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  → Saved {outpath}")

    _plot_salt_bridges(
        consensus_bridges,
        title="",  # Title removed per request.
        outpath=f"{OUT}/3.6_salt_bridges_consensus.png",
    )
    _plot_salt_bridges(
        partial_bridges,
        title=f"Partial-consensus Salt Bridges (2/{len(TRAJS)} replicas) — MYC-MAX",
        outpath=f"{OUT}/3.7_salt_bridges_partial.png",
        alpha=0.75,
    )

    if not consensus_bridges and not partial_bridges:
        print("  ✗ No salt bridges found above threshold in any replica — "
              "consider lowering SALT_OCCUPANCY or checking segid assignments.")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("  Analysis complete. Output files:")
for f in sorted(os.listdir(OUT)):
    print(f"    {OUT}/{f}")
print("="*60)
print("\n  HOTSPOT CROSS-REFERENCE WORKFLOW:")
print("  1. Open 3.0_hotspots_summary.txt  → MM-GBSA energy hotspots")
print("  2. Open 3.0_salt_bridges_summary.txt → persistent salt bridges")
print("  3. Residues in BOTH lists = highest-confidence drug targets")
print("="*60)
print("="*60)