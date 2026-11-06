#!/usr/bin/env python3

"""
Multi-Method Cross-Validation & Consensus Pipeline
================================================================
Integrates Hotspot consensus metrics (MM-GBSA, salt bridges) with MDpocket 
sub-pocket identification (SP1..SPn) to discover high-confidence binding targets.

Pipeline Overview:
    1. Loads and aligns residue data from MM-GBSA calculations, salt bridge 
       analyses, and spatial sub-pocket centroids.
    2. Maps residues against pockets using core definitions or extended distance 
       thresholds.
    3. Scores and filters cross-validated targets by consensus agreement.

Outputs (saved under Results/Pockets/):
    - cross_validation_unified.csv: Filtered consensus dataset.
    - 6.1_cross_validation_table.png: Rendered cross-validation summary table.
    - visualize_cross_validation.cxc: ChimeraX script that reproduces Figure 8.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import MDAnalysis as mda

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "Results/Pockets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. SUB-POCKET DATA (from extract_subpockets.py - the real, validated pipeline)
SP_SUMMARY_CSV  = "Results/Pockets/summary/subpockets_core_summary.csv"   # sp_name, centroid_x/y/z, volume, freq, ...
SP_RESIDUES_CSV = "Results/Pockets/summary/subpockets_core_residues.csv"  # sp_name, chain, resnum, resname, dist_to_pocket_A

# 2. PARAMETERS
# EXTENDED_CUTOFF is applied to hotspot/salt-bridge residues that fall
# outside any pocket's core (defined by RESIDUE_CUTOFF in pocket identification).
# It flags them as plausibly associated with a nearby pocket (e.g., "SP2+").
EXTENDED_CUTOFF = 8.0
# Off: only core lining residues count as pocket members. The static 8 A
# COM-to-centroid rule flagged VAL953 / MET253 as "SP6+", but along the pooled
# trajectory their mean heavy-atom distance to SP6 is > 4.7 A (MET253 within
# 8 A of the centroid in only 28% of frames), so they are not pocket residues.
USE_EXTENDED_CUTOFF = False
PROTEIN_PDB     = "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb"

# 3. EXTERNAL DATA PATHS
CSV_MMPBSA = "Results/Complex/mmpbsa_hotspots.csv"
CSV_SALT   = "Results/Complex/salt_bridges.csv"

# Fallback salt bridges if CSV_SALT is missing. There is deliberately no MM-GBSA
# fallback: an outdated hotspot list would silently skew the consensus.
SALT_BRIDGES_FALLBACK = {
    ("A", 957, "GLU"), ("B", 256, "LYS"),
    ("A", 970, "ARG"), ("B", 275, "GLU"),
}

# MM-GBSA hotspot criterion: a residue counts as an energetic hotspot when its
# mean per-residue binding contribution (DELTAS/TDC, mean of 3 replicas) is
# <= this value (kcal/mol). -2.0 is a convention by analogy with the classical
# alanine-scanning hotspot definition (Bogan & Thorn 1998), not a standard for
# MM-GBSA decomposition; -1.0 adds four further consensus residues.
HOTSPOT_THRESHOLD = -2.0

TABLE_COLORS = {
    "header":     "#2C3E50",
    "text":       "#2B2B2B",
    "muted":      "#8A8A8A",
    "MYC":        "#FADBD8",
    "MAX":        "#D6EAF8",
    "check_bg":   "#C8E6C9",
}

# Pastel colour per sub-pocket for the MDpocket column of the table
SP_COLORS = {
    "SP1": "#A1C9F4", "SP2": "#FFB482", "SP3": "#8DE5A1",
    "SP4": "#D0BBFF", "SP5": "#FF9F9B", "SP6": "#DEBB9B",
}

# =============================================================================
# DATA PARSING
# =============================================================================

def load_external_data():
    hotspots, salt_bridges = set(), set()
    sb_partners = {}   # residue -> "PARTNER (occupancy%)"
    energies = {}   # (chain, number, resname) -> (mean, std) over replicas, kcal/mol
    print("\n[Loading External Data]")
    if os.path.exists(CSV_MMPBSA):
        df_gb = pd.read_csv(CSV_MMPBSA)
        for _, row in df_gb.iterrows():
            energies[(row["Chain"], int(row["Number"]), row["Residue"])] = (
                row["Energy_Mean"], row["Energy_Std"])
        top_gb = df_gb[df_gb["Energy_Mean"] <= HOTSPOT_THRESHOLD]
        for _, row in top_gb.iterrows():
            hotspots.add((row["Chain"], int(row["Number"]), row["Residue"]))
        print(f"  \u2192 Loaded {len(hotspots)} MM-GBSA hotspots from CSV "
              f"(\u0394G <= {HOTSPOT_THRESHOLD} kcal/mol).")
    else:
        raise FileNotFoundError(
            f"Missing {CSV_MMPBSA} - run section 2 of 03_myc_max_complex_analysis.py first."
        )

    if os.path.exists(CSV_SALT):
        df_sb = pd.read_csv(CSV_SALT)
        for _, row in df_sb.iterrows():
            res_a = ("A", int(row["Residue_A_Number"]), row["Residue_A_Name"])
            res_b = ("B", int(row["Residue_B_Number"]), row["Residue_B_Name"])
            occ = f"{row['Mean_Occupancy'] * 100:.0f}%"
            salt_bridges.update((res_a, res_b))
            sb_partners[res_a] = f"{res_b[2]}{res_b[1]} ({occ})"
            sb_partners[res_b] = f"{res_a[2]}{res_a[1]} ({occ})"
        print(f"  \u2192 Loaded {len(salt_bridges)} Salt Bridge residues from CSV.")
    else:
        print("  \u2192 CSV_SALT not found. Using Fallback Salt Bridge data.")
        salt_bridges = SALT_BRIDGES_FALLBACK
    return hotspots, salt_bridges, energies, sb_partners


def load_subpockets():
    """
    Load sub-pocket data from the pocket identification step.
    
    Returns:
    - sp_residues: Dict mapping pocket name to its "core" residues.
    - sp_centroids: Dict mapping pocket name to its centroid coordinates.
    """
    print("\n[Loading Sub-pockets from extract_subpockets.py output]")
    missing = [p for p in (SP_SUMMARY_CSV, SP_RESIDUES_CSV) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} - run extract_subpockets.py first (it writes "
            f"these into Results/Pockets/summary/)."
        )

    res_df = pd.read_csv(SP_RESIDUES_CSV)
    summary_df = pd.read_csv(SP_SUMMARY_CSV)

    sp_residues = {}
    for sp_name, group in res_df.groupby("sp_name"):
        sp_residues[sp_name] = set(
            (str(row["chain"]), int(row["resnum"]), str(row["resname"]))
            for _, row in group.iterrows()
        )

    sp_centroids = {
        row["sp_name"]: np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]])
        for _, row in summary_df.iterrows()
    }

    for name in sorted(sp_residues, key=lambda n: int(n.replace("SP", ""))):
        n_res = len(sp_residues[name])
        interface_flag = ""
        row = summary_df[summary_df["sp_name"] == name]
        if not row.empty and bool(row.iloc[0].get("is_interface_pocket", False)):
            interface_flag = "  [INTERFACE]"
        print(f"  \u2192 {name}: {n_res} core residues{interface_flag}")

    return sp_residues, sp_centroids


def get_nearest_sp(res, centroids_sp):
    if not centroids_sp:
        return None
    u = mda.Universe(PROTEIN_PDB)
    seg = "PROA" if res[0] == "A" else "PROB"
    sel = u.select_atoms(f"segid {seg} and resid {res[1]}")
    if not sel:
        # fall back to plain chain ID selection, in case this structure
        # doesn't use CHARMM-GUI's PROA/PROB segid convention
        sel = u.select_atoms(f"chainID {res[0]} and resid {res[1]}")
    if not sel:
        return None
    pos = sel.center_of_mass()
    best_sp, min_dist = None, 999.9
    for name, center in centroids_sp.items():
        dist = np.linalg.norm(pos - center)
        if dist < min_dist:
            best_sp, min_dist = name, dist

    if min_dist < EXTENDED_CUTOFF:
        return f"{best_sp}+"
    return None


def main():
    print("\n" + "=" * 60 + "\n      MYC-MAX UNIFIED CROSS-VALIDATION\n" + "=" * 60)

    hotspots, salt_bridges, energies, sb_partners = load_external_data()
    sp_residues, sp_centroids = load_subpockets()
    all_sp_residues = set().union(*sp_residues.values()) if sp_residues else set()
    print("SP core residues (union across all pockets):", len(all_sp_residues))

    all_res = salt_bridges | hotspots | all_sp_residues
    rows = []
    print("\n[Assigning Pockets & Generating Table]")
    for res in sorted(all_res, key=lambda x: (x[0], x[1])):
        in_sb, in_gb = res in salt_bridges, res in hotspots
        sps = [name for name, s in sp_residues.items() if res in s]

        # EXTENDED-NEIGHBORHOOD FALLBACK: if this residue isn't a core
        # member of any pocket, but sits within EXTENDED_CUTOFF of one,
        # flag it as e.g. "SP2+".
        if USE_EXTENDED_CUTOFF and not sps and sp_centroids:
            nearest = get_nearest_sp(res, sp_centroids)
            if nearest:
                sps = [nearest]

        if not in_sb and not in_gb and not sps:
            continue

        score = sum([in_sb, in_gb, len(sps) > 0])
        dg = energies.get(res)

        rows.append({
            "Chain": res[0], "Protein": "MYC" if res[0] == "A" else "MAX",
            "Residue": res[2], "Number": res[1],
            "Salt Bridge": sb_partners.get(res, "\u2713") if in_sb else "-",
            "\u0394G (kcal/mol)": f"{dg[0]:.2f} \u00b1 {dg[1]:.2f}" if dg else "-",
            "MDpocket": "+".join(sps) if sps else "-",
            "Score": f"{score}/3", "_score_val": score,
            "_in_sb": in_sb, "_in_gb": in_gb, "_in_sp": bool(sps),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[!] No residues met any criteria - check your input CSVs and paths.")
        return
    df_filtered = df[df["_score_val"] >= 2].sort_values(
        by=["_score_val", "Chain", "Number"], ascending=[False, True, True])

    out_csv = os.path.join(OUTPUT_DIR, "cross_validation_unified.csv")
    df_filtered.to_csv(out_csv, index=False)
    plot_consensus_table(df_filtered, os.path.join(OUTPUT_DIR, "6.1_cross_validation_table.png"))
    write_chimerax_script(df_filtered, os.path.join(OUTPUT_DIR, "visualize_cross_validation.cxc"))

    print(f"\n[\u2713] Finished. {len(df_filtered)} high-confidence residues identified.")
    print(f"    Results saved in {OUTPUT_DIR}/")


def plot_consensus_table(df, output_path):
    """Render the three-method consensus table (Table 3).

    Rows are tinted by protein (MYC / MAX). A method cell turns green with a
    check mark when the residue meets that method's criterion; the MDpocket
    cell takes the colour of its sub-pocket. The MM-GBSA column always shows
    mean +/- SD over replicas so the margin to the threshold is visible.
    """
    c = TABLE_COLORS
    minus = lambda t: t.replace("-", "\u2212")
    cols = ["Protein", "Residue", "Salt Bridge",
            f"MM-GBSA \u0394G\n(kcal/mol)", "MDpocket", "Score"]
    cells, met = [], []
    for _, row in df.iterrows():
        cells.append([
            row["Protein"], f"{row['Residue']}{row['Number']}",
            f"\u2713  {row['Salt Bridge']}" if row["_in_sb"] else "\u2013",
            minus(row["\u0394G (kcal/mol)"]),
            row["MDpocket"] if row["_in_sp"] else "\u2013",
            row["Score"],
        ])
        met.append([False, False, row["_in_sb"], row["_in_gb"], row["_in_sp"], False])

    fig, ax = plt.subplots(figsize=(11, 0.55 * (len(df) + 1) + 0.6))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center",
                     colWidths=[0.11, 0.13, 0.22, 0.22, 0.16, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.6)

    for j in range(len(cols)):
        head = table[0, j]
        head.set_facecolor(c["header"])
        head.set_edgecolor("grey")
        head.set_text_props(color="white", fontweight="bold")
        head.set_height(head.get_height() * 1.3)

    for i, (_, row) in enumerate(df.iterrows()):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_edgecolor("grey")
            cell.set_linewidth(1.5)
            cell.set_facecolor(c[row["Protein"]])
            cell.set_text_props(color=c["text"])
            if met[i][j]:
                cell.set_facecolor(c["check_bg"])
            elif cells[i][j] == "\u2013":
                cell.set_text_props(color=c["muted"])
            if j == 4 and met[i][j]:
                cell.set_facecolor(SP_COLORS.get(row["MDpocket"], c["check_bg"]))
            if j == 1:
                cell.set_text_props(fontweight="normal", color=c["text"])

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --- Figure 8 (ChimeraX) -----------------------------------------------------
FIGURE_PDB    = "crystal_proteins.pdb"   # same structure used for the original render
FIGURE_COLORS = {"A": "#ed8282", "B": "#99c1f1", "flagged": "red"}  # MYC, MAX, consensus
FIGURE_PNG    = "Results/Pockets/final_residues.png"
SHOW_LABELS   = False   # set False to skip labels entirely - add them yourself
                        # later in an image editor, on the rendered PNG
LABEL_OFFSET  = (2.0, 2.0, 2.0)  # Angstrom - nudges labels away from the atom
                                   # they're attached to, so they don't sit
                                   # directly on top of overlapping sticks


def write_chimerax_script(df, cxc_path):
    """ChimeraX script reproducing Figure 8: MYC / MAX cartoons in light red /
    light blue, consensus (score >= 2) residues as red sticks."""
    by_chain = {}
    for _, r in df.iterrows():
        by_chain.setdefault(r["Chain"], []).append(str(r["Number"]))
    flagged = " | ".join(f"/{ch}:{','.join(nums)}" for ch, nums in sorted(by_chain.items())) or "none"

    with open(cxc_path, "w") as f:
        f.write(f"open {os.path.abspath(FIGURE_PDB)}\n")
        f.write("set bgColor white\n")
        f.write(f"color /A {FIGURE_COLORS['A']}\n")
        f.write(f"color /B {FIGURE_COLORS['B']}\n")

        f.write("\n# --- Cross-validated (score >= 2) residues, in red ---\n")
        f.write(f"name flagged {flagged}\n")
        f.write("show flagged atoms\n")
        f.write("style flagged stick\n")
        f.write(f"color flagged {FIGURE_COLORS['flagged']}\n")

        if SHOW_LABELS:
            ox, oy, oz = LABEL_OFFSET
            f.write(f"label flagged residues text {{0.name}}{{0.number}} height 1.5 "
                    f"color black offset {ox},{oy},{oz}\n")
        else:
            f.write("# labels skipped (SHOW_LABELS = False) - add them manually in an\n"
                    "# image editor afterward, using cross_validation_unified.csv as\n"
                    "# the reference for which residue is which.\n")

        f.write("\nview\n")
        f.write("# rotate to the figure orientation, then:\n")
        f.write(f"# save {os.path.abspath(FIGURE_PNG)} width 3000 height 2400 "
                f"supersample 3 transparentBackground false\n")


if __name__ == "__main__":
    main()