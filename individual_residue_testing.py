#!/usr/bin/env python3
"""
Residue Dynamics vs Pocket Envelope
===================================
Analyzes the distance between a specific residue and a pocket's spatial 
envelope across the MD trajectory.

Reports the fraction of frames in which the residue is within a specified
distance of the pocket.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import MDAnalysis as mda

# ============================== CONFIG ===================================
STRUCTURE_PDB = "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb"  # topology (atom names/order)
TRAJECTORY = "MYC-MAX/traj_all_fit_v3.xtc"                                     # the real, aligned, pooled trajectory
POCKET_GRID_PDB = "Results/Pockets/pocket_definitions/SP2_pocket_grid.pdb" # modify depending on the pocket you want to check

CHAIN, RESNUM = "A", 914
STRIDE = 1  # process every Nth frame 
CUTOFFS = [3.0, 4.0, 5.0]   # Angstrom - report "fraction of frames within X A" for a few reference distances
HEAVY_ATOMS_ONLY = True     # exclude hydrogens - see note on HB2 from the static check

OUT_CSV = "residue_vs_pocket_distance.csv"
# ===========================================================================


def load_pocket_voxels(pocket_grid_pdb):
    coords = []
    with open(pocket_grid_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(coords)


def residue_dynamics_vs_pocket(chain=CHAIN, resnum=RESNUM, structure_pdb=STRUCTURE_PDB,
                                trajectory=TRAJECTORY, pocket_grid_pdb=POCKET_GRID_PDB,
                                stride=STRIDE, cutoffs=CUTOFFS, heavy_only=HEAVY_ATOMS_ONLY,
                                out_csv=OUT_CSV):
    pocket_voxels = load_pocket_voxels(pocket_grid_pdb)
    tree = cKDTree(pocket_voxels)

    u = mda.Universe(structure_pdb, trajectory)
    sel = u.select_atoms(f"chainID {chain} and resid {resnum}")
    if len(sel) == 0:
        seg = "PROA" if chain == "A" else "PROB"
        sel = u.select_atoms(f"segid {seg} and resid {resnum}")
    if len(sel) == 0:
        raise ValueError(f"No atoms found for chain {chain} resid {resnum}")

    if heavy_only:
        heavy_sel = sel.select_atoms("not name H*")
        if len(heavy_sel) > 0:
            sel = heavy_sel

    times, min_dists = [], []
    for ts in u.trajectory[::stride]:
        d, _ = tree.query(sel.positions)
        times.append(ts.time)
        min_dists.append(d.min())

    df = pd.DataFrame({"time_ps": times, "min_dist_A": min_dists})
    df.to_csv(out_csv, index=False)

    arr = df["min_dist_A"].values
    print(f"{chain}{resnum} vs {pocket_grid_pdb} across {len(arr)} sampled frames "
          f"(stride={stride}, {'heavy atoms only' if heavy_only else 'all atoms'}):")
    print(f"  min={arr.min():.2f} A   mean={arr.mean():.2f} A   max={arr.max():.2f} A")
    for c in cutoffs:
        frac = (arr <= c).mean() * 100
        print(f"  within {c:.1f} A in {frac:.1f}% of sampled frames")
    print(f"\nWrote per-frame distances to {out_csv}")

    return df


if __name__ == "__main__":
    residue_dynamics_vs_pocket()