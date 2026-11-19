# MYC-MAX MD Analysis Pipeline
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24377743.svg)](https://doi.org/10.5281/zenodo.24377743)

This repository contains the Python scripts used for analyzing Molecular Dynamics simulations of the MYC-MAX complex and isolated c-MYC. The analysis pipeline directly mirrors the methodology and results discussed in the associated paper: *Identification of Candidate Pockets at the MYC-MAX Interface for Protein-Protein Interaction Disruption*.

## Results Directory Structure
All scripts automatically output to the `Results/` directory:
- `Results/Individual/`: RMSD, RMSF, and Rg for single-protein replicas (isolated c-MYC).
- `Results/Comparison/`: Comparative analysis between isolated c-MYC and c-MYC in the complex.
- `Results/Complex/`: Interface Contacts, MM-GBSA results and Salt Bridges for the MYC:MAX dimer.
- `Results/Pockets/`: MDpocket descriptors, sub-pocket clustering, and multi-method cross-validation tables.

Some figures were renamed after generation to match the paper numbering (e.g. `subpockets_render.png` is saved as `4.0_subpockets_render.png`).

---

## Workflow & Methodology

The files have been renamed and numbered to follow the analytical steps of the paper. Run the scripts from the repository root and in the following order. Steps 4 and 5 also need MDpocket to be run, as explained below.

### 1. `01_isolated_myc_analysis.py`
- **Paper Section:** 1. Conformational disorder of isolated c-MYC
- **Purpose:** Primary analysis of isolated c-MYC replicas. It calculates the backbone RMSD, per-residue RMSF, and Radius of Gyration (Rg) to demonstrate the high structural flexibility and lack of stable 3D folded structure of the isolated c-MYC.
- **Note:** The script asks which analyses to run. Press Enter to run all of them.
- **Outputs:** Saves metrics and plots to `Results/Individual/`.

### 2. `02_comparative_myc_analysis.py`
- **Paper Section:** 2. MAX binding stabilises c-MYC conformation
- **Purpose:** Compares the structural dynamics (RMSD, RMSF, and Rg) of c-MYC in isolation versus c-MYC in the MAX complex. This step demonstrates how MAX binding drastically reduces c-MYC's conformational drift and stabilizes its structure.
- **Outputs:** Saves comparative plots to `Results/Comparison/`.

### 3. `03_myc_max_complex_analysis.py`
- **Paper Section:** 3. The MYC-MAX binding interface is stable and persistent
- **Purpose:** Analyzes the MYC-MAX complex to confirm binding interface stability. It monitors inter-chain atomic contacts, calculates the MM-GBSA per-residue energy decomposition (through `gmx_MMPBSA`) and persistent inter-chain salt bridges over the trajectory.
- **Note:** The script asks which analyses to run. Press Enter to run all of them. The MM-GBSA calculation can take several hours per replica.
- **Outputs:** Saves Interface Contacts, MM-GBSA hotspots and Salt Bridges results to `Results/Complex/`.

### 4. `04_pocket_identification.py`
- **Paper Section:** 4.1 Trajectory-based pocket detection
- **Before running:** The three MYC-MAX replicas have to be aligned and pooled into a single trajectory, and MDpocket has to be run on it to obtain the frequency grid (`mdpout_freq_grid.dx`) and the density grid (`mdpout_dens_grid.dx`).
- **Purpose:** Uses MDpocket frequency grid clustering on the pooled complex trajectory to identify contiguous sub-pockets at the MYC-MAX interface. Extracts the core lining residues and writes a ChimeraX script to visualize the sub-pockets.
- **Outputs:** Saves sub-pocket tables and the ChimeraX script to `Results/Pockets/summary/`, and one pocket definition per sub-pocket to `Results/Pockets/pocket_definitions/`.

### 5. `05_pocket_characterization.py`
- **Paper Section:** 4.1 Trajectory-based pocket detection (continued)
- **Before running:** MDpocket has to be run again for each sub-pocket using its pocket definition. Script 04 prints the exact command, for example for SP1:
  ```bash
  mdpocket --trajectory_file <pooled_trajectory>.xtc --trajectory_format xtc \
           -f MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb \
           --selected_pocket Results/Pockets/pocket_definitions/SP1_pocket_grid.pdb \
           -o Results/Pockets/SP1/SP1
  ```
- **Purpose:** Evaluates physicochemical properties for the identified pockets over time. It calculates Mean Pocket Volume, Hydrophobicity Score, Mean Local Hydrophobic Density, and Polarity Score.
- **Outputs:** Saves summary tables and plots to `Results/Pockets/sub_pockets_analysis/`.

### 6. `06_multi_method_validation.py`
- **Paper Section:** 4.3 Triple-method consensus
- **Purpose:** Integrates the results from three independent approaches: MDpocket cavity detection, MM-GBSA per-residue energy decomposition, and persistent salt bridge analysis. Identifies the highest-confidence interface hotspots (residues scoring at least 2 out of 3 methods, e.g., in sub-pocket SP2).
- **Outputs:** Generates the final consensus table, cross-validation summaries and a ChimeraX script (`visualize_cross_validation.cxc`) in `Results/Pockets/`.

### Extra Scripts
- `individual_residue_testing.py`: Helper script that checks how close a specific residue stays to a given sub-pocket along the pooled trajectory, reporting the fraction of frames within several distance cutoffs.

---

## 🛠 Prerequisites
- **Python 3.10+** (required by the versions pinned in `requirements.txt`)
- **Python Libraries:** `MDAnalysis`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `gridData`. Install them with `pip install -r requirements.txt`.
- **External Tools & Viewers:** `gmx_MMPBSA` (for free energy calculations), `mdpocket` from fpocket (for cavity analysis), VMD and **ChimeraX** (for visualizing the generated `.cxc` structural scripts). Versions are listed in `requirements.txt`.

## Data Handling & GitHub
Due to the significant size of Molecular Dynamics trajectories (individual `.xtc` files range from 100MB to 7GB, total project >50GB), raw data files are **excluded** from this repository.

**To use this repository with your own data:**
1. Keep the same directory structure (`MYC/` and `MYC-MAX/` folders with one subfolder per replica).
2. Make sure your files match the paths defined in the `CONFIG` section at the top of each script, or edit those paths.
3. Run the scripts from the repository root, since the paths are relative to it.

## Citation
If you use these scripts, please cite the archived release on Zenodo: [10.5281/zenodo.24377743](https://doi.org/10.5281/zenodo.24377743)

## License
The code in this repository is released under the [MIT License](LICENSE).
