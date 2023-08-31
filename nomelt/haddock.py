"""This module contains functions for running HADDOCK3 on a protein-DNA complex.

The first component supperimposes the protein in the complex with a new protein structure, and removes the old protein from the complex. This function is very unique to the homeodomain complex structure on the pdb and contains references to specific chain IDs. Not likely to work on any other system
The second component runs HADDOCK3 on the resulting complex.

"""
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import MDAnalysis.analysis.align
import os
import numpy as np
from string import Template
import subprocess

import logging
logger = logging.getLogger(__name__)

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

HADDOCK3_CONFIG_TEMPLATE = Template("""
# ====================================================================
# protein-DNA docking example

# directory in which the scoring will be done
run_dir = "$run_dir"

# ###
mode = "local"

# postprocess caprieval folders with haddock3-analyse
postprocess = true

# molecules to be docked
molecules =  [
    "$protein_pdb",
    "$dna_pdb"
    ]

# ====================================================================
# Parameters for each stage are defined below, prefer full paths
# ====================================================================
[topoaa]
autohis = true

[rigidbody]
randorien = false
sampling = 500
epsilon = 78
dielec = "cdie"
surfrest = true

[seletop]
select = 20

[flexref]
epsilon = 78
dielec = "cdie"
dnarest_on = true
contactairs = true

[emref]
dnarest_on = true
dielec = "cdie"
""")

def prepare_one_complex(complex_struct: str, variant_struct: str, output_dir: str):
    """
    Prepare a protein-DNA complex for HADDOCK3 by superimposing a new protein structure onto 
    the protein in the given complex, while removing the old protein.

    This function is specific to the homeodomain complex structure in the pdb and contains 
    references to specific chain IDs. It may not work for other systems.

    Parameters:
    - complex_struct (str): Path to the initial complex PDB file.
    - variant_struct (str): Path to the new protein structure PDB file.
    - output_dir (str): Directory to save the processed structures (protein, DNA, complex).

    Outputs:
    The following files are written to the output_dir:
    - protein.pdb: The superimposed new protein structure.
    - dna.pdb: The DNA structure from the initial complex.
    - complex.pdb: The new complex formed by merging the superimposed protein and DNA.

    Returns:
    None
    """
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    variant_struct = os.path.abspath(variant_struct)
    # Load the PDB files
    complex_u = mda.Universe(complex_struct)
    new_protein_u = mda.Universe(variant_struct)
    logger.info(f"New protein residues: {len(new_protein_u.select_atoms('protein').residues)}")

    # get rid of second protein in complex
    complex_u.atoms = complex_u.atoms - complex_u.select_atoms("chainID B")
    # get rid of the water
    complex_u.atoms = complex_u.atoms - complex_u.select_atoms("resname HOH")

    # Select the protein atoms for alignment (Modify the selections as necessary).
    ref_selection = complex_u.select_atoms("chainID A") # we only want chain A, since there is a second homeodomain in here.
    mobile_selection = new_protein_u.select_atoms("protein")
    dna_selection = complex_u.select_atoms("(chainID C) or (chainID D)")
    logger.info(f"{len(ref_selection)} atoms in original protein, {len(dna_selection)} in DNA, {len(mobile_selection)} in new protein")

    ref_seq = ''.join(AA_MAP.get(residue.resname, 'X') for residue in ref_selection.residues)
    variant_seq = ''.join(AA_MAP.get(residue.resname, 'X') for residue in mobile_selection.residues)
    logger.info(f"Got residues for ref, variant: {len(ref_seq), len(variant_seq)}")

    # save temporary fasta in order to align
    with open('./tmp/temp.fasta', 'w') as f:
        f.write(f">ref\n{ref_seq}\n>variant\n{variant_seq}\n")
    
    selection = MDAnalysis.analysis.align.fasta2select(
        './tmp/temp.fasta',
        ref_resids=[a.resid for a in ref_selection.select_atoms('name CA')],
        target_resids=[a.resid for a in mobile_selection.select_atoms('name CA')],
        
    )
    ref_sel_str = selection['reference']
    if type(ref_sel_str) == list:
        ref_sel_str = ref_sel_str[0]
    mob_sel_str = selection['mobile']
    if type(mob_sel_str) == list:
        mob_sel_str = mob_sel_str[0]

    # return selection, ref_selection, mobile_selection
    # superimpose
    old_rmsd, new_rmsd = MDAnalysis.analysis.align.alignto(
        ref_selection,
        mobile_selection,
        select=(ref_sel_str, mob_sel_str),
        weights=None,
        match_atoms=False,
        strict=False
    )

    # we need to get rid of tails that are sterically clashing
    # identify terminals of the mobile selection outside of the aligned region
    align_mob_sel = mobile_selection.select_atoms(mob_sel_str)
    align_resids = [r.resid for r in align_mob_sel.residues]
    n_terminus_unaligned_resids = []
    for r in mobile_selection.residues:
        if r.resid in align_resids:
            break
        else:
            n_terminus_unaligned_resids.append(r.resid)
    c_terminus_unaligned_resids = []
    for r in reversed(mobile_selection.residues):
        if r.resid in align_resids:
            break
        else:
            c_terminus_unaligned_resids.append(r.resid)
    # select those atoms and check for clashes.
    if len(n_terminus_unaligned_resids) == 0:
        pass
    else:
        sel_str = ""
        for id_ in n_terminus_unaligned_resids:
            sel_str += f"(resid {id_}) or"
        sel_str = sel_str[:-3]
        n_tail = mobile_selection.select_atoms(sel_str)
        distances = distance_array(n_tail.positions, dna_selection.positions, box=complex_u.dimensions)
        clashing_pairs = np.where(distances < 2.0) # angstrom
        tail_clashes = False
        for i, j in zip(*clashing_pairs):
            tail_clashes = True
            atom1 = n_tail[i]
            atom2 = dna_selection[j]
            logger.info(f"Clash between {atom1.resname} {atom1.resid} {atom1.name} and {atom2.resname} {atom2.resid} {atom2.name}")
        if tail_clashes:
            mobile_selection = mobile_selection.atoms - n_tail.atoms
            logger.info(f"Removed clashing tail, new atom count {len(mobile_selection)}")
            
    if len(c_terminus_unaligned_resids) == 0:
        pass
    else:
        sel_str = ""
        for id_ in c_terminus_unaligned_resids:
            sel_str += f"(resid {id_}) or"
        sel_str = sel_str[:-3]
        c_tail = mobile_selection.select_atoms(sel_str)
        distances = distance_array(c_tail.positions, dna_selection.positions, box=complex_u.dimensions)
        clashing_pairs = np.where(distances < 2.0) # angstrom
        tail_clashes = False
        for i, j in zip(*clashing_pairs):
            tail_clashes = True
            atom1 = c_tail[i]
            atom2 = dna_selection[j]
            logger.info(f"Clash between {atom1.resname} {atom1.resid} {atom1.name} and {atom2.resname} {atom2.resid} {atom2.name}")
        if tail_clashes:
            mobile_selection = mobile_selection.atoms - c_tail.atoms
            logger.info(f"Removed clashing tail, new atom count {len(mobile_selection)}")
        
    
    # save the protein alone
    mobile_selection.atoms.write(os.path.join(output_dir, 'protein.pdb'))
    # logger.info(f"RMSD of alignment: {old_rmsd} -> {new_rmsd}")
    # Remove the old protein atoms from the complex
    complex_u.atoms = complex_u.atoms - ref_selection.atoms 
    # write the DNA alone
    complex_u.atoms.write(os.path.join(output_dir, 'dna.pdb'))

    # Merge the aligned new protein atoms with the rest of the complex
    merged_universe = mda.Merge(complex_u.atoms, mobile_selection.atoms)

    # save the resulting complex
    merged_universe.atoms.write(os.path.join(output_dir, 'complex.pdb'))

def run_haddock(protein_pdb: str, dna_pdb: str, run_dir: str):
    """
    Execute a HADDOCK3 run on a protein-DNA complex.

    This function sets up the configuration required for a HADDOCK3 run, writes the configuration
    to a file, and then invokes HADDOCK3 on the provided protein and DNA structures.

    Parameters:
    - protein_pdb (str): Path to the protein PDB file to be docked.
    - dna_pdb (str): Path to the DNA PDB file to be docked.
    - run_dir (str): Directory where the HADDOCK3 run will be executed and results saved.

    Outputs:
    The HADDOCK3 results will be saved in the specified run_dir.

    Returns:
    None
    """
    protein_pdb = os.path.abspath(protein_pdb)
    dna_pdb = os.path.abspath(dna_pdb)
    run_dir = os.path.abspath(run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'haddock3_config.cfg'), 'w') as f:
        f.write(HADDOCK3_CONFIG_TEMPLATE.substitute(
            run_dir=os.path.join(run_dir, 'run'),
            protein_pdb=protein_pdb,
            dna_pdb=dna_pdb
        ))
    # move to the run directory and open a subprocess
    try:
        subprocess.run(['haddock3', os.path.join(run_dir, 'haddock3_config.cfg'), '--restart', '0'], check=True)
    except:
        subprocess.run(['haddock3', os.path.join(run_dir, 'haddock3_config.cfg')], check=True)