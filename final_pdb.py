# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:04:00 2025

@author: Efrati
"""

import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from prody import parsePDBHeader
from typing import Optional


def read_pdb_to_dataframe(pdb_path: Optional[str] = None, model_index: int = 1, parse_header: bool = True) -> pd.DataFrame:
    """קוראת קובץ PDB ומחזירה DataFrame."""
    atomic_df = PandasPdb().read_pdb(pdb_path)
    if parse_header:
        header = parsePDBHeader(pdb_path)
    else:
        header = None
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")
    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]), header


def calculate_distance(row, ligand_coords):
    """מחשבת מרחק בין אטום לליגנד."""
    return np.sqrt(
        (row['x_coord'] - ligand_coords[0]) ** 2 +
        (row['y_coord'] - ligand_coords[1]) ** 2 +
        (row['z_coord'] - ligand_coords[2]) ** 2
    )


def calculate_angle(p1, p2, p3):
    """מחשבת זווית בין שלוש נקודות."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    dot_product = np.dot(v1, v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    dataset_path = "C:/Users/Efrati/anaconda3/envs/pdb_env/dataset"  # החלף בנתיב אמיתי
    searched_ligand = 'V'
    amino_acids_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    all_amino_acid_counts = {aa: 0 for aa in amino_acids_list}
    all_angles = []
    log_file_path = "C:/Users/Efrati/anaconda3/envs/pdb_env/processing_log.txt"

    with open(log_file_path, "w") as log_file:
        processed_files_count = 0
        failed_files = []

        for filename in os.listdir(dataset_path):
            if filename.endswith((".ent", ".pdb")):
                file_path = os.path.join(dataset_path, filename)
                log_file.write(f"Processing file: {filename}\n")
                try:
                    df, df_header = read_pdb_to_dataframe(file_path)
                except Exception as e:
                    log_file.write(f"Error processing file {filename}: {e}\n")
                    failed_files.append(filename)
                    continue

                ligand_location = df.loc[df['atom_name'] == searched_ligand, ['x_coord', 'y_coord', 'z_coord']]
                if not ligand_location.empty:
                    first_ligand_location = ligand_location.head(1)
                    first_ligand_coords = first_ligand_location.values[0]

                    df['distance_to_ligand'] = df.apply(lambda atom_record: calculate_distance(atom_record, first_ligand_coords), axis=1)

                    ligands_nearby_atoms = df[df['distance_to_ligand'] <= 5]
                    ligands_nearby_atoms = ligands_nearby_atoms[ligands_nearby_atoms['atom_name'] != searched_ligand]

                    result = ligands_nearby_atoms[['record_name', 'atom_name', 'residue_name', 'chain_id', 'residue_number', 'x_coord', 'y_coord', 'z_coord', 'distance_to_ligand']]

                    # Filter out ATOM type atoms for counting amino acids in the binding site
                    result = result[result['record_name'] == 'ATOM']

                    unique_residues = result.drop_duplicates(subset=['residue_name', 'residue_number'])
                    amino_acid_counts = unique_residues.groupby('residue_name').size()

                    for aa, count in amino_acid_counts.items():
                        if aa in amino_acids_list:
                            all_amino_acid_counts[aa] += count

                    hetatm_vo4_nearby = ligands_nearby_atoms[
                        (ligands_nearby_atoms['record_name'] == 'HETATM') &
                        (ligands_nearby_atoms['residue_name'] == 'VO4')
                    ]
                    if not hetatm_vo4_nearby.empty:
                        coordinates_hetatm_vo4_nearby = hetatm_vo4_nearby[['x_coord', 'y_coord', 'z_coord']].values
                        num_atoms = len(coordinates_hetatm_vo4_nearby)

                        if num_atoms > 1:
                            for i in range(num_atoms):
                                for j in range(i + 1, num_atoms):
                                    angle = calculate_angle(
                                        coordinates_hetatm_vo4_nearby[i],
                                        first_ligand_coords,
                                        coordinates_hetatm_vo4_nearby[j]
                                    )
                                    if angle is not None:
                                        all_angles.append(angle)

                processed_files_count += 1
                log_file.write(f"File {filename} processed successfully.\n")

        log_file.write(f"\nTotal files processed: {processed_files_count}\n")
        log_file.write(f"Failed files: {failed_files}\n")

    print("\nTotal Amino Acid Counts:")
    print(all_amino_acid_counts)
    print("\nAll Angles:")
    print(all_angles)
    print(f"Log file saved to: {log_file_path}")
