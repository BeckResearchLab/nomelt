"""Runs AF """

import os
import subprocess
from dataclasses import dataclass, field
import tempfile

@dataclass
class AlphaFoldParams:
    max_template_date: str = "2021-11-01"
    model_preset: str = "monomer"
    db_preset: str = "reduced_dbs"
    data_dir: str = "$AF_DOWNLOAD_DIR"
    output_dir: str = './af_output/'
    num_replicates: int = 5

def write_fasta(sequences, sequence_names, num_replicates, fasta_file):
    for sequence_name, sequence in zip(sequence_names, sequences):
        for i in range(num_replicates):
            fasta_file.write(f'>{sequence_name}_{i}\n')
            fasta_file.write(f'{sequence}\n')
    fasta_file.flush()

def run_alphafold(sequences: list[str], sequence_names: list[str], params: AlphaFoldParams):
    """
    Run the AlphaFold prediction process using specified parameters.

    This function will generate a temporary FASTA file containing the specified sequences, and then execute the AlphaFold docker script using the given parameters. The temporary FASTA file is deleted once the script execution completes.

    Args:
        params (AlphaFoldParams): An instance of AlphaFoldParams dataclass which includes:
            sequences (list): A list of amino acid sequences for which the structure prediction should be done.
            sequence_name (str): A unique identifier for the sequence.
            max_template_date (str): The maximum date for templates used in the prediction.
            model_preset (str): The type of model preset to use for the prediction.
            db_preset (str): The type of database preset to use for the prediction.
            data_dir (str): The path to the directory containing necessary AlphaFold data.
            output_dir (str): The path to the directory where the output files should be saved.
            num_replicates (int): The number of replicate predictions to run for each sequence.

    Returns:
        None. The function will directly produce output files in the specified output directory.

    Raises:
        subprocess.CalledProcessError: If the AlphaFold docker script fails to execute.


    Notes:
    Output files of the following format:
        <target_name>/
            features.pkl
            ranked_{0,1,2,3,4}.pdb
            ranking_debug.json
            relax_metrics.json
            relaxed_model_{1,2,3,4,5}.pdb
            result_model_{1,2,3,4,5}.pkl
            timings.json
            unrelaxed_model_{1,2,3,4,5}.pdb
            msas/
                bfd_uniref_hits.a3m
                mgnify_hits.sto
                uniref90_hits.sto
    """
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as fasta_file:
        write_fasta(sequences, sequence_names, params.num_replicates, fasta_file)
        cmd = [
            'python3', 'docker/run_docker.py',
            '--fasta_paths=' + fasta_file.name,
            '--max_template_date=' + params.max_template_date,
            '--model_preset=' + params.model_preset,
            '--db_preset=' + params.db_preset,
            '--data_dir=' + params.data_dir,
            '--output_dir=' + params.output_dir,
        ]
        subprocess.run(cmd, check=True)
