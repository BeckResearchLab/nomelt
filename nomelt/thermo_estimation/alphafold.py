"""Runs AF """

import os
import subprocess
from dataclasses import dataclass
import tempfile

@dataclass
class AlphaFoldParams:
    sequence: str = "<SEQUENCE>"
    sequence_name: str = "sequence"
    max_template_date: str = "2021-11-01"
    model_preset: str = "monomer"
    db_preset: str = "reduced_dbs"
    data_dir: str = "$AF_DOWNLOAD_DIR"
    output_dir: str
    num_replicates: int = 5

def write_fasta(sequence, sequence_name, num_replicates, fasta_file):
    for i in range(num_replicates):
        fasta_file.write(f'>{sequence_name}_{i}\n')
        fasta_file.write(f'{sequence}\n')
    fasta_file.flush()

def run_alphafold(params):
    """
    Run the AlphaFold prediction process using specified parameters.

    This function will generate a temporary FASTA file containing the specified sequence, and then execute the AlphaFold docker script using the given parameters. The temporary FASTA file is deleted once the script execution completes.

    Args:
        params (AlphaFoldParams): An instance of AlphaFoldParams dataclass which includes:
            sequence (str): The amino acid sequence for which the structure prediction should be done.
            sequence_name (str): A unique identifier for the sequence.
            max_template_date (str): The maximum date for templates used in the prediction.
            model_preset (str): The type of model preset to use for the prediction.
            db_preset (str): The type of database preset to use for the prediction.
            data_dir (str): The path to the directory containing necessary AlphaFold data.
            output_dir (str): The path to the directory where the output files should be saved.
            num_replicates (int): The number of replicate predictions to run.

    Returns:
        None. The function will directly produce output files in the specified output directory.

    Raises:
        subprocess.CalledProcessError: If the AlphaFold docker script fails to execute.
    """
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as fasta_file:
        write_fasta(params.sequence, params.sequence_name, params.num_replicates, fasta_file)
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

