#!/usr/bin/env python3

"""

Split a FASTA file into chunks and submit hmmsearch jobs to SLURM.

Designed to be run from the root of the project directory.

"""

import os

import sys

import math

import argparse

import subprocess

import logging

from pathlib import Path

from Bio import SeqIO

 

# Set up logging

logger = logging.getLogger(__name__)

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(levelname)s - %(message)s'

)

 

def chunk_fasta(input_fasta, chunk_size, output_dir):

    """

    Split a FASTA file into chunks of approximately equal size.

   

    Args:

        input_fasta (str): Path to input FASTA file

        chunk_size (int): Number of sequences per chunk

        output_dir (str): Directory to write chunk files

   

    Returns:

        int: Number of chunks created

    """

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

   

    # Count total sequences

    logger.info(f"Counting sequences in {input_fasta}")

    total_seqs = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))

    logger.info(f"Found {total_seqs} sequences")

   

    # Calculate number of chunks needed

    num_chunks = math.ceil(total_seqs / chunk_size)

    logger.info(f"Splitting into {num_chunks} chunks of ~{chunk_size} sequences each")

   

    # Split the file

    records = list(SeqIO.parse(input_fasta, "fasta"))

    for i in range(num_chunks):

        start = i * chunk_size

        end = min((i + 1) * chunk_size, total_seqs)

        chunk_path = os.path.join(output_dir, f"chunk_{i+1}.fasta")

        with open(chunk_path, "w") as handle:

            SeqIO.write(records[start:end], handle, "fasta")

        logger.debug(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")

   

    return num_chunks

 

def create_slurm_script(chunk_path, pfam_path, output_dir, account, job_index):

    """

    Create a SLURM script for running hmmsearch on a chunk.

   

    Args:

        chunk_path (str): Path to FASTA chunk file

        pfam_path (str): Path to Pfam HMM file

        output_dir (str): Directory for hmmsearch output

        account (str): SLURM account name

        job_index (int): Chunk/job number for naming

   

    Returns:

        str: Path to created SLURM script

    """

    script = f"""#!/bin/bash

#SBATCH --account={account}

#SBATCH --partition=compute

#SBATCH --job-name=hmm_{job_index}

#SBATCH --output=./data/pfam/scripts/hmm_{job_index}_%j.out

#SBATCH --cpus-per-task=10

#SBATCH --time=24:00:00

#SBATCH --mem=100G

 

# Echo job info

echo "Running hmmsearch job {job_index}"

echo "Input: {chunk_path}"

echo "Output: {output_dir}/domtblout_{job_index}.txt"

echo "Started at: $(date)"

 

# Run hmmsearch

source ~/.bashrc
conda activate hmmer

hmmsearch --cpu 10 --domE 1e-5 --domtblout {output_dir}/domtblout_{job_index}.txt {pfam_path} {chunk_path}

 

echo "Finished at: $(date)"

"""

    script_path = os.path.join(output_dir, f"slurm_script_{job_index}.sh")

    with open(script_path, "w") as f:

        f.write(script)

    return script_path

 

def parse_args():

    parser = argparse.ArgumentParser(description="Split FASTA file and submit hmmsearch jobs to SLURM")

    parser.add_argument(

        "--fasta",

        type=str,

        default="./data/sequences.fasta",

        help="Path to input FASTA file (default: ./data/sequences.fasta)"

    )

    parser.add_argument(

        "--pfam",

        type=str,

        required=True,

        help="Path to Pfam HMM file"

    )

    parser.add_argument(

        "--outdir",

        type=str,

        default="./data/pfam_search",

        help="Output directory (default: ./data/pfam_search)"

    )

    parser.add_argument(

        "--chunk-size",

        type=int,

        default=1000,

        help="Number of sequences per chunk (default: 1000)"

    )

    parser.add_argument(

        "--account",

        type=str,

        required=True,

        help="SLURM account name"

    )

    return parser.parse_args()

 

def main():

    args = parse_args()

 

    # Validate input paths

    if not os.path.exists(args.fasta):

        logger.error(f"Input FASTA file not found: {args.fasta}")

        sys.exit(1)

    if not os.path.exists(args.pfam):

        logger.error(f"Pfam HMM file not found: {args.pfam}")

        sys.exit(1)

 

    # Create output directory structure

    chunks_dir = os.path.join(args.outdir, "chunks")

    scripts_dir = os.path.join(args.outdir, "scripts")

    results_dir = os.path.join(args.outdir, "results")

   

    for d in [chunks_dir, scripts_dir, results_dir]:

        os.makedirs(d, exist_ok=True)

        logger.info(f"Created directory: {d}")

 

    # Split the FASTA file

    logger.info("Splitting FASTA file into chunks...")

    num_chunks = chunk_fasta(args.fasta, args.chunk_size, chunks_dir)

   

    # Create and submit jobs

    logger.info("Creating and submitting SLURM jobs...")

    for i in range(num_chunks):

        chunk_path = os.path.join(chunks_dir, f"chunk_{i+1}.fasta")

        script_path = create_slurm_script(

            chunk_path,

            args.pfam,

            scripts_dir,

            args.account,

            i+1

        )

       

        # Make script executable

        os.chmod(script_path, 0o755)

       

        # Submit job

        try:

            subprocess.run(["sbatch", script_path], check=True)

            logger.info(f"Submitted job {i+1}/{num_chunks}")

        except subprocess.CalledProcessError as e:

            logger.error(f"Failed to submit job {i+1}: {e}")

            continue

 

    logger.info(f"Submitted {num_chunks} jobs. Results will be in: {results_dir}")

 

if __name__ == "__main__":

    main()