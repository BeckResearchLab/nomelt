import subprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from Bio.Blast.NCBIXML import parse
import os

def create_blast_db(dataset, working_dir='./blast_workdir/'):
    i = 0
    db_file_path = f'{working_dir}training_meso_db.fasta'

    if os.path.exists(db_file_path):
        return db_file_path
    
    with open(db_file_path, 'w') as f:
        loader = DataLoader(dataset, batch_size=10000)
        for batch in tqdm(loader):
            for meso_seq, thermo_seq in zip(batch['meso_seq'], batch['thermo_seq']):
                f.write(f">seq_{i}\n{meso_seq}\n")
                i += 1

    cmd = ['makeblastdb', '-in', db_file_path, '-dbtype', 'prot']
    subprocess.run(cmd, check=True)

    return db_file_path

def parse_blast_output(output_path):
    with open(output_path, 'r') as f:
        records = [r for r in parse(f)]
    return records

def run_blast_search(query_sequence, dataset_path, working_dir='./blast_workdir/'):
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # Load dataset
    ds = datasets.load_from_disk(dataset_path)['train']
    
    # Create BLAST database
    db_path = create_blast_db(ds, working_dir)
    
    # Write query sequence to file
    query_file_path = f'{working_dir}query.fasta'
    with open(query_file_path, 'w') as f:
        f.write(f">query\n{query_sequence}\n")
    
    # Run BLAST
    output_path = f'{working_dir}blast_out.tsv'
    cmd = [
        'blastp', '-db', db_path, '-query', query_file_path,
        '-evalue', '1.0', '-outfmt', '5', '-out', output_path,
        '-num_threads', '32', '-word_size', '3',
        '-matrix', 'BLOSUM62', '-qcov_hsp_perc', '80'
    ]
    subprocess.run(cmd, check=True)
    
    # Parse and return alignments
    return parse_blast_output(output_path)