"""Requires diamond and blast installed! These are not in the repo environment
because it is not necessary for the main pipeline.
"""

import subprocess
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

WORKING_DIR= './diamond_1enh/'

def create_blast_db(dataset):
    i=0
    thermo_file = open(WORKING_DIR+'training_thermo.fasta', 'w')
    with open(WORKING_DIR+'training_meso_db.fasta', 'w') as f:
        loader = DataLoader(dataset, batch_size=10000)
        for batch in tqdm(loader):
            for meso_seq, thermo_seq in zip(batch['meso_seq'], batch['thermo_seq']):
                f.write(f">seq_{i}\n")
                f.write(meso_seq + "\n")
                thermo_file.write(f">seq_{i}\n")
                thermo_file.write(thermo_seq + "\n")
                i +=1
    thermo_file.close()
    
    # Create BLAST database
    cmd = ['makeblastdb', '-in', WORKING_DIR+'training_meso_db.fasta', '-dbtype', 'prot']
    subprocess.run(cmd, check=True)
    # cmd = f"diamond makedb --in {WORKING_DIR}training_meso_db.fasta -d {WORKING_DIR}training_meso_db.diamond".split()
    # subprocess.run(cmd, check=True)
    return f'{WORKING_DIR}training_meso_db.diamond'

def run_diamond():
    with open(f'{WORKING_DIR}query_1enh.fasta', 'w') as f:
        f.write("> enh1\n")
        f.write('RPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKI' + "\n")
        # f.write("> training_example\n")
        # f.write("MVFGDYLLQRAKEYGIQLSKEQYEKFNAYYELLLEWNSKINLTAITAPDEVAIKHIVDSLSAWDEKRFSPTAKIIDVGTGAGFPGIPLKIMHPGLQLTLLDSLAKRIKFLQTVAAKLNITDIEFLHGRAEEIGRRKPYREKFDIVFSRAVARMPVLCEYALPLVKKEGYFTALKGRQYEAEAGEAHNAIKVLGGVLAEVKPVKLPGLDDVRAVIYVQKVSKTPAVYPRKAGTPERKPLL" + "\n")
    
    cmd = f"blastp -db {WORKING_DIR}training_meso_db.fasta -query {WORKING_DIR}query_1enh.fasta -evalue 1.0 -outfmt 7 -out {WORKING_DIR}blast_out.tsv -num_threads 32  -word_size 3 -matrix blosum62 -qcov_hsp_perc 80 -dbsize 1000".split()

    # cmd = ['diamond', 'blastp', '-d', f'{WORKING_DIR}training_meso_db.diamond', '-q', f'{WORKING_DIR}query_1enh.fasta', '-o', f'{WORKING_DIR}diamond_1enh_result.tsv']
    # cmd.extend(['--outfmt',  '6', 'qseqid', 'sseqid', 'pident', 'length', 'slen', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
    # cmd.extend(['--threads', '32'])
    # cmd.append('--ultra-sensitive')
    # cmd.extend(['--matrix', 'BLOSUM62', '--gapopen', '11', '--gapextend', '1'])
    # cmd.extend(['--evalue', '0.8'])
    # cmd.append('-k0')
    # # cmd.extend(['--query-cover', '5'])
    # cmd.extend(['--unal', '1'])
    subprocess.run(cmd, check=True)
    return 'results.txt'

if __name__ == "__main__":

    ds = datasets.load_from_disk('../data/dataset')['train']
    db = create_blast_db(ds)

    # Replace 'MADSEQHERE' with your query protein sequence.
    run_diamond() # 1enh
