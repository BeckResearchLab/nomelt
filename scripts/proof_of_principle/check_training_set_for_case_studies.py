"""Script to ensure that enh is not in the training set."""

from nomelt.blast import run_blast_search
import json
from datasets import load_from_disk, concatenate_datasets

ENH1 = "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK"
LovD = "MGSIIDAAAAADPVVLMETAFRKAVKSRQIPGAVIMARDASGNLNYTRCFGARTVRRDENNQLPPLQVDTPCRLASATKLLTTIMALQCMERGLVDLDETVDRLLPDLSAMPVLEGFDDAGNARLRERRGKITLRHLLTHTSGLSYVFLHPLLREYMAQGHLQSAEKFGIQSRLAPPAVNDPGAEWIYGANLDWAGKLVERATGLDLEQYLQENICAPLGITDMTFKLQQRPDMLARRADQTHRNSADGRLRYDDSVYFRADGEECFGGQGVFSGPGSYMKVLHSLLKRDGLLLQPQTVDLMFQPALEPRLEEQMNQHMDASPHINYGGPMPMVLRRSFGLGGIIALEDLDGENWRRKGSLTFGGGPNIVWQIDPKAGLCTLAFFQLEPWNDPVCRDLTRTFEHAIYAQYQQG"
LipA = "AEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDKTGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDPNQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN"


def do_one(seq, dataset):
    records = run_blast_search(seq, dataset, './tmp/blast/')
    r = records[0]
    alignments = r.alignments
    for a in alignments:
        hsp = a.hsps[0]
        e = hsp.expect
        s_cov = (hsp.sbjct_end - hsp.sbjct_start) / a.length
        q_cov = (hsp.query_end - hsp.query_start) / r.query_length
        break
    outs = {'e': e, 's_cov': s_cov, 'q_cov': q_cov}
    return outs

if __name__ == '__main__':

    dataset = load_from_disk('./data/dataset/')
    dataset = concatenate_datasets([dataset['train'], dataset['eval'], dataset['test']])
    dataset.save_to_disk('./tmp/dataset-all')
    
    enh_outs = do_one(ENH1, dataset)
    lovd_outs = do_one(LovD, dataset)
    lipa_outs = do_one(LipA, dataset)

    with open('./data/enh/training_data_homologs.json', 'w') as f:
        json.dump({'enh1': enh_outs, 'lovd': lovd_outs, 'lipa': lipa_outs}, f)
