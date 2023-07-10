"""Uses MinHashing and Jaccard distance to remove datapoints within the dataset that are highly similar.

This functions to reduce data leakage - two proteins that are extremely similar in primary
sequence should likely not exist twice. Not that this differs from homology - we want homologous proteins to appear,
but not ones that are learly identical.

Pairwise alignment in order to confirm %id is too costly, so instead we borrow MinHash distance from
NLP: https://arxiv.org/abs/2107.06499

In order to produce a bit vector to MinHash, k-gram protein representation is used:
https://academic.oup.com/bioinformatics/article/34/9/1481/4772682

Deduplication code is modified from 
https://github.com/conceptofmind/Huggingface-deduplicate/blob/main/minhash_deduplication.py
"""
import json
import multiprocessing as mp
import re
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Type

from datasets import Dataset
import pandas as pd
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from dpu_utils.utils.iterators import ThreadedIterator
import nltk

import logging
logger = logging.getLogger(__name__)

def compute_protein_kgram(seq, k=3):
    """Find all k-grams in a amino acid sequence.
    
    No spaces are accepted.
    
    Parameters
    ----------
    seq: str
        Protein sequence
    k : int
        Size of grams
        
    Returns
    -------
    list of tuple of str
    """
    if " " in seq:
        raise ValueError(f"Ensure no spaces are in input: {seq}")
    else:
        return nltk.ngrams(seq, k)

def compute_min_hash(seq, k=3, num_perm=100):
    """Compute the MinHash of a protein sequence based on kgrams
    
    Parameters
    ----------
    seq: str of protein sequence
    k: size of k grams
    n: number of permutations for minhash
    """
    minhash = MinHash(num_perm=num_perm)
    grams = compute_protein_kgram(seq, k=k)
    for g in grams:
        minhash.update("".join(g).encode())
    return minhash


class DuplicationIndex:
    def __init__(
        self,
        *,
        duplication_jaccard_threshold: float = 0.95,
        num_perm: int=100
    ):
        self._duplication_jaccard_threshold = duplication_jaccard_threshold
        self._num_perm = num_perm
        self._index = MinHashLSH(threshold=self._duplication_jaccard_threshold, num_perm=self._num_perm)

        self._duplicate_clusters = defaultdict(set)

    def add(self, code_key: Tuple, min_hash: MinHash) -> None:
        """Add a key to _index (MinHashLSH)
        the min_hash is used to query closest matches based on the jaccard_threshold.
        The new key is either added to a existing cluster of one close match,
        or a new cluster is created. The clusters created in this way, depend on the order of add.
        Args:
            code_key (Tuple of (index, repo_name, path)):
                Theoritically any hasbale key. Here we use a tuple to retrieve the information later.
            min_hash: MinHash of the code_key.
        """
        close_duplicates = self._index.query(min_hash)
        if code_key in self._index.keys:
            logger.info(f"Found duplicate key {code_key}")
            return

        self._index.insert(code_key, min_hash)
        if len(close_duplicates) > 0:

            for base_duplicate in close_duplicates:
                if base_duplicate in self._duplicate_clusters:
                    self._duplicate_clusters[base_duplicate].add(code_key)
                    break
            else:
                self._duplicate_clusters[close_duplicates[0]].add(code_key)

    def get_duplicate_clusters(self) -> List[List[Dict]]:
        """Export the duplicate clusters.
        For each cluster, the first element is the base element of the cluster.
        The base element has an estimation jaccard similarity higher than the threshold with all the other elements.
        Returns:
            duplicate_clusters (List[List[Dict]]):
                List of duplicate clusters.
        """
        duplicate_clusters = []
        for base, duplicates in self._duplicate_clusters.items():
            cluster = [base] + list(duplicates)
            # reformat the cluster to be a list of dict
            cluster = [{"base_index": el} for el in cluster]
            duplicate_clusters.append(cluster)
        return duplicate_clusters

    def save(self, filepath) -> None:
        duplicate_clusters = self.get_duplicate_clusters()
        with open(filepath, "w") as f:
            json.dump(duplicate_clusters, f)

def _compute_min_hash(ins):
    sequence_key, (e, k, num_perm) = ins
    index, data = e
    if 'index' in data:
        index = data['index']
    min_hash = compute_min_hash(data[sequence_key], k=k, num_perm=num_perm)
    return index, min_hash

def _dataset_iter(dataset, k, num_perm, sequence_key):
    for e in dataset:
        yield (sequence_key, (e, k, num_perm))

def _minhash_iter(dataset_iterator: Type[Dataset], k, num_perm, sequence_key):
    with mp.Pool() as pool:
        for data in pool.imap_unordered(
            _compute_min_hash,
            ThreadedIterator(_dataset_iter(dataset_iterator, k=k, num_perm=num_perm, sequence_key=sequence_key), max_queue_size=10000),
            chunksize=100,
        ):
            if data is not None:
                yield data

def make_duplicate_clusters(
    dataset: Type[Dataset],
    jaccard_threshold: float,
    sequence_key: str = "protein_seq",
    num_perm: int=100,
    k: int = 3):
    """Find duplicate clusters in the dataset in two steps:
    1. Compute MinHash for each code snippet. MinHash is a tool for fast jaccard similarity estimation.
    This step is computed using an asynchronous multiprocessing pool, minhash_iter
    2. Find duplicate clusters. The computed MinHash is added sequentially to the DuplicationIndex.
    This step cannot be parallelized. So using asynchronous thread in the previous step helps to speed up the process.
    """
    di = DuplicationIndex(duplication_jaccard_threshold=jaccard_threshold, num_perm=num_perm)
    
    
    # prepare iterators
    for index, min_hash in tqdm(ThreadedIterator(_minhash_iter(enumerate(dataset), k=k, num_perm=num_perm, sequence_key=sequence_key), max_queue_size=100)):
        di.add(index, min_hash)
    
    # Returns a List[Cluster] where Cluster is List[str] with the index.
    return di.get_duplicate_clusters()

_shared_dataset = None


def _find_cluster_extremes_shared(cluster, jaccard_threshold, k, num_perm, sequence_key):
    """Find a reduced cluster such that each code in the origin cluster is similar to at least one code in the reduced cluster.
    Two codes are similar if their Jaccard similarity is above the threshold.
    Args:
        cluster (List[dict]):
           cluster is a list of dict, each dict contains the following keys:
                - base_index
                - repo_name
                - path
            This is a typical output of DuplicationIndex.get_duplicate_clusters()
        jaccard_threshold (float):
            threshold for Jaccard similarity.
            Two codes are similar if their Jaccard similarity is above the threshold.
    Returns:
        extremes (List[dict]):
            A reduced representation of the cluster. The field copies is added to each dict.
            The copies field indicates the number of similar codes in the cluster for a extreme.
    """
    extremes = []
    for element1 in cluster:
        code1 = compute_min_hash(_shared_dataset[element1["base_index"]][sequence_key], k=k, num_perm=num_perm)
        for element2 in extremes:
            code2 = compute_min_hash(_shared_dataset[element2["base_index"]][sequence_key], k=k, num_perm=num_perm)
            if code1.jaccard(code2) >= jaccard_threshold:
                element2["copies"] += 1
                break
        else:
            element1["copies"] = 1
            extremes.append(element1)
    return extremes


def find_extremes(cluster_list, dataset, jaccard_threshold, k, num_perm, sequence_key):
    """Call the _find_cluster_extremes_shared function in a parallel fashion.
    Args:
        cluster_list (List[List[Dict]]):
            each cluster is a list of dicts with the key base_index,
            referring to the index of the base code in the dataset.
        dataset (Type[Dataset]):
            dataset is used to access the content of the code snippets,
            using the base_index from the cluster_list.
            dataset is shared between all the processes using a glabal variable (any other way to share the dataset?),
            otherwise the multi processing is not speeded up.
        jaccard_threshold (float):
            the threshold for the jaccard similarity. The default value is 0.85
    Returns:
        extremes_list (List[Dict]):
            Each cluster is reduced to extremes.
            See _find_cluster_extremes_shared for the definition of extremes.
    """
    global _shared_dataset
    _shared_dataset = dataset
    extremes_list = []
    f = partial(_find_cluster_extremes_shared, jaccard_threshold=jaccard_threshold, k=k, num_perm=num_perm, sequence_key=sequence_key)
    with mp.Pool() as pool:
        for extremes in tqdm(
            pool.imap_unordered(
                f,
                cluster_list,
            ),
            total=len(cluster_list),
        ):
            extremes_list.append(extremes)
    return extremes_list


def deduplicate_dataset(
    dataset: Type[Dataset],
    sequence_key: str,
    jaccard_threshold: float = 0.85,
    num_perm: int=100,
    k: int = 3
) -> Tuple[Type[Dataset], List[List[Dict]]]:
    """Deduplicate the dataset using minhash and jaccard similarity.
    This function first generate duplicate clusters, then each cluster
    is reduced to the extremes that are similar to the other elements in the cluster.
    Codes are called similar if their Jaccard similarity is greater than jaccard_threshold (0.85 default).
    Args:
        dataset (Type[Dataset]):
            The dataset to deduplicate.
        jaccard_threshold (float, default=0.95):
            jaccard threshold to determine if two codes are similar
        num_perm (int, default=100)
            Number of permuations for MinHash
        k (int, default=3)
            size of k grams for representation
    Returns:
        duplicate_clusters Dict[Dict]
            keys are cluster indexes, values are cluster data, contained indexes, and extreme indexes
            cluster data is dict of raw cluster from original implementation
              list of dict with keys "base_index": the id, "is_extreme": boolean, "copies": number of copies (if extreme)
        cluster_summary DataFrame
            Contains cluster id, number of items, number of extreme items
        duplicate_indices Set[int]
            Set of indices of duplicate items in clusters
        extreme_indices Set[int]
            Set of indices of extreme items in clusters
    """
    duplicate_clusters = make_duplicate_clusters(dataset, jaccard_threshold, sequence_key=sequence_key, k=k, num_perm=num_perm)
    duplicate_indices = set(x["base_index"] for cluster in duplicate_clusters for x in cluster)
    extreme_dict = {}
    extremes_clusters = find_extremes(duplicate_clusters, dataset, jaccard_threshold, k=k, num_perm=num_perm, sequence_key=sequence_key)
    for extremes in extremes_clusters:
        for element in extremes:
            extreme_dict[element["base_index"]] = element
    extreme_indices = set(extreme_dict.keys())

    # update duplicate_clusters
    for cluster in duplicate_clusters:
        for element in cluster:
            element["is_extreme"] = element["base_index"] in extreme_dict
            if element["is_extreme"]:
                element["copies"] = extreme_dict[element["base_index"]]["copies"]

    cluster_summary = []
    new_duplicate_clusters = {}
    for i, cluster in enumerate(duplicate_clusters):
        cluster_summary.append({
            "cluster_id": i,
            "cluster_size": len(cluster),
            "extreme_size": sum(x["is_extreme"] for x in cluster),
        })
        cluster_ids = set(x["base_index"] for x in cluster)
        cluster_extreme_ids = set(x["base_index"] for x in cluster if x["is_extreme"])
        new_duplicate_clusters[i] = {'cluster': cluster, 'ids': cluster_ids, 'extreme_ids': cluster_extreme_ids}
    cluster_summary = pd.DataFrame(cluster_summary)
    

    logger.info(f"Original dataset size before deduplication: {len(dataset)}")
    logger.info(f"Number of duplicate clusters: {len(duplicate_clusters)}")
    logger.info(f"Items in duplicate cluster: {len(duplicate_indices)}")
    logger.info(f"Extreme items in duplicate cluster: {len(extreme_dict)}")
    logger.info(f"Number of items not in duplicate clusters: {len(dataset) - len(duplicate_indices)}")
    logger.info(f"Summary of duplicate clusters: \n{cluster_summary.to_string()}")

    return new_duplicate_clusters, cluster_summary, duplicate_indices, extreme_indices
