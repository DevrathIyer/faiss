import pickle
import time
from functools import partial

import numpy as np

import faiss

data_dir = "/drive_sdc/deviyer/faiss_data/"
hyper_pass_path = f"{data_dir}data/evaluation_passage_embeddings_64_parameterization_mpnet.npy"
hyper_query_path = f"{data_dir}data/evaluation_query_embeddings_64_parameterization_mpnet.npy"

with open(f"{data_dir}data/qidx2gold_pidx.pkl", "rb") as f:
    qidx2gold_pidx = pickle.load(f)

hyper_query = np.load(hyper_query_path)
hyper_pass = np.load(hyper_pass_path)

q = np.array(list(qidx2gold_pidx.keys()))

print("[setup] Generating answers...")
max_p = max([len(v) for v in qidx2gold_pidx.values()])
a = np.full((hyper_query.shape[0], max_p), -1)
for k, v in qidx2gold_pidx.items():
    a[k, : len(v)] = np.array(list(v))
print("[setup] Done!")

d = hyper_pass.shape[-1]
K = 1000
Q = 100
print(q.shape)
print(hyper_pass.shape)


def recall_at(K, I, a, queries):
    return np.mean((I[:, :K, None] == a[queries, None, :]).any(axis=(-1, -2)))


def test_hnsw_index(index, pass_data, query_data, queries, filename):
    if index is None:
        start = time.time()
        index = faiss.read_index(f"{data_dir}{filename}")
        print("Done reading index in %.2f seconds" % (time.time() - start))
    else:
        start = time.time()
        assert index.is_trained
        index.add(pass_data)  # add vectors to the index
        print("Done building index in %.2f seconds" % (time.time() - start))

    start = time.time()
    _, I = index.search(query_data[queries], K)
    print("Done searching in %.2f seconds" % (time.time() - start))
    stats = faiss.cvar.hnsw_stats

    print("=====STATS=====")
    print("Queries searched: %d" % stats.n1)
    print("Queries failed: %d" % stats.n2)
    print("Distances computed: %d" % stats.ndis)
    print("Edges Traversed: %d" % stats.nhops)

    print("=====STATS(/Query)=====")
    print("Distances computed: %.02f" % (stats.ndis/stats.n1))
    print("Edges Traversed: %.02f" % (stats.nhops/stats.n1))
    print("=======================")

    start = time.time()
    r10 = recall_at(10, I, a, queries)
    r100 = recall_at(100, I, a, queries)
    r1000 = recall_at(1000, I, a, queries)

    print("Done calculating recall in %.2f seconds" % (time.time() - start))
    print(f"Recall@10: {r10:.4f}")
    print(f"Recall@100: {r100:.4f}")
    print(f"Recall@1000: {r1000:.4f}")
    
    print("Writing index to disk...")
    faiss.write_index(index, f"{data_dir}{filename}")
    

def test_hyper_flat():
    print("HYPER FLAT")
    index = faiss.IndexFlat(d, faiss.METRIC_Poincare)  # build the index
    test_index(index, hyper_pass, hyper_query, q[:Q])


def test_hyper_ivf():
    print("HYPER IVF")
    nlist = 100
    quantizer = faiss.IndexFlat(d, faiss.METRIC_Poincare)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    assert not index.is_trained
    index.train(hyper_pass)
    assert index.is_trained

    test_index(index, hyper_pass, hyper_query, q[:Q])


def test_hyper_hnsw():
    print("HYPER HNSW")
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_Poincare)  # build the index
    test_index(index, hyper_pass, hyper_query, q, filename="hnsw2.bin")
    #test_index(index, hyper_pass, hyper_query, q, filename=None)
    #faiss.write_index(index, "hnsw2.bin")

def test_arange_hyper_hnsw(read=False):
    print("HYPER ARANGE HNSW")
    
    if not read:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_Poincare)  # build the index
        levels = np.concatenate((np.repeat(1, 5), np.repeat(2, 250), np.repeat(3, 8405),np.repeat(4, 267641), np.repeat(5, 8565522))).astype(np.int32)
        assert levels.shape[0] == hyper_pass.shape[0]
        faiss.copy_array_to_vector(levels, index.hnsw.levels)

    test_hnsw_index(None if read else index, hyper_pass, hyper_query, q, filename="hnsw_arange_backup.bin")

def test_rand_hyper_hnsw(read=False):
    print("HYPER RAND HNSW")
    
    if not read:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_Poincare)  # build the index
        levels = np.concatenate((np.repeat(1, 5), np.repeat(2, 250), np.repeat(3, 8405),np.repeat(4, 267641), np.repeat(5, 8565522))).astype(np.int32)
        levels = np.random.permutation(levels)
        assert levels.shape[0] == hyper_pass.shape[0]
        faiss.copy_array_to_vector(levels, index.hnsw.levels)

    test_hnsw_index(None if read else index, hyper_pass, hyper_query, q, filename="hnsw_rand.bin")

def test_norm_hyper_hnsw(read=False):
    print("HYPER NORM HNSW")
    
    if not read:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_Poincare)  # build the index
        levels = np.concatenate((np.repeat(1, 5), np.repeat(2, 250), np.repeat(3, 8405),np.repeat(4, 267641), np.repeat(5, 8565522))).astype(np.int32)
        assert levels.shape[0] == hyper_pass.shape[0]
        levels = faiss.copy_array_to_vector(levels, index.hnsw.levels)

        norms = np.linalg.norm(hyper_pass, axis=-1)
        norm_idx = np.argsort(norms)
        sort_hyper_pass = hyper_pass[norm_idx]
        print(sort_hyper_pass.shape)

    test_hnsw_index(None if read else index, sort_hyper_pass, hyper_query, q, filename="hnsw_norm.bin")

def test_rev_norm_hyper_hnsw(read=False):
    print("HYPER REV NORM HNSW")
    
    if not read:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_Poincare)  # build the index
        levels = np.concatenate((np.repeat(1, 5), np.repeat(2, 250), np.repeat(3, 8405),np.repeat(4, 267641), np.repeat(5, 8565522))).astype(np.int32)
        assert levels.shape[0] == hyper_pass.shape[0]
        levels = faiss.copy_array_to_vector(levels, index.hnsw.levels)

        norms = np.linalg.norm(hyper_pass, axis=-1)
        norm_idx = np.argsort(norms)
        sort_hyper_pass = (hyper_pass[norm_idx])[::-1]
        print(sort_hyper_pass.shape)

    test_hnsw_index(None if read else index, sort_hyper_pass, hyper_query, q, filename="hnsw_rev_norm.bin")

if __name__ == "__main__":
    test_hyper_flat()
    # test_hyper_hnsw()
    #test_rand_hyper_hnsw()
    #test_arange_hyper_hnsw(read=True)
    #test_rev_norm_hyper_hnsw()
    #test_norm_hyper_hnsw()
