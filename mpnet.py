import pickle
from functools import partial

import numpy as np

import faiss

hyper_pass_path = "data/evaluation_passage_embeddings_64_parameterization_mpnet.npy"
hyper_query_path = "data/evaluation_query_embeddings_64_parameterization_mpnet.npy"

with open(r"data/qidx2gold_pidx.pkl", "rb") as f:
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


def recall_at(K, I, a, queries):
    return np.mean((I[:, :K, None] == a[queries, None, :]).any(axis=(-1, -2)))


def test_index(index, pass_data, query_data, queries):
    assert index.is_trained
    index.add(pass_data)  # add vectors to the index
    print("Done building index")
    _, I = index.search(query_data[queries], K)

    r10 = recall_at(10, I, a, queries)
    r100 = recall_at(100, I, a, queries)
    r1000 = recall_at(1000, I, a, queries)

    print(f"Recall@10: {r10:.4f}")
    print(f"Recall@100: {r100:.4f}")
    print(f"Recall@1000: {r1000:.4f}")


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
    test_index(index, hyper_pass, hyper_query, q[:Q])


if __name__ == "__main__":
    test_hyper_hnsw()
    test_hyper_flat()
