import json

import numpy as np

import faiss

hyper_path = "data/poincare_glove_100D_cosh-dist-sq_vocab50k.npz"
vanilla_path = "data/vanilla_glove_100D_vocab50k.npz"


def load_data(path):
    loaded = np.load(path)
    return loaded["words"], loaded["data"].astype(np.float32)


vwords, vdata = load_data(vanilla_path)
hwords, hdata = load_data(hyper_path)

faiss.normalize_L2(vdata)

d = vdata.shape[-1]
k = 5
q = 5237


def test_index(index, data, words):
    assert index.is_trained

    index.add(data)  # add vectors to the index

    D, I = index.search(data[q, np.newaxis], k)
    print(D)
    print(I)
    print(words[I])


def test_vanilla_flat():
    print("VANILLA FLAT")
    index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)  # build the index
    test_index(index, vdata, vwords)


def test_vanilla_ivf():
    print("VANILLA IVF")
    nlist = 100
    quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    assert not index.is_trained
    index.train(vdata)
    assert index.is_trained

    test_index(index, vdata, vwords)


def test_vanilla_hnsw():
    print("VANILLA HNSW")
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # build the index
    test_index(index, vdata, vwords)


def test_hyper_flat():
    print("HYPER FLAT")
    index = faiss.IndexFlat(d, faiss.METRIC_Poincare)  # build the index
    test_index(index, hdata, hwords)


def test_hyper_ivf():
    print("HYPER IVF")
    nlist = 100
    quantizer = faiss.IndexFlat(d, faiss.METRIC_Poincare)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    assert not index.is_trained
    index.train(vdata)
    assert index.is_trained

    test_index(index, hdata, hwords)


def test_hyper_hnsw():
    print("HYPER HNSW")
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_Poincare)  # build the index
    test_index(index, hdata, hwords)


if __name__ == "__main__":
    test_vanilla_flat()
    test_hyper_flat()

    # K means crashes atm
    test_vanilla_ivf()
    test_hyper_ivf()

    test_vanilla_hnsw()
    test_hyper_hnsw()
