import numpy as np

hyper_data = "poincare_glove_100D_cosh-dist-sq_vocab50k.txt"
vanilla_data = "vanilla_glove_100D_vocab50k.txt"


for path in [hyper_data, vanilla_data]:
    with open(path, "r") as f:
        data = None
        words = []
        i = 0
        for line in f:
            word, *dims = line.split()
            words.append(word)

            dimarray = np.asarray(dims).astype(float)
            if data is None:
                data = dimarray
            else:
                data = np.vstack([data, dimarray])
            if i % 1000 == 0:
                print(i, word)
            i += 1
        np.savez(path.replace(".txt", ".npz"), words=words, data=data)
