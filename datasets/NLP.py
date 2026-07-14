"""IMDb / BERT text benchmark with a controllable ground-truth MI.

Each review is represented by a precomputed BERT embedding and carries a binary
sentiment label. Pairs ``(X, Y)`` are formed from two disjoint halves of the
data; a fraction ``percent_shuffle`` of the X labels is then scrambled, which
sets a *known* mutual information between the paired labels ``Lx`` and ``Ly``
(computed exactly by :func:`calculate_MI`). This gives a real-data task whose
target MI can still be dialled in.
"""

import os
import glob
import torch
import numpy as np


class TextDataset(torch.utils.data.Dataset):
    """Precomputed per-document embeddings for one sentiment class.

    Args:
        dataname: subdirectory holding the ``.npy`` embedding files.
        root: dataset root under which ``dataname`` lives.
        label: which sentiment class (0 or 1) to load.
        n_sample: optional cap on the number of files loaded.
    """

    def __init__(self, dataname="imdb.bert-imdb-finetuned",
                 root="../mi_benchmark/mibenchmark-main/dataset", label=0, n_sample=None):
        super(TextDataset, self).__init__()

        root = os.path.join(root, dataname)

        self.classes = [label]

        self.data = []
        self.counts = dict()
        self.label = []
        for idx, subclass in enumerate(self.classes):
            file_list = glob.glob(os.path.join(root, str(subclass), '*.npy'))
            file_list.sort()

            if n_sample not in (None, "None") and n_sample > 0:
                file_list = file_list[:n_sample]

            self.counts[idx] = len(self.data)
            self.data += [(filename, idx) for filename in file_list]
            self.label += [idx] * len(file_list)
        self.counts[len(self.classes)] = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Return ``(embedding_tensor, class_index)`` for document ``index``."""
        filename, target_idx = self.data[index]
        data = np.load(filename)
        return torch.Tensor(data), target_idx


def load_imdb_bert_dataset(percent_shuffle=0.5, data_range=[0, 5000], device='cuda:0'):
    """Build a paired IMDb/BERT dataset with a tunable label-MI.

    Two disjoint halves of the embeddings form ``X`` and ``Y``; a fraction
    ``percent_shuffle`` of the X rows (and their labels) is shuffled to weaken the
    label dependence. The ``device`` argument is accepted for backwards
    compatibility but the returned tensors are on CPU.

    Returns:
        Tuple ``(Xs, Ys, Lx, Ly)``: paired embeddings and their integer labels.
    """
    # load data from the two classes
    dataset0 = TextDataset(label=0)
    dataset1 = TextDataset(label=1)

    x0, l0 = torch.cat([dataset0[i][0] for i in range(12500)], dim=0), [0 for i in range(12500)]
    x1, l1 = torch.cat([dataset1[i][0] for i in range(12500)], dim=0), [1 for i in range(12500)]

    print('class 0', x0.size(), l0[0:5])
    print('class 1', x1.size(), l1[0:5])

    # construct X, Y, Lx, Ly
    start, end = data_range[0], data_range[1]
    N = end - start
    X = torch.cat([x0[start:end], x1[start:end]], dim=0).cpu().numpy()
    Y = torch.cat([x0[end:end+N], x1[end:end+N]], dim=0).cpu().numpy()
    Lx = np.array(l0[start:end] + l1[start:end])
    Ly = np.array(l0[end:end+N] + l1[end:end+N])

    print('X', X.shape, 'Y', Y.shape)

    # randomly permute the paired rows in unison
    N_samples = len(Lx)
    inds = np.arange(len(Lx))
    np.random.shuffle(inds)

    Xs = X[inds[:N_samples]].copy()
    Ys = Y[inds[:N_samples]].copy()
    Lx = Lx[inds[:N_samples]].copy()
    Ly = Ly[inds[:N_samples]].copy()

    rows_to_shuffle = int(percent_shuffle*len(Xs))

    # scramble the first `rows_to_shuffle` X rows and their labels identically
    # thanks to https://stackoverflow.com/questions/4601373/
    # better-way-to-shuffle-two-numpy-arrays-in-unison
    np.random.shuffle(Xs[:rows_to_shuffle])
    np.random.set_state(np.random.get_state())
    np.random.shuffle(Lx[:rows_to_shuffle])

    # sanity check
    target = 1-percent_shuffle + percent_shuffle*0.5
    print(f"Sanity check (percent identical labels, should be ~{target}):", (sum(Lx == Ly)/len(Lx)))

    # to torch
    Xs, Ys, Lx, Ly = torch.Tensor(Xs), torch.Tensor(Ys), torch.Tensor(Lx), torch.Tensor(Ly)
    return Xs, Ys, Lx, Ly


def calculate_MI(Lx, Ly):
    """Exact MI in nats between the two binary label vectors ``Lx`` and ``Ly``."""
    n = len(Lx)
    # joint probabilities
    p00 = ((Lx == 0).int() * (Ly == 0).int()).sum() * 1.0 / n
    p01 = ((Lx == 0).int() * (Ly == 1).int()).sum() * 1.0 / n
    p11 = ((Lx == 1).int() * (Ly == 1).int()).sum() * 1.0 / n
    p10 = ((Lx == 1).int() * (Ly == 0).int()).sum() * 1.0 / n
    # marginal probabilities
    pX0 = (Lx == 0).int().sum() * 1.0 / n
    pX1 = (Lx == 1).int().sum() * 1.0 / n
    pY0 = (Ly == 0).int().sum() * 1.0 / n
    pY1 = (Ly == 1).int().sum() * 1.0 / n

    print('p00, p01, p11, p10=', p00, p01, p11, p10)
    print('pX0, pX1, pY0, pY1=', pX0, pX1, pY0, pY1)

    MI = p00 * torch.log(p00/(pX0*pY0)) + p01 * torch.log(p01/(pX0*pY1)) + p10 * torch.log(p10/(pX1*pY0)) + p11 * torch.log(p11/(pX1*pY1))
    return MI.item()
