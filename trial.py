from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding
from utils import files_utils
from sklearn.manifold import TSNE
from custom_types import *
from matplotlib import pyplot as plt


def metric(x, y):
    np.linalg.norm(x)

    return -(x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))


def main():
    path = r"C:\Users\hertz\Downloads\mc_latent_train.npy"
    emb = np.load(path, allow_pickle=True).tolist()
    all_emb = [item for key, item in emb.items()]
    all_len = [len(item) for key, item in emb.items()]
    labels_names = [key for key, item in emb.items()]
    all_emb = np.concatenate(all_emb)
    embedding = SpectralEmbedding(n_components=2)
    tsne = TSNE(n_components=2, random_state=0, metric=metric)
    X_transformed = tsne.fit_transform(all_emb)
    # X_transformed = embedding.fit_transform(all_emb)
    labels = np.zeros(X_transformed.shape[0])
    cur = 0
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, length in enumerate(all_len):
        labels[cur: cur + length] = i
        cur += length

    for i, (c, label) in enumerate(zip(colors, labels_names)):
        plt.scatter(X_transformed[labels == i, 0], X_transformed[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()