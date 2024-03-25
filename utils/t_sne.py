from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pdb
import numpy as np
# %config InlineBackend.figure_format = "svg"
# digits = load_digits()

# X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
# X_pca = PCA(n_components=2).fit_transform(digits.data)

# font = {"color": "darkred",
#         "size": 13,
#         "family" : "serif"}
# # plt.style.use("dark_background")
# plt.figure(figsize=(8.5, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6,
#             cmap=plt.cm.get_cmap('rainbow', 10),s=5)
# plt.title("t-SNE", fontdict=font)
# cbar = plt.colorbar(ticks=range(10))
# cbar.set_label(label='digit value', fontdict=font)
# plt.clim(-0.5, 9.5)
# plt.subplot(1, 2, 2)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6,
#             cmap=plt.cm.get_cmap('rainbow', 10),s=5)
# plt.title("PCA", fontdict=font)
# cbar = plt.colorbar(ticks=range(10))
# cbar.set_label(label='digit value', fontdict=font)
# plt.clim(-0.5, 9.5)
# plt.tight_layout()
# plt.show()


def t_SNE_show(data, label):
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)
    plt.figure(figsize=(10.0, 10.0))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10),s=5)
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    # plt.show()
    plt.savefig()

def mat_show():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.ticker as ticker
    import torch
    a = torch.randn(12, 2)
    b = a.softmax(dim=1)
    c = a.softmax(dim=0).transpose(0, 1)
    print(a, '\n',  b, '\n', c)
    d = b.matmul(c)
    print(d)

    d = d.numpy()

    variables = ['A','B','C','X','A1','B1','C1','X1','A2','B2','C2','X2']
    labels = ['ID_0','ID_1','ID_2','ID_3','ID1_0','ID1_1','ID1_2','ID1_3','ID2_0','ID2_1','ID2_2','ID2_3']

    df = pd.DataFrame(d, columns=variables, index=labels)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))

    plt.show()
    plt.savefig()
