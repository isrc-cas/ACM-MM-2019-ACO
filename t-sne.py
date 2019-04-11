import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_domains(domain1, domain2, name1, name2):
    x = np.concatenate((domain1, domain2), axis=0)
    print(x.shape)
    t_sne = TSNE(n_components=2)
    y = t_sne.fit_transform(x)

    labels = np.zeros((x.shape[0],), dtype=np.int32)
    labels[domain1.shape[0]:] = 1

    plt.scatter(y[labels == 0, 0], y[labels == 0, 1], marker='^', label=name1)
    plt.scatter(y[labels == 1, 0], y[labels == 1, 1], marker='o', label=name2)
    plt.legend(loc=3)


def load_vectors(path):
    arr = np.load(path)
    return arr.reshape(arr.shape[0], -1)


if __name__ == '__main__':
    source_vectors1 = load_vectors('vectors_v2/group1_source_domain_v2.npy')
    target_vectors1 = load_vectors('vectors_v2/group1_target_domain_v2.npy')

    source_vectors2 = load_vectors('vectors_v2/group2_source_domain_v2.npy')
    target_vectors2 = load_vectors('vectors_v2/group2_target_domain_v2.npy')

    source_vectors3 = load_vectors('vectors_v2/group4_source_domain.npy')
    target_vectors3 = load_vectors('vectors_v2/group4_target_domain.npy')

    plt.figure(num=1, figsize=(12, 8))
    plt.subplot(131)
    plt.xlabel('(a)')
    plot_domains(source_vectors1, target_vectors1, 'Source Domain (w/o GAN)', 'Target Domain')

    plt.subplot(132)
    plt.xlabel('(b)')
    plot_domains(source_vectors2, target_vectors2, 'Source Domain (w/ GAN)', 'Target Domain')

    plt.subplot(133)
    plt.xlabel('(c)')
    plot_domains(source_vectors3, target_vectors3, 'Source Domain (w/ GAN+DP)', 'Target Domain')
    plt.show()
