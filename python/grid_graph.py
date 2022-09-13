import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
#t-SNEで次元削減
from sklearn.manifold import TSNE
import torch

def get_mean(topic_num, x, topic, num):
    for i in range(topic_num):
        i = topic == num
        return np.mean(x[i], axis=0)

def grid_graph(computed_z, topic_num):
    tsne = TSNE(n_components=2, random_state = 1, n_iter = 1000)
    x = tsne.fit_transform(computed_z)
    zsoft = torch.softmax(computed_z , dim=1).detach().cpu().numpy()
    topic = []
    for i in zsoft:
        topic.append(np.argmax(i))
    topic = np.array(topic)

    plt.figure(figsize=(10, 8)) 
    plt.scatter(x[:, 0], x[:, 1], c=topic, cmap="rainbow", alpha=0.6)
    for i in range(topic_num):
        m = get_mean(topic_num, x, topic, i)
        plt.text(m[0], m[1], "{}".format(i), fontsize=20)

    plt.colorbar()
    plt.grid()
