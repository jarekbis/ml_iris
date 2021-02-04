#! /usr/bin/python

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

# tak tworzymy kolejkę, w której najpierw wykonamy przeskalowanie,
#a później algorytmem regresji logistystycznej (inna miara dokładności dopasowania) zrobimy dopasowanie
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

dane = load_iris()

# teraz obliczamy dopasowanie
pipe.fit(dane["data"][:,2:], dane["target"])
#Pipeline(steps=[('standardscaler', StandardScaler()),('logisticregression', LogisticRegression())])

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, target_names):
    res = 0.02
    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # zamaluje powierzchnie decyzyjne
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),
                           np.arange(x2_min, x2_max, res))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # rysuje wykres wszystkich próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    marker=markers[idx], label=target_names[cl])

plot_decision_regions(dane["data"][:,2:], dane["target"], pipe, dane.target_names)
plt.xlabel('Długość płatka [cm]')
plt.ylabel('Szerokość płatka [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./irys_klasyf.png', dpi=300)
plt.show()

