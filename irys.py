from sklearn.datasets import load_iris
dane = load_iris()

print(dane.keys())

print("ilość danych z rozmiarami płatków: {}".format(dane.data.shape))

print("dane iris zawierają {} cech o nazwach: {}".format(dane.data.shape[1], dane.feature_names))
print("dane iris zawierają {} gatunków o nazwach: {}".format(dane.target.shape[0], dane.target_names))

#dane ze 150 pomiarów tworzą 3 kolejne bloki po 50 dla każdego gatunku
x=dane["data"][:50, 2] #pierwsze 50 pomiarów długości płatka (gatunek setosa)
y=dane["data"][:50, 3] #piewsze 50 pomiarów szerokości płatka (gatunek setosa)


#import matplotlib.pyplot as plt

#plt.plot(x, y, "go", alpha=0.3)
#plt.xlabel("długość płatka setosa (cm)", fontsize=14)
#plt.ylabel("szerokość płatka setosa (cm)", rotation=90, fontsize=14)
#plt.axis([0, 2, 0, 1])

#plt.show() #pokaż interaktywny wykres

from sklearn.linear_model import LinearRegression
import numpy as np

x = x.reshape(-1,1) #w x powinna być lista list - tak chyba najprościej ją zrobić

l_r = LinearRegression().fit(x, y)

#print(l_r.get_params())

x_pred = [[0], [2]] # policzmy dla tych dwu szerokosci
y_pred = l_r.predict(x_pred) #jaka powinny miec dlugosc w naszym modelu

plt.plot(x, y, "go", alpha=0.3) 
plt.plot(x_pred, y_pred, "r-")
plt.xlabel("długość płatka setosa (cm)", fontsize=14)
plt.ylabel("szerokość płatka setosa (cm)", rotation=90, fontsize=14)
plt.axis([0, 2, 0, 1])

plt.savefig("pomiary_setosa.png", dpi=300) #zapisz wykres w formacie png
plt.show() #pokaż interaktywny wykres
