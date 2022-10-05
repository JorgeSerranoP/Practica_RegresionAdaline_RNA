import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Funcion para desnormalizar datos
def desnormalizar (max, min, datos):
    denormalized = []
    for i in range(len(datos)):
        denormalized.append((datos[i] * (max - min)) + min)
    return denormalized

tabla = pd.ExcelFile("comparacionSalidas.xlsx")
datos = tabla.parse("Hoja1")
datos = np.array(datos)
obtenida = datos[:,0]
esperada = datos[:,1]
patronTest = len(esperada)

# Desnormalizar los datos
max=82.6
min=2.33   
denormalized = desnormalizar(max, min, obtenida)
denormalized2 = desnormalizar(max, min, esperada)

plt.ylabel('Salida',Fontsize = 12)
plt.xlabel('Patron',Fontsize = 12)
plt.title("MLP")
x = np.arange(patronTest)
y = [sorted(denormalized2), sorted(denormalized)]
labels = ["Salidas deseadas", "Salidas obtenidas"]
for y_arr, label in zip(y, labels):
    plt.plot(x, y_arr, label=label)
plt.legend(loc='upper right')
plt.show()

 # Fichero salidas de la red desnormalizados
f = open ('output_expected_obtained.txt','w')
for i in range(patronTest):
    f.write(str(denormalized[i]) + "\t" + str(denormalized2[i]) + "\n")
f.close()
