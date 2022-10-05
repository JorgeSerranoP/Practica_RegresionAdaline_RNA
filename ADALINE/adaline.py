# Importar bibliotecas
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ADALINE():
    # Constructor
    def __init__(self,dEntrenamiento,dValidacion,dTest,xiEntrenamiento,xiValidacion,xiTest,patronEntrenamiento,patronValidacion,patronTest,wi,fac_ap,ciclos,w_ajustado):
        self.dEntrenamiento = dEntrenamiento # Valores deseados entrenamiento
        self.dValidacion = dValidacion # Valores deseados validacion
        self.dTest = dTest # Valores deseados test
        self.xiEntrenamiento = xiEntrenamiento # Datos de entrada entrenamiento
        self.xiValidacion = xiValidacion # Datos de entrada validacion
        self.xiTest = xiTest # Datos de entrada test
        self.patronEntrenamiento = patronEntrenamiento # Numero de muestras (filas) entrenamiento
        self.patronValidacion = patronValidacion # Numero de muestras (filas) validacion
        self.patronTest = patronTest # Numero de muestras (filas) test
        self.wi = wi # Pesos asociados
        self.fac_ap = fac_ap # Factor de aprendizaje
        self.ciclos = ciclos # Numero de ciclos
        self.yEntrenamiento = 0 # Salida de la red entrenamiento
        self.yValidacion = 0 # Salida de la red validacion
        self.yTest = 0 # Salida de la red test
        self.w_ajustado = w_ajustado
    
    def Entrenamiento(self):
        E_ac = 0 # Error actual
        Ew = 1 # Error cuadratico medio
        E_redEntrenamiento = [] # Error de la red entrenamiento
        E_redValidacion = [] # Error de la red validacion
        E_total = 0 
        EwModelo = 1
        ciclosModelo = 0
        wiModelo = []
        
        while (self.ciclos < 60000):
            for i in range(self.patronEntrenamiento):
                self.yEntrenamiento = sum(self.xiEntrenamiento[i,:] * self.wi) # Calculo de la salida de la red
                E_ac = (self.dEntrenamiento[i] - self.yEntrenamiento) # Calculo del error
                self.wi = self.wi + (self.fac_ap * E_ac * self.xiEntrenamiento[i,:]) # Ajustar los pesos
                
                E_total = E_total + ((E_ac)**2)
            
            # Calcular el error cuadratico medio entrenamiento
            Ew = ((1/self.patronEntrenamiento) * (E_total))
            E_total = 0
            E_redEntrenamiento.append(np.abs(Ew))

            # Calcular el error cuadratico medio validacion
            for i in range(self.patronValidacion):
                self.yValidacion = sum(self.xiValidacion[i,:] * self.wi) # Calculo de la salida de la red
                E_ac = (self.dValidacion[i] - self.yValidacion) # Calculo del error
                
                E_total = E_total + ((E_ac)**2)
            Ew = ((1/self.patronValidacion) * (E_total))
            
            # Guardamos los valores del modelo y el numero de ciclos optimo
            if(Ew < EwModelo):
                EwModelo = Ew
                ciclosModelo = self.ciclos
                wiModelo = self.wi

            E_total = 0
            E_redValidacion.append(np.abs(Ew))

            self.ciclos += 1
        return wiModelo, self.ciclos, E_redEntrenamiento, E_redValidacion, ciclosModelo 

    def F_operacion(self):
        salida = []
        E_redTest = 1 # Error de la red test
        E_ac = 0 
        Ew = 1 # Error cuadratico medio
        E_total = 0

        for j in range(self.patronTest):
            self.yTest = sum(self.xiTest[j,:] * self.w_ajustado) # Calculo de la salida de la red
            salida.append(self.yTest)
            E_ac = (self.dTest[j] - self.yTest) # Calculo del error 
            E_total = E_total + ((E_ac)**2)

        Ew = ((1/self.patronTest) * (E_total))
        E_redTest = Ew
        return salida, E_redTest

# Ciclo principal
if __name__ == "__main__":

    #Función para desnormalizar datos
    def desnormalizar (max, min, datos):
        denormalized = []
        for i in range(len(datos)):
            denormalized.append((datos[i] * (max - min)) + min)
        return denormalized

    # Leer ficheros de excel
    tablaEntrenamiento = pd.ExcelFile("concrete_training.xlsx")
    tablaValidacion = pd.ExcelFile("concrete_validation.xlsx")
    tablaTest = pd.ExcelFile("concrete_test.xlsx")
    
    datosEntrenamiento = tablaEntrenamiento.parse("Worksheet")
    datosValidacion = tablaValidacion.parse("Hoja1")
    datosTest = tablaTest.parse("Hoja1")

    # Convertir los datos en matrices
    datosEntrenamiento = np.array(datosEntrenamiento)
    datosValidacion = np.array(datosValidacion)
    datosTest = np.array(datosTest)

    # Datos de entrada
    xiEntrenamiento = datosEntrenamiento[:,0:8]
    xiValidacion = datosValidacion[:,0:8]
    xiTest = datosTest[:,0:8]

    # Valores deseados
    dEntrenamiento = datosEntrenamiento[:,9]
    dValidacion = datosValidacion[:,9]
    dTest = datosTest[:,9]

    # Numero de muestras/patrones
    patronEntrenamiento = len(dEntrenamiento)
    patronValidacion = len(dValidacion)
    patronTest = len(dTest)

    # Establecer el vector de pesos w
    wi = np.array([random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5),random.uniform(-0.5, 0.5)])
    # Factor de aprendizaje
    fac_ap = 0.0001
    # Ciclos
    ciclos = 0
    # Pesos ajustados
    w_ajustado = []
    
    # Inicializar la Red ADALINE
    red = ADALINE(dEntrenamiento,dValidacion,dTest,xiEntrenamiento,xiValidacion,xiTest,patronEntrenamiento,patronValidacion,patronTest,wi,fac_ap,ciclos,w_ajustado)
    w_ajustado, epocas, errorEntrenamiento, errorValidacion, ciclosModelo = red.Entrenamiento()
    
    # Grafica
    plt.ylabel('Error',Fontsize = 12)
    plt.xlabel('Ciclos',Fontsize = 12)
    plt.title("Evolucion del error")
    x = np.arange(epocas)
    y = [errorEntrenamiento, errorValidacion]
    labels = ["Error entrenamiento", "Error validacion"]
    for y_arr, label in zip(y, labels):
        plt.plot(x, y_arr, label=label)
    plt.legend(loc='upper right')
    plt.show()

    # Fichero modelo final
    f = open ('modelo.txt','w')
    f.write("Pesos ajustados: " + str(w_ajustado) + "\n" + "Numero de ciclos: " + str(ciclosModelo)) # Destacar que el último peso se corresponde con el umbral
    f.close()

    # Fichero error validacion
    f = open ('errorValidacion.txt','w')
    f.write("Evolución del error: " + str(errorValidacion))
    f.close()

    # Fichero error entrenamiento
    f = open ('errorEntrenamiento.txt','w')
    f.write("Evolución del error: " + str(errorEntrenamiento))
    f.close()

    # Fichero evolucion error
    f = open ('iterative_errors.txt','w')
    for i in range(epocas):
        f.write(str(errorEntrenamiento[i]) + "\t" + str(errorValidacion[i]) + "\n")
    f.close()

    #Salidas de la red
    red = ADALINE(dEntrenamiento,dValidacion,dTest,xiEntrenamiento,xiValidacion,xiTest,patronEntrenamiento,patronValidacion,patronTest,wi,fac_ap,ciclos,w_ajustado)
    salidas, errorTest = red.F_operacion()

    # Fichero error test
    f = open ('errorTest.txt','w')
    f.write("Error de la red: " + str(errorTest))
    f.close()

    # Fichero salidas de la red
    f = open ('salidas.txt','w')
    f.write("Salidas de la red: " + str(salidas))
    f.close()

    # Errores para el modelo optimo
    f = open ('erroresModeloOptimo.txt','w')
    f.write("Error de entrenamiento: " + str(errorEntrenamiento[ciclosModelo]) + "\n" + "Error validacion: " + str(errorValidacion[ciclosModelo]) + "\n" + "Error test: " + str(errorTest) + "\n" + "Numero de ciclos: " + str(ciclosModelo))
    f.close()

    # Desnormalizar los datos
    max=82.6
    min=2.33   
    denormalized = desnormalizar(max, min, salidas)
    denormalized2 = desnormalizar(max, min, dTest)

    # Fichero salidas de la red desnormalizados
    f = open ('salidas_desnormalizadas.txt','w')
    f.write("Salidas de la red: " + str(denormalized))
    f.close()

    plt.ylabel('Salida',Fontsize = 12)
    plt.xlabel('Patron',Fontsize = 12)
    plt.title("ADALINE, Regla Delta")
    x = np.arange(patronTest)
    y = [sorted(denormalized2), sorted(denormalized)]
    labels = ["Salidas deseadas", "Salidas obtenidas"]
    for y_arr, label in zip(y, labels):
        plt.plot(x, y_arr, label=label)
    plt.legend(loc='upper right')
    plt.show()

    # Fichero salidas de la red desnormalizados
    f = open ('salidas_esperada_obtenida.txt','w')
    for i in range(patronTest):
        f.write(str(denormalized2[i]) + "\t" + str(denormalized[i]) + "\n")
    f.close()