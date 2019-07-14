# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:27:48 2019

@author: Laura

Ejercicio 0 - Introducción a Python
"""


# Bibliotecas necesarias
from sklearn import datasets        # Importa la base de datos de iris

import numpy as np                  # Para trabajar con arrays
import matplotlib.pyplot as plt     # Para pintar gráficos

import math as m                    # Para poder tener la constante pi


# Parte 1
iris = datasets.load_iris()             # Se importa la base de datos

labels = (iris.feature_names).copy()    # Etiquetas
entrada = (iris.data).copy()            # Datos de entrada - X
clase = (iris.target).copy()            # Clases - Y

entr_column = entrada[:,2:]             # Se seleccionan las dos últimas 
                                        # columnas de los datos de entrada

# Pintar las gráficas
plt.title('Parte 1')                    # Título de la gráfica

# En rojo se representan los datos del grupo 0, en verde los del grupo 1 y en 
# azul los del grupo 2. 
plt.scatter(entr_column[:50,0:1],entr_column[:50,1:2], c='r', label='Grupo 0', s=10)
plt.scatter(entr_column[50:100,0:1],entr_column[50:100,1:2], c='g', label='Grupo 1', s=10)
plt.scatter(entr_column[100:,0:1],entr_column[100:,1:2], c='b', label='Grupo 2', s=10)

# Se representa en el eje x la primera de las dos columnas seleccionadas, y en 
# el eje y, la segunda de ellas. 
plt.xlabel(labels[2])
plt.ylabel(labels[3])      

plt.legend()                    # Para sacar la leyenda de la gráfica
plt.show()                      # Muestra la gráfica en pantalla

# Pausa de la primera parte
input('Pulsa \'enter\' para continuar con la segunda parte:')


# Parte 2
print ('\n\tPARTE 2\n')

# En primer lugar, se especifica la semilla. 
#
# Para elegir la semilla se comparó el resultado que ofrecían las distintas 
# posibilidades para saber cuál daba el valor más equitativo, es decir, 
# mantenía la misma proporción tanto en el training como en el test con 
# respecto al total de la muestra. 
# 
# Esta comparación se consiguió por  medio de un bucle for, de 1 a 256 (2^8),
# que mostraba la proporción de datos en función de la semilla. Las distintas 
# posibilidades que encontraron son {36,136,155,176,242,251}. Si se cambia la 
# semilla por cualquiera de ellos, se seguirá manteniendo el mismo porcentaje 
# (33.33%). 
np.random.seed(36)                          

# Se juntan los datos, las dos columnas aisladas en el apartado anterior, con 
# su respectiva clase, consiguiendo una matriz de (150,3)
datos = np.column_stack((entr_column, clase))
 
np.random.shuffle(datos)            # Se desordenan los datos                     

limite = int(len(datos)*0.8)        # Número de datos que tendrá el training
                                    # (num_total_datos * 0.8), ya que se quiere
                                    # 80% - training y 20% - test

training = datos[:limite]           # Se le asigna al training el 80% de datos:
                                    # índices: [0, limite - 1]                                
test = datos[limite:]               # Se le asigna al test el 20% de los datos;
                                    # índices: [limite, len(datos) - 1]


# Se comprueba que se mantiene la proporción de valores. 
                                
#Tamaño de los datos totales y del tamaño del training y del test. 
print ('Tamaño total de los datos: \t', len(entrada))
# >> Tamaño total de los datos: 150

print ('Tamaño del training: \t\t', len(training))
# >> Tamaño del training: 120

print ('Tamaño del test: \t\t', len(test))
# >> Tamaño del test: 30
  
# Se calcula el número de valores en cada clase
#
# Para ello, se compara el array de clases, en el caso del número de clases 
# totales, o la última columna de la matriz, en el caso del training y del test,
# con cada una de las posibles clases. Esto creará un vector booleano por cada
# clase, donde 0 será que la clase no es igual y 1 que si lo es. Si se suma el 
# contenido de cada vector booleano, dará el número total de elementos de esa
# clase en ese vector/matriz. 

# Número total de clases
clase0 = clase == 0
clase1 = clase == 1
clase2 = clase == 2

print ('\nValores totales:\t [0,%d] [1,%d] [2,%d]' % (clase0.sum(),clase1.sum(),clase2.sum()))
# >> Valores totales: [0, 50] [1, 50] [2, 50]

# Número de clases en el training
training0 = training[:,2:] == 0
training1 = training[:,2:] == 1
training2 = training[:,2:] == 2

print ('Valores del training:\t [0,%d] [1,%d] [2,%d]' % (training0.sum(),training1.sum(),training2.sum()))
# >> Valores del trining: [0, 40] [1, 40] [2, 40]

# Número de clases en el test
test0 = test[:,2:] == 0
test1 = test[:,2:] == 1
test2 = test[:,2:] == 2

print ('Valores del test:\t [0,%d] [1,%d] [2,%d]' % (test0.sum(),test1.sum(),test2.sum()))
# >> Valores del test: [0, 10] [1, 10] [2, 10]

# Pausa de la parte 2. 
input('Pulsa \'enter\' para continuar con la tercera parte:')


# Parte 3
valores = np.linspace(0,2*m.pi,100)     # Se obtienen 100 valores equiespaciados
                                        # entre 0 y 2pi.
                                        
seno = np.sin(valores)               # Se calcula el seno de los valores
cos = np.cos(valores)                # Se calcula el coseno de los valores

suma = seno + cos                    # Se calcula la suma del seno y del coseno

# Se visualizan los datos
plt.title('Parte 3')        # Título de la gráfica

plt.plot(valores, seno, 'k--', label='sin(x)')              # Seno

plt.plot(valores, cos, 'b--', label='cos(x)')               # Coseno

plt.plot(valores, suma, 'r--', label='sin(x) + cos(x)')     # Suma 

plt.legend()        # Leyenda
plt.show()          # Se muestra la gráfica en pantalla

# Pausa de la tercera parte. 
input('Pulsa \'enter\' para finalizar la ejecución:')
