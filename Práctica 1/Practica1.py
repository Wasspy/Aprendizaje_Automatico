# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:34:25 2019

@author: Laura Rabadán Ortega - 79088745W

"""

import random
import math as m
import numpy as np
import matplotlib.pyplot as plt

"""

Ejercicio 1: ejercicio sobre la búsqueda iterativa de óptimos
    
    GRADIENTE DESCENDENTE

"""

# Inicio de ejercicio
print ('\n\n\tEJERCICIO 1 - Ejercicio sobre la búsqueda iterativa de óptimos\n\n')


# DEFINICIÓN DE FUNCIONES 

# Función general para calcular el Gradiente Descente
# Argumentos:
#    -> funcion: función a minimizar.
#    -> gradiente: gradiente de la función a minimizar.
#    -> pesos: array de pesos. 
#    -> rate: tasa de aprendizaje que se quiere utilizar.
#    -> cota: valor límite que se quiere conseguir (valor < cota).
#    -> iter_max: número máximo de iteraciones.
# Return: número de iteraciones requeridas, pesos finales y pesos y valores de 
#         cada iteración.
def GradienteDescendente (funcion, gradiente, pesos, rate, cota=-999999, iter_max=100):
    
    # Número de iteraciones
    itera = 0  

    # Se calcula el valor de la función 'funcion' con los pesos iniciales 
    valor = funcion(pesos[0],pesos[1])
    
    # Vector que guardará el peso generado en cada iteración y el valor obtenido
    valores = np.array([itera, pesos[0], pesos[1], valor], np.float64)
    
    # Mientras el valor de la función sea mayor o igual que el valor requerido,
    # se itera, hasta conseguir el resultado esperado.                                     
    while (valor > cota and itera < iter_max):

        # Función de la diapositiva 10 de la sesión 2: Gradiente Descendente
        # Los nuevos valores de los pesos serán la diferencia entre los valores
        # anteriores y el producto de la tasa de aprendizaje (rate) y el 
        # gradiente de función
        pesos = pesos - rate * gradiente(pesos[0],pesos[1]) #CalcularGradiente(funcion, pesos)
    
        # Se calcula el nuevo valor de la función con los nuevos pesos. 
        valor = funcion(pesos[0],pesos[1])  
        
        # Se añade una nueva iteración
        itera += 1; 
        
        # Nueva entrada al vector de valores
        aux = np.array([itera, pesos[0], pesos[1], valor], np.float64)
        valores = np.vstack((valores, aux))
        
    # Fin del while
        
    # Se devuelve el número de iteraciones y el último valor de los pesos   
    return itera, pesos, valores

################################
# Función E(u,v)
def E (u,v):
    return ((u**2)*m.exp(v) - 2*(v**2)*m.exp(-u))**2

# Gradiente de E(u,v)
def GradE (u,v):
    return np.array([duE(u,v), dvE(u,v)], np.float64)

# Derivada de E(u,v) respecto a u
def duE (u,v):
    return 2*((u**2)*m.exp(v) - 2*(v**2)*m.exp(-u))*(2*u*m.exp(v) + 2*v**2*m.exp(-u))

# Derivada de E(u,v) respecto a v
def dvE (u,v):
    return 2*((u**2)*m.exp(v) - 2*(v**2)*m.exp(-u))*((u**2)*m.exp(v) - 4*v*m.exp(-u))

################################
# Función f(x,y)
def F (x,y):
    return (x**2) + (2*y**2) + 2*m.sin(2*m.pi*x)*m.sin(2*m.pi*y)

# Gradiente de f(x,y)
def GradF (x,y):
    return np.array([dxF(x,y), dyF(x,y)], np.float64)

# Derivada de f(x,y) respecto a x
def dxF (x,y):
    return 2*x + 4*m.pi*m.cos(2*m.pi*x)*m.sin(2*m.pi*y)

# Derivada de f(x,y) respecto a y
def dyF (x,y):
    return 4*y + 4*m.pi*m.sin(2*m.pi*x)*m.cos(2*m.pi*y)

################################
#
# APARTADO 2, E(u,v)

pesos = np.array([1,1], np.float64)     # Array de coordenadas iniciales

rate = 0.01                             # Constante de aprendizaje

cota = np.float64(1e-14)                # Valor mínimo que se quiere conseguir

# Se llama al Gradiente Descendente con los valores anteriores y la función 
# E(u,v) y su gradiente
iteraciones, pesos, valores = GradienteDescendente(E, GradE, pesos, rate, cota)

# Se muestran los resultados obtenidos
print ('Apartado 2 - Función E(u,v): ')
print ('\n\tValor conseguido: ', E(pesos[0],pesos[1]))
print ('\n\tNúmero de iteraciones requeridas: ', iteraciones)
print ('\n\tCoordenadas para el primer valor < 10^(-14): (%f,%f)' % (pesos[0], pesos[1]))

input('Pulsar \'enter\' para continuar con el Apartado 3')
 
################################
#
# APARTADO 3, f(x,y)   

pesos = np.array([0.1,0.1], np.float64)     # Array de coordenadas iniciales

rate = 0.01                                 # Constante de aprendizaje

iter_max = 50                               # Número máximo de iteraciones

# Se llama al Gradiente Descendente con los valores anteriores y la función 
# f(x, y) y su gradiente
iteraciones1, pesos1, valores1 = GradienteDescendente(F, GradF, pesos, rate, iter_max=iter_max)

rate = 0.1                        # Se cambia la constante de aprendizaje

# Se vuelve a llamar a la función igual que antes pero cambiando el learning rate
iteraciones2, pesos2, valores2 = GradienteDescendente (F, GradF, pesos, rate, iter_max=iter_max)

# Se pinta una gráfica comparando los valores obtenidos en función del númeor 
# de iteraciones
plt.title('Apartado 3.a - Función f(x,y)')
plt.plot(valores1[:,0], valores1[:,3], 'g.', label= 'rate = 0.01')
plt.plot(valores2[:,0], valores2[:,3], 'b.', label= 'rate = 0.1')

plt.xlabel('Número de iteración')
plt.ylabel('Valor de f(x,y)') 
plt.legend()   
  
plt.show()

input('Pulsar \'enter\' para pasar a la siguiente parte del apartado')

# Apartado 3.b 

 # Array de coordenadas iniciales para los distintos casos
pesos = np.array([(0.1,0.1), (1,1), (-0.5,-0.5), (-1,-1)], np.float64)

itera_max = 50                  # Número máximo de iteraciones

rate = 0.01                     # Constante de aprendizaje 

id = 221                        # Identificador de gráfica

cont = 0                        # Contador

c = ['g', 'b', 'r', 'purple']   # Lista de colores

# Identificador de apartado
print ('Valores obtenidos en el apartado 3.b (tabla en la memoria): ')

# Se llama a la función con cada uno de las coordenadas y se pintan sus valores
for p in pesos:
    
    itera, peso, valores = GradienteDescendente (F, GradF, p, rate, iter_max=itera_max)
    
    ax = plt.subplot(str(id))
    ax.set_title('G%d - (%f, %f)' % (cont + 1, p[0], p[1]))
    ax.plot(valores[:, 0], valores[:, 3], c=c[cont], label=p)
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Valor')
    
    id += 1
    cont += 1
    
    # Se pintan los pesos finales de cada punto de incio
    print('\t -> Gráfica %d - pesos: (%f, %f)' % (cont, peso[0], peso[1]))

plt.tight_layout()
plt.show()

input('Pulsar \'enter\' para continuar con el ejercicio 2')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Ejercicio 2: ejercicio sobre la Regresión Lineal

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Inicio de ejercicio
print ('\n\n\tEJERCICIO 2 - Ejercicio sobre la Regresión Lineal\n\n')

# DEFINICIÓN DE FUNCIONES

# Lectura de datos
# RETURN: conjunto de datos del train y conjunto de datos del test
def LeerDatos (): 
    
    # Carga el test y el train desde los ficheros correspondientes
    test = np.load('datos\\features_test.npy')
    train = np.load('datos\\features_train.npy')
    
    # Declaración de etiquetas de clase
    label1 = -1
    label5 = 1
    
    # Se modifican las etiquetas de las clases en función a los valores declarados
    test = Datos(test, label1, label5)
    train = Datos(train, label1, label5)
    
    return train, test

# Modificación de los datos
# RETURN: los mismos datos con las etiquetas cambiadas 
def Datos (datos, label1, label5):
    
    # Se añade una columna de '1' adicional al final de la matriz para poder 
    # calcular el w constante sin problema. 
    aux = np.ones((len(datos),), np.float64)
    datos = np.column_stack((datos,aux))
    
    # Se recorre la matriz cambiando las etiquetas por los valores 
    # correspondientes:
    #       -> 1 por -1
    #       -> 5 por 1
    for i in datos:
        if (i[0] == 1):
            i[0] = label1
        else:
            i[0] = label5
     
    return datos

################################
       
# Función del Gradiente Desdencente estocástico
# RETURN: pesos definitivos, soluciones obtenidas en cada iteración, error 
#         conseguido con cada iteración y número de iteraciones
def SGD (datos, pesos, rate= 0.01, max_itera= 50, cota=1e-2):
    
    # Se fija una semilla para la generación de números aleatorios. 
    np.random.seed(4)
    
    # Se determina el tamaño del batch 
    tam_bach = 64

    itera = 0                           # Contador de iteraciones
    
    iteracion = np.array(itera)         # Array de iteraciones
    
    error = Error(pesos, datos)           # Error inicial con los pesos iniciales
    
    errores = np.array(error)           # Array de errores
    
    # Se añade a la solución los pesos iniciales y su error asociados
    soluciones = np.array([itera, pesos[0], pesos[1], error])
    
    # Se itera hasta conseguir un error menor a 1e-1 o hasta gastar las iteraciones 
    while (error > cota and itera < max_itera):
        
        # En cada iteración se mezclan los datos y se escogen los primeros 
        # 'tam_bach' valores. 
        np.random.shuffle(datos)
        batch = datos[:tam_bach,:]
        
        # Se recorren todos los pesos 
        for i in range(0, len(pesos), 1):
        
            # Se aplica la fórmula del SGD para cada peso de cada característica
            pesos[i] = pesos[i] - rate * GradEin(pesos, batch, i)
        
        # Se aumenta el número de iteraciones
        itera += 1
        iteracion = np.vstack((iteracion, np.array(itera)))
        
        # Se calcula el nuevo error y se añade al array de errores
        error = Error(pesos, datos)
        errores = np.vstack((errores, np.array(error)))
        
        # Se añade la solución de esta iteración al array de soluciones
        soluciones = np.vstack((soluciones, np.array((itera, pesos[0], pesos[1], error))))
            
    return pesos, soluciones, errores, iteracion

################################

# Función para la pseudoinversa
# RETURN: pesos
def Pseudoinversa(datos):
    
    # Se dividen los datos en características (x) y etiquetas (y)
    x = datos[:,1:]
    y = datos[:,0]
    
    # Se calcula la inversa del producto matricial de x transpuesta y x 
    xi = np.linalg.pinv(np.transpose(x) @ x)
    
    # Los pesos se calculan con el producto de la inversa anterior, x transpuesta
    # y el vector y. 
    w = xi @ np.transpose(x) @ y
    
    return w

################################

# Error a la hora de etiquetal los datos
def Error (w, datos):
    
    # Se dividen los datos en características (x) y etiquetas (y)
    x = datos[:,1:]
    y = datos[:,0]
    
    # Variable que recogera la suma del error total
    suma = 0
    
    # Se recorren todos los datos del conjunto de entrada
    for i in range (0, len(y), 1):
        suma += (np.transpose(w)@x[i] - y[i])**2
    
    # Se calcula el error medio en la muestra
    suma = suma/len(datos)
    
    return suma

# Gradiente del error dentro de la muestra (Ein)    
def GradEin (w, datos, i):
    
    # Se dividen los datos en características (x) y etiquetas (y)
    x = datos[:,1:]
    y = datos[:,0]
    
    # Variable que recogera la suma del error total
    suma = 0
    
    # Se recorren todos los datos del conjunto de entrada y se calcula su error
    for xj in x[i]:
        suma += xj * (np.transpose(w)@(x[i]) - y[i])
    
    # Se calcula el error medio en la muestra
    suma = (2*suma)/len(datos)
 
    return suma

################################

# Función para determinar las etiquetas de unos datos de entrada en función a 
# los pesos calculados 
# RETURN: array de elementos de un grupo con su correspondiente etiqueta, array
#         de elementos del otro grupo con su correspondiente etiqueta y error 
#         generado. 
def Prediccion(datos, pesos):
    
    # Se calcula el error medio en los datos con esos pesos
    error = Error(pesos, datos)
    
    grupo1 = []         # Lista de elementos con etiquetas de grupo (1)
    grupo2 = []         # Lista de elementos con etiquetas de grupo (2)
    
    # Se itera sobre todos los datos y, dependiendo del valor que se obtenga al
    # mutiplicar las características por los pesos, se le asigna un valor u otro
    #   -> Si el resultado es mayor que 0, se le asigna la clase 1
    #   -> Si el resultado es menor que 0, se le asigna la clase -1
    for i in datos:
        x = (pesos*i[1:]).sum()
        
        if (x > 0):
            grupo1.append((i[1], i[2], 1))
        else:
            grupo2.append((i[1], i[2], -1))
            
    # Se crean los array correspondientes a cada clase
    grupo1 = np.array(grupo1)
    grupo2 = np.array(grupo2)

    return grupo1, grupo2, error

################################

# Función del apartado 1. 
# Desde esta función se llama al SGD y a la pseudo inversa, se muestran los 
# valores que consiguen para los pesos y el error que obtienen, tanto dentro 
# de la muestra como fuera. Muestra un grafo con la asigación de clases dentro
# para los datos de entrenamiento, un grafo para cada función.  
def RegressLin (test, train):
    
    rate = 0.1                 # Constante de aprendizaje que se va a usar

    pesos = np.zeros((3,))      # Inicialización del vector de pesos
    pesos.astype(np.float64)    # Se le cambia el tipo de dato del vector
    
    # Primer algoritmo: SDG 
    print ('\nPrimer algoritmo: SGD')
    
    # Se llama a la función con los datos de la muestra
    pesos, soluciones, errores, iteraciones = SGD(train, pesos, rate, max_itera=100)
     
    # Se calcula el error dentro de la muestra al aplicar los pesos calculados
    grupo1, grupo2, sgdein  = Prediccion(train, pesos)
 
    # Se pinta la gráfica con las clases que se le han asignado. 
    plt.title('SGD - Predicción del train')
    plt.scatter(grupo1[:,0], grupo1[:,1], c='r', label='1')  
    plt.scatter(grupo2[:,0], grupo2[:,1], c='b', label='-1')
        
    plt.xlabel('Intensidad promedio')
    plt.ylabel('Simetría')      
    
    plt.legend()                           
    plt.show()
    
    # Se calcula el error fuera de la muestra
    a, b, sgdeout  = Prediccion(test, pesos)
    
    # Se pintan los resultados obtenidos
    print ('\t -> Learning Rate: ', rate)
    print ('\t -> Pesos finales (x1, x2, 1): (%f, %f, %f)' % (pesos[0], pesos[1], pesos[2]))
    print ('\t -> Error dentro de la muestra: ', sgdein)
    print ('\t -> Error fuera de la muestra:  ', sgdeout)

    input ('\nPulsar \'enter\' para pasar al siguiente algoritmo')
    
    # Segundo algoritmo: Pseudoinversa 
    print ('\nSegundo algoritmo: Pseudoinversa \n\t')
    
    # Se vuelve a inicializar el vector de pesos a 0
    pesos = np.zeros((2,))      
    
    # Se llama a la función con los datos de entrenamiento
    pesos = Pseudoinversa(train)
    
    # Se calculan las etiquetas de los datos de entrenamiento a partir de los 
    # pesos calculados
    grupo1, grupo2, errorin = Prediccion(train, pesos)

    # Se pinta la gráfica con las clases asignadas
    plt.title('Pseudoinverse - Predicción del test')
    plt.scatter(grupo1[:,0], grupo1[:,1], c='r', label='1')  
    plt.scatter(grupo2[:,0], grupo2[:,1], c='b', label='-1')
        
    plt.xlabel('Intensidad promedio')
    plt.ylabel('Simetría')      
    
    plt.legend()                   
    plt.show()
    
    # Se calcula el error con los datos fuera de la muestra
    a, b, errorout  = Prediccion(test, pesos)
    
    # Se pintan los resultados
    print ('\t -> Pesos finales (x1, x2): (%f, %f, %f)' % (pesos[0], pesos[1], pesos[2]))
    print ('\t -> Error dentro de la muestra: ', errorin)
    print ('\t -> Error fuera de la muestra:  ', errorout)  

################################
################################

# Apartado 1
# Se leen los datos y se dividen en test y train
train, test = LeerDatos()

# Se llama a la función del primer apartado
RegressLin(test, train)

input('Pulsar \'enter\' para continuar con el segundo apartado')

##############################################################################
# EXPERIMENTO (Apartado 2)
##############################################################################

# Simula datos en un cuadrado [-size,size]x[-size,size]
# RETURN: valores generados con su etiqueta correspondiente y ruido en un 10% 
#         de datos de la muestra
def SimulaUnif(N, d, size):
    
    # Se crean los N datos de d dimensiones con valores entre N y N 
    valores = np.random.uniform(-size,size,(N,d))
    
    # Se calcula su valor en función a la ecuación f(x1,x2)
    y = np.sign(((valores[0,0]-0.2)**2) + (valores[0,1]**2) - 0.6)
    
    # Se construye la matrix de valores con el formato adecuado para el SGD
    soluciones = np.array((y, valores[0,0], valores[0,1], 1))
    
    # Se calcula el resto de valores de la matriz
    for x in range (1, len(valores), 1):
        y = np.sign(((valores[x,0]-0.2)**2) + (valores[x,1]**2) - 0.6)
        
        soluciones = np.vstack((soluciones, np.array((y, valores[x,0], valores[x,1], 1))))
    
    # Se mezclan los valores de la matriz
    np.random.shuffle(soluciones)
    
    # Se guarda el número de elementos de la matriz que corresponden al 10%
    N =  len(soluciones)*0.1
    
    # Al 10% de los datos se le cambia la etiqueta
    for i in range (0, int(N), 1):
        
        # Se sustituye la etiqueta por su nuevo valor
        soluciones[i,0] = soluciones[i,0] * -1

    return soluciones

###############################
    
# Función experimento: cubre los subapartados a, b, c
# RETURN: matriz de datos con etiquetas, error en los datos y pesos finales. 
def Experimento (N, d, size, i):
    
    # Se genera la matriz de datos
    valores = SimulaUnif(N, d, size)

    # Se inicializa el vector de pesos
    pesos = np.zeros((d + 1,), np.float64)
    
    # Se establece la constante de aprendizaje
    rate = 0.01
    
    # Se llama a la función del SGD son una cota máxima de 1 y un máximo de 10 iteraciones
    pesos, soluciones, errores, iteraciones = SGD(valores, pesos, rate=rate, cota=0.9, max_itera=i)
    
    # Se comprueba el error de la predicción en los propios datos
    a, b, error = Prediccion(valores, pesos)

    return valores, error, pesos

################################
################################
    
# Experimento base 
    
# Se fijan la semillas semillas del experimento
random.seed(2)
np.random.seed(5)

# Se lleva a cabo el experimento con 1000 datos de 2D entre [-1, 1] x [-1, 1]
valores, error, pesos = Experimento(1000, 2, 1, 100)

# Se pintan los valores genreados
plt.title('Valores de entrenamiento - X = [-1,1] x [-1,1]')
plt.scatter(valores[:,1], valores[:,2], c='purple', s=1)
plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')
plt.show()

input('Pulsar \'enter\' para continuar con el siguiente apartado')

# Se pintan los puntos generados identificados por la etiqueta correspondiente
grupo1 = []
grupo2 = []

# Se divide en grupos según su etiqueta
for i in valores:
    
    if (i[0] == -1):
        grupo1.append(i)

    else:
        grupo2.append(i)

grupo1 = np.array(grupo1)
grupo2 = np.array(grupo2)

# Se pinta la gráfica
plt.title('Etiquetado con ruido - f(x1, x2) - X = [-1,1] x [-1,1]')
plt.scatter(grupo1[:,1], grupo1[:,2], label='Grupo: -1', c='chartreuse', s=15)
plt.scatter(grupo2[:,1], grupo2[:,2], label='Grupo:  1', c='dodgerblue', s=15)
plt.axis([-1,1.75,-1,1])
plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')
plt.legend()
plt.show()
        
input('Pulsar \'enter\' para continuar con el siguiente apartado')

# Se muestran los pesos finales y el error de ajuste de la muestra 
print ('\nPesos: ', pesos)
print ('Error de ajuste: ', error)

input('Pulsar \'enter\' para continuar con el siguiente apartado')

# Experimento con 1000 iteraciones:
# NOTA: Tada unos 7 minutos en ejecutar este apartado

# Número de iteraciones 
N = 1000

# Errores medios dentro y fuera de la muestra
errorin = 0
errorout = 0

# Se fija una semilla para este apartado
np.random.seed(20)

# Bucle de experimentos 
for i in range (0, N, 1):
    
    # Se realiza el experimento
    valores, error, pesos = Experimento(1000, 2, 1, 10)
    
    # Se acumula el error obtenido 
    errorin += error
    
    # Se generan datos de entrenamiento
    valores = SimulaUnif(1000, 2, 1)
      
    # Se comprueba el error generado
    a, b, error = Prediccion(valores, pesos)
    
    # Se acumula el error
    errorout += error

# Se calcula el error medio en la muestra y fuera de ella 
errorin = errorin/N
errorout = errorout/N

# Se muestran los resultados
print ('Error medio en la muestra: \t', errorin)
print ('Error medio fuera de la muestra: \t', errorout)

input ('Pulsar \'enter\' para continuar con el ejercicio BONUS')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Ejercicio 2.1: BONUS

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Inicio de ejercicio
print ('\n\n\tBONUS1 - Método de Newton\n\n')

################################

# DEFINICIÓN DE FUNCIONES 

# Método de Newton 
# RETURN: número de iteraciones, pesos finales y valores
def Newton (funcion, gradiente, seggrad, pesos, rate=1, cota=-999999, iter_max=50):
    
    # Contador de iteraciones
    itera = 0
    
    # Se calcula el valor de la función 'funcion' con los pesos iniciales 
    valor = funcion(pesos[0],pesos[1])
    
    # Vector que guardará el peso generado en cada iteración y el valor obtenido
    valores = np.array([itera, pesos[0], pesos[1], valor], np.float64)
    
    # Mientras el valor de la función sea mayor o igual que el valor requerido,
    # se itera, hasta conseguir el resultado esperado.                                     
    while (valor > cota and itera < iter_max):

        # Se calcula la matriz Hessiana de la función 'funcion'    
        H = np.array([(seggrad[0](pesos[0],pesos[1]), seggrad[1](pesos[0],pesos[1])), 
              (seggrad[2](pesos[0],pesos[1]), seggrad[3](pesos[0],pesos[1]))])
        
        # Se calcula el valor de los pesos con el método de Newton
        pesos = pesos - rate*np.linalg.inv(H)@gradiente(pesos[0], pesos[1])
        
        # Se calcula el nuevo valor de la función con los nuevos pesos. 
        valor = funcion(pesos[0],pesos[1])  
        
        # Se añade una nueva iteración
        itera += 1; 
        
        # Nueva entrada al vector de valores
        aux = np.array([itera, pesos[0], pesos[1], valor], np.float64)
        valores = np.vstack((valores, aux))
        
    # Fin del while
        
    # Se devuelve el número de iteraciones y el último valor de los pesos   
    return itera, pesos, valores

################################
    
# Segunda derivada de f(x,y) respecto a x (primera) y x (segunda)
def dSxxF (x,y):
    return 2 - 8*(m.pi**2)*m.sin(2*m.pi*y)*m.sin(2*m.pi*x)

# Segunda derivada de f(x,y) respecto a x (primera) e y (segunda)
def dSxyF (x,y):
    return 8*(m.pi**2)*m.cos(2*m.pi*x)*m.cos(2*m.pi*y)

# Segunda derivada de f(x,y) respecto a y (primera) e y (segunda)
def dSyyF (x,y):
    return 4 - 8*(m.pi**2)*m.sin(2*m.pi*x)*m.sin(2*m.pi*y)

# Segunda derivada de f(x,y) respecto a y (primera) y x (segunda)
def dSyxF (x,y):
    return 8*(m.pi**2)*m.cos(2*m.pi*y)*m.cos(2*m.pi*x)

################################
################################
    
 # Array de coordenadas iniciales para los distintos casos
pesos = np.array([(0.1,0.1), (1,1), (-0.5,-0.5), (-1,-1)], np.float64)

# Lista de segundas derivadas de la función
seggrad = [dSxxF,dSxyF, dSyyF, dSyxF]

itera_max = 50                   # Número máximo de iteraciones

id = 221                        # Identificador de gráfica

cont = 0                        # Contador

c = ['g', 'b', 'r', 'purple']   # Lista de colores


# Presentación de apartado
print ('Experimento del apartado 3.b del ejercicio 1: ')

# Primer caso: sin aplicar Learning Rate
print ('\t -> Número de iteraciones máximo: ', itera_max, '\n')

plt.title('Learning Rate = 1')

# Se llama a la función con cada uno de las coordenadas y se pintan sus valores
for p in pesos:
    
    itera, peso, valores = Newton(F, GradF, seggrad, p, iter_max=itera_max)
    
    ax = plt.subplot(str(id))
    ax.set_title('G%d - (%f, %f)' % (cont + 1, p[0], p[1]))
    ax.plot(valores[:, 0], valores[:, 3], c=c[cont], label=p)
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Valor')
    
    id += 1
    cont += 1
    
    # Se pintan los pesos finales de cada punto de incio
    print('\t -> Gráfica %d - pesos: (%f, %f)' % (cont, peso[0], peso[1]))

plt.tight_layout()
plt.show()

input ('Pulsar \'enter\' para ejecutar el segundo caso')

# Segundo caso: aplicando un Learning Rate de 0.01
id = 221                        # Identificador de gráfica

cont = 0                        # Contador

# Presentación de apartado
print ('Experimento del apartado 3.b del ejercicio 1: ')
print ('\t -> Número de iteraciones máximo: ', itera_max, '\n')

plt.title('Learning Rate = 0.01')

# Se llama a la función con cada uno de las coordenadas y se pintan sus valores
for p in pesos:
    
    itera, peso, valores = Newton(F, GradF, seggrad, p, rate=0.01, iter_max=itera_max)
    
    ax = plt.subplot(str(id))
    ax.set_title('G%d - (%f, %f)' % (cont + 1, p[0], p[1]))
    ax.plot(valores[:, 0], valores[:, 3], c=c[cont], label=p)
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Valor')
    
    id += 1
    cont += 1
    
    # Se pintan los pesos finales de cada punto de incio
    print('\t -> Gráfica %d - pesos: (%f, %f)' % (cont, peso[0], peso[1]))

plt.tight_layout()
plt.show()

input ('Pulsar \'enter\' para finalizar la ejecución')
