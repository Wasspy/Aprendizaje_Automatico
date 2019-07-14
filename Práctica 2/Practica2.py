# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Laura Rabadán Ortega
"""
import numpy as np
import matplotlib.pyplot as plt
import math

# Fijamos la semilla
np.random.seed(1)   # 3

##############################################################################
# Funciones
##############################################################################

# Función que calcula una lista de N vectores de dimensión dim. Cada vector
# contiene dim números aleatorios uniformes en el intervalo rango.
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

# Función que calcula una lista de longitud N de vectores de dimensión dim,
# donde cada posición del vector contiene un número aleatorio extraido de una
# distribucción Gaussiana de media 0 y varianza dada, para cada dimension, por 
# la posición del vector sigma.
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

# Funcion que simula de forma aleatoria los parámetros, v = (a, b) de una recta,
# y = ax + b, que corta al cuadrado [−50, 50] × [−50, 50].
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

###############################################################################
###############################################################################
###############################################################################

"""

 EJERCICIO 1 - Ejercicio sobre la complejidad de H y el ruido 

"""

print ('\n\t--- EJERCICIO 1 ---')

###############################################################################
###############################################################################

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('\n -> Apartado 1.1')

# Datos conseguidos con simula_unif
unif = simula_unif(50, 2, [-50,50])

# Se pintan los datos
plt.title('simula_unif')
plt.scatter(unif[:,0], unif[:,1], c='b', s=10)
plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')

plt.show()

input("\n--- Pulsar enter para mostrar 'simula_gaus' ---\n")

# Datos conseguidos con simula_gaus
gaus = simula_gaus(50, 2, np.array([5,7]))

# Se pintan los datos
plt.title('simula_gaus')
plt.scatter(gaus[:,0], gaus[:,1], c='g', s=10)
plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')

plt.show()

input("\n--- Pulsar enter para continuar con el apartado 1.2 ---\n")

###############################################################################
###############################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('\nApartado 1.2')

# Establece el signo de la etiqueta, [-1,1]
def Signo(x):
	if x >= 0:
		return 1
	return -1

# Función que determina el símbolo del dato
def F(x, y, a, b):
	return Signo(y - a*x - b)

# Función que mete ruido en las etiquetas de los datos. 
def Ruido (etiquetas):
    
    datos = np.copy(etiquetas)
    pos = int(len(datos) * 0.1)
    neg = pos
    
    # Se cambia un 10% de los datos de cada etiqueta
    for i in range(0, len(datos), 1):
        
        if (datos[i] == -1 and neg > 0):
            datos[i] = 1
            neg -= 1
        
        elif (datos[i] == 1 and pos > 0):
            datos[i] = -1
            pos -= 1

    return datos

###############################################################################

# Intervalo de los datos
intervalo = np.array([-50,50])

# Datos conseguidos
datos = simula_unif(100, 2, intervalo)

# Se simula la recta del intervalo
a, b = simula_recta(intervalo)

print (" -> Recta generada: y = %fx + %f" % (a, b))

# Se calculan las etiquetas
labels = []
for d in datos:
    
    labels.append(F(d[0], d[1], a, b))

labels = np.array(labels)

# Se pintan los datos en función de la etiqueta asignada
plt.title('Muestra SIN Ruido')
plt.plot([-70, 50], [-70*a + b, 50*a + b ], c='k', label="Recta simulada")

plt.scatter(np.squeeze(datos[np.where(labels == -1),0]), np.squeeze(datos[np.where(labels == -1),1]), c='r', s=10, label="-1")
plt.scatter(np.squeeze(datos[np.where(labels == 1),0]), np.squeeze(datos[np.where(labels == 1),1]), c='b', s=10, label=" 1")

plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')

plt.legend()
plt.show()

input("\n--- Pulsar enter para mostrar el 1.2.b ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su 
# etiqueta, junto con la recta usada para ello 
# Array con 10% de indices aleatorios para introducir ruido

# Se le añade ruido a las etiquetas
ruido = Ruido(labels)

# Muestran los datos con las etiquetas con ruido
plt.title('Muestra CON Ruido')
plt.plot([-70, 50], [-70*a + b, 50*a + b ], c='k', label="Recta simulada")

plt.scatter(np.squeeze(datos[np.where(ruido == -1),0]), np.squeeze(datos[np.where(ruido == -1),1]), c='r', s=10, label="-1")
plt.scatter(np.squeeze(datos[np.where(ruido == 1),0]), np.squeeze(datos[np.where(ruido == 1),1]), c='b', s=10, label=" 1")

plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')

plt.legend()
plt.show()

input("\n--- Pulsar enter para continuar con el apartado 1.3 ---\n")

###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la 
#               frontera de clasificación de los puntos de la muestra en lugar 
#               de una recta

print('\nApartado 1.3')

# fz = función a dibujar
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
 
# Distintas funciones presentadas en el ejercicio
def F1 (datos):
    return np.array((datos[:,0] - 10)**2 + (datos[:,1] - 20)**2 - 400)

def F2 (datos):
    return np.array(0.5*(datos[:,0] + 10)**2 + (datos[:,1] - 20)**2 - 400)

def F3 (datos):
    return np.array(0.5*(datos[:,0] - 10)**2 - (datos[:,1] + 20)**2 - 400)

def F4 (datos):
    return np.array(datos[:,1] - 20*datos[:,0]**2 + 5*datos[:,0] +3)

###############################################################################

# Se pintan las funciones del ejercicio con los datos con etiquetas con ruido
plot_datos_cuad(datos, ruido, F1, 'f(x,y) = (x - 10)**2 + (y - 20)**2 - 400')

input("\n--- Pulsar enter para mostrar la siguiente gráfica ---\n")

plot_datos_cuad(datos, ruido, F2, 'f(x,y) = 0.5(x + 10)**2 + (y - 20)**2 - 400')

input("\n--- Pulsar enter para mostrar la siguiente gráfica ---\n")

plot_datos_cuad(datos, ruido, F3, 'f(x,y) = 0.5(x - 10)**2 - (y + 20)**2 - 400')

input("\n--- Pulsar enter para mostrar la siguiente gráfica ---\n")

plot_datos_cuad(datos, ruido, F4, 'f(x,y) = y - 20x**2 - 5x + 3')

input("\n--- Pulsar enter para continuar con el ejercicio 2 ---\n")

###############################################################################
###############################################################################
###############################################################################

"""

 EJERCICIO 2 - Modelos Lineales

"""

print ('\n\t--- EJERCICIO 2 ---')

###############################################################################
###############################################################################

# Apartado 2.1: ALGORITMO PERCEPTRON

print('\nApartado 2.1')

# Algoritmo PLA
# ENTRADA:
#   -> muestra: datos a clasificar
#   -> label: etiquetas de los datos
#   -> max_iter: número máximo de iteraciones
#   -> vini: valor inicial de los datos
def ajusta_PLA(muestra, label, max_iter, vini):
    
    # Se le añade una columna de unos a los datos 
    datos = np.hstack((np.copy(muestra), np.ones((len(muestra),1))))
    
    # Pesos inicializados con vini (wnew) y con vini + 1 (wold)
    wold = np.full((len(datos[0]),), vini + 1)
    wnew = np.full((len(datos[0]),), vini)
    
    # Número de iteraciones
    itera = 0
    
    # Se itera hasta no encontrar mejora o hasta gastar un número determindado 
    # de iteraciones, max_iter
    while (np.array_equal(wold, wnew) == False and itera < max_iter):

        # Se guarda los pesos conseguidos
        wold = wnew
        
        # Se itera sobre todos los datos
        for i in range (0, len(datos)):
            
            # Si la etiqueta asignada con los pesos es distinta a la etiqueta
            # real, se modifica el vector de pesos en dirección a la etiqueta 
            # de ese dato
            if (Signo(np.dot(np.transpose(wnew), datos[i])) != label[i]):
                
                wnew = wnew + label[i]*datos[i]
        
        # Se aumenta el número de iteraciones
        itera = itera + 1
       
    # Se devuelven los pesos conseguidos y el número de iteraciones necesitado
    return wnew, itera

###############################################################################

# a) Usando los datos 'muestra'
print (" \nSubapartado a - Muestra SIN ruido" )

# Inicialización del algoritmo con 0
print (" - Vector inicial de 0")

# Se determinan los pesos para la clasificación de los datos 
w, itera = ajusta_PLA(datos, labels, 1000, 0.0)

print ("\n -> Valor inicial 0 - Número medio de iteraciones: ", itera)

# Se calculan las constantes necesarias para generar la recta que divide los datos
a = (-(w[2] / w[1]) / (w[2] / w[0])) 
b =  (-w[2] / w[1])

# Se pinta la predicción empezando con valor inicial 0
plt.title ("Valor inicial: 0")
plt.plot([-50, 50], [(-50) * a + b, 50 * a + b], c='b', label="prediccion")

plt.scatter(np.squeeze(datos[np.where(labels == -1),0]), np.squeeze(datos[np.where(labels == -1),1]), c='r', s=10, label="-1")
plt.scatter(np.squeeze(datos[np.where(labels == 1),0]), np.squeeze(datos[np.where(labels == 1),1]), c='b', s=10, label=" 1")

plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')

plt.legend()
plt.show()

print ("\t - Pesos: (%f, %f, %f) \n\t - y = ax + b => a = %f ; b = %f" % (w[0], w[1], w[2], a, b))

input("\n--- Pulsar enter para continuar con el siguiente ejemplo ---\n")

print (" - Vector inicial aleatorio (valores entre [0,1], 10 veces)")

# Random initializations
iterations = []
aleatorios = np.random.random((10,))

# Se relizan 10 ejecuciones del algoritmo PLA empezando en puntos aleatorios 
# entre [0,1]
for i in range(0,10):
    
    # Se ejecuta el algoritmo con el valor aleatorio
    numero = aleatorios[i]
    w, itera = ajusta_PLA(datos, labels, 1000, numero)
    
    print ("\n\t-> Valor inicial %f - Número de iteraciones: %d" % (numero, itera))

    # Se pinta la gráfica con el ajuste conseguido
    a = (-(w[2] / w[1]) / (w[2] / w[0])) 
    b =  (-w[2] / w[1])
    
    plt.title ("Valor inicial: %f" % (numero))
    plt.plot([-50, 50], [(-50) * a + b, 50 * a + b], c='b', label="prediccion")
    
    plt.scatter(np.squeeze(datos[np.where(labels == -1),0]), np.squeeze(datos[np.where(labels == -1),1]), c='r', s=10, label="-1")
    plt.scatter(np.squeeze(datos[np.where(labels == 1),0]), np.squeeze(datos[np.where(labels == 1),1]), c='b', s=10, label=" 1")

    plt.xlabel('Coordenada x1')
    plt.ylabel('Coordenada x2')
    
    plt.legend()
    plt.show()
    
    print ("\t\t - Pesos: (%f, %f, %f) \n\t\t - y = ax + b => a = %f ; b = %f" % (w[0], w[1], w[2], a, b))

    # Se añade el número de iteraciones necesitadas para conseguir el ajuste
    iterations.append(itera)
    
    input("\n\t--- Pulsar enter para continuar con el siguiente valor ---\n")
    
print('\n -> Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar enter para repetir el estudio con los datos con ruido ---\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# b) Usando los datos 'm_ruido'
print (" \nSubapartado b - Muestra CON ruido" )

# Inicialización del algoritmo con 0
print (" - Vector inicial de 0")

# Se determinan los pesos para la clasificación de los datos 
w, itera = ajusta_PLA(datos, ruido, 1000, 0.0)

print ("\n -> Valor inicial 0 - Número medio de iteraciones: ", itera)

# Se calculan las constantes necesarias para generar la recta que divide los datos
a = (-(w[2] / w[1]) / (w[2] / w[0])) 
b =  (-w[2] / w[1])

# Se pinta la predicción empezando con valor inicial 0
plt.title ("Valor inicial: 0")
plt.plot([-50, 50], [(-50) * a + b, 50 * a + b], c='b', label="prediccion")

plt.scatter(np.squeeze(datos[np.where(ruido == -1),0]), np.squeeze(datos[np.where(ruido == -1),1]), c='r', s=10, label="-1")
plt.scatter(np.squeeze(datos[np.where(ruido == 1),0]), np.squeeze(datos[np.where(ruido == 1),1]), c='b', s=10, label=" 1")

plt.xlabel('Coordenada x1')
plt.ylabel('Coordenada x2')

plt.legend()
plt.show()

print ("\t - Pesos: (%f, %f, %f) \n\t - y = ax + b => a = %f ; b = %f" % (w[0], w[1], w[2], a, b))

input("\n--- Pulsar enter para continuar con el siguiente ejemplo ---\n")

print (" - Vector inicial aleatorio (valores entre [0,1], 10 veces)")

# Random initializations
iterations = []

# Se relizan 10 ejecuciones del algoritmo PLA empezando en puntos aleatorios 
# entre [0,1]
for i in range(0,10):
    
    # Se ejecuta el algoritmo con el valor aleatorio
    numero = aleatorios[i]
    w, itera = ajusta_PLA(datos, ruido, 1000, numero)
    
    print ("\n\t-> Valor inicial %f - Número de iteraciones: %d" % (numero, itera))

    # Se pinta la gráfica con el ajuste conseguido
    a = (-(w[2] / w[1]) / (w[2] / w[0])) 
    b =  (-w[2] / w[1])
    
    plt.title ("Valor inicial: %f" % (numero))
    plt.plot([-50, 50], [(-50) * a + b, 50 * a + b], c='b', label="prediccion")
 
    plt.scatter(np.squeeze(datos[np.where(ruido == -1),0]), np.squeeze(datos[np.where(ruido == -1),1]), c='r', s=10, label="-1")
    plt.scatter(np.squeeze(datos[np.where(ruido == 1),0]), np.squeeze(datos[np.where(ruido == 1),1]), c='b', s=10, label=" 1")

    plt.xlabel('Coordenada x1')
    plt.ylabel('Coordenada x2')
    
    plt.legend()
    plt.show()
    
    print ("\t\t - Pesos: (%f, %f, %f) \n\t\t - y = ax + b => a = %f ; b = %f" % (w[0], w[1], w[2], a, b))

    # Se añade el número de iteraciones necesitadas para conseguir el ajuste
    iterations.append(itera)
    
    input("\n\t--- Pulsar enter para continuar con el siguiente valor ---\n")
    
print('\n -> Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar enter para continuar con el apartado 2.2 ---\n")

###############################################################################
###############################################################################

# Apartado 2.2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# Algoritmo del Gradiente Descendente Estocástico con Regresión Logística
# ENTRADA:
#   -> muestra: datos utilizados para el ejuste con etiquetas
#   -> w: pesos de inicio
def sgdRL(muestra,w):
    
    # Número de iteraciones
    itera = 0
    
    # Se le añade la columna de unos a los datos
    datos = np.hstack((np.ones((len(muestra),1)), muestra))
    
    # Pesos 
    wnew = np.copy(w)
    wold = np.ones_like(w)
    
    # Constante de aprendizaje
    rate = 0.01
    
    # Tamaño de batch
    tam_batch = 1
    
    # Punto de inicio y fin para escoger los datos para el batch 
    inicio = 0
    fin = tam_batch
    
    # Se itera hasta que la norma de la resta de los pesos conseguidos en la 
    # iteración anterior y los pesos conseguidos en esta iteración 
    while (np.linalg.norm(wold - wnew) > 0.01):
        
        # Se guardan los pesos de la iteración anterior
        wold = wnew
        
        # Se mezclan los datos para una nueva época
        np.random.shuffle(datos)
        
        # Valores para coger el primer batch 
        inicio = 0
        fin = tam_batch 
        
        # Se itera sobre toda la época
        while (inicio <= len(muestra) - tam_batch): 
            
            # Se escoge el batch que se va a estudiar en esa iteración
            batch = datos[inicio:fin]
            
            # Se actualizan los pesos en función al error conseguido
            wnew = wnew - rate * GradFerror(batch, wnew)
            
            # Se aumenta el punto de inicio
            inicio = inicio + tam_batch
            
            # Se aumenta el punto final
            fin = fin + tam_batch 
            
            if (fin > len(muestra)):
                fin = len(muestra)    
        
        # Se aumenta el número de iteraciones
        itera += 1
    
    print ("Iteraciones: ", itera)
            
    return wnew

# Función que determina el error de ajuste con los pesos conseguidos
def Error (muestra, w, a, b):
   
    error = 0
    
    # Se recorren los datos de la muestra
    for d in muestra:
        
        # Si la etiqueta asignada no corresponde con la etiqueta real, se suma
        # uno al número de errores
        if (Etiquetar(d,w) != F(d[0], d[1], a, b)):
            error += 1
    
    # Se calcula la media de errores cometidos en la asignación de datos en el 
    # ajuste
    error = error/len(muestra)
    
    return error
    
# Función del gradiente del Error Logístico
def GradFerror (datos, w):
    
    error = 0
    
    # Se recorren todos los datos
    for d in datos:
        
        error += -d[3] * d[0:3] * MEP(-d[3] * np.transpose(w) @ d[0:3])
    
    # Se calcula la media del error
    error =  error/len(datos)
    
    return error

# Función sigma
def MEP (x):
    
    return (1 / (1 + math.exp(-x)))

# Función para etiquetar los datos en función de los pesos 
def Etiquetar (x, w):
    
    suma = w[0] + x[0] * w[1] + x[1] * w[2]
    
    return Signo(suma)

###############################################################################

# Se establece una nueva semilla
np.random.seed(8)

# Se generan los datos para el train y el test
train = simula_unif(100, 2, [0, 2])
test = simula_unif(1000, 2, [0,2])

# Se simula la recta 
a, b = simula_recta([0,2])

# Se asignan las etiquetas
lbstrain = []
aux = []

for t in train:
    
    etiqueta = F(t[0], t[1], a, b)
    
    lbstrain.append(etiqueta)
    aux.append((t[0], t[1], etiqueta))
    
lbstrain = np.array(lbstrain)
train = np.array(aux)

# Se inicializan el vector de pesos inicial a 0
pesos = np.zeros((len(train[0]),))

# Se ejecuta el algoritmo para conseguir los pesos
pesos = sgdRL(train,pesos)

# Se calculan las constantes de las rectas
A = (-(pesos[0] / pesos[2]) / (pesos[0] / pesos[1])) 
B = - pesos[0] / pesos[2]

# Se pintan los datos y el ajuste conseguido
plt.title ("Datos del Train con recta de ajuste")
plt.plot([0.4, 0.75], [0.4 * A + B, 0.75 * A + B], c='b', label="prediccion")
plt.plot([0.53, 0.6], [0.53 * a + b, 0.6 * a + b], c='k', label="ajuste")

plt.scatter(np.squeeze(train[np.where(lbstrain == -1),0]), np.squeeze(train[np.where(lbstrain == -1),1]), c='r', s=10, label="0")
plt.scatter(np.squeeze(train[np.where(lbstrain == 1),0]), np.squeeze(train[np.where(lbstrain == 1),1]), c='b', s=10, label="1")

plt.xlabel('Coordenada x1')

plt.ylabel('Coordenada x2')

plt.legend()
plt.show()

print ("\t -> g(x) = %fx %f" % (A, B))

input("\n--- Pulsar enter para continuar con la clasificación de nuevos datos ---\n")

# Se calcula el error en los datos de test
error = Error(test, pesos, a, b)

# Se calculan las etiquetas de los datos
y = []
prediccion = []

for t in test:
    y.append(F(t[0], t[1], a, b))
    
    if (F(t[0], t[1], a, b) == F(t[0], t[1], A, B)):
        prediccion.append(F(t[0], t[1], A, B))
    else:
        prediccion.append(0)
    
y = np.array(y)
prediccion = np.array(prediccion)

# Se pintan los datos y el ajuste conseguido
plt.title ("Clasificación del test")
plt.plot([0.4, 0.75], [0.4 * A + B, 0.75 * A + B], c='k', label="ajuste")

plt.scatter(np.squeeze(test[np.where(prediccion == -1),0]), np.squeeze(test[np.where(prediccion == -1),1]), c='r', s=10, label="0")
plt.scatter(np.squeeze(test[np.where(prediccion == 1),0]), np.squeeze(test[np.where(prediccion == 1),1]), c='b', s=10, label="1")
plt.scatter(np.squeeze(test[np.where(prediccion == 0),0]), np.squeeze(test[np.where(prediccion == 0),1]), c='g', s=10, label="Error")

plt.xlabel('Coordenada x1')

plt.ylabel('Coordenada x2')

plt.legend()
plt.show()

print ("El error en el test es de ", error)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

"""

 EJERCICIO 3 - BONUS: Clasificación de Dígitos

"""

print ('\n\t--- EJERCICIO 3 - BONUS ---')

###############################################################################
###############################################################################

# Función para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Error de la Regresión Lineal
def LinearRegressionE (x, y, w):
    
    error = 0
    
    # Se recorren todos los datos
    for i in range (0, len(x)):
        
#        error += (Signo(w[2] + w[0] * x[i,0] + w[1] * x[i,1] ) - y[i])**2
    
        # Si la etiqueta no corresponde, se suma 1 al error
        if (Signo(w[2] + w[0] * x[i,0] + w[1] * x[i,1] ) != y[i]):
            error += 1
            
    error = error / len(x)
    
    return error

# Error fuera de la muestra
# Se calcula el error esperado en la población
def ErrorOut (ein, x, y, w):
    # TRIAN 
    # FULL SUMARY -> OUT H = 1
    error = (4 * ((2 * len(x))**len(w) + 1)) / 0.05
    
    error = 8 * math.log(error)
    
    error = error / len(x)
    
    error = ein + math.sqrt(error)
    
    return error

# Algoritmo de Bolsillo
# ENTRADA:
#   -> muestra: datos a clasificar
#   -> labels: etiquetas de los datos
def Pocket (muestra, labels):
    
    # Se calcula el mejor peso con la primera iteración, valor de inicio 0
    wmejor, a = ajusta_PLA(muestra, labels, 100, 0)
    
    # Se guarda el error conseguido con esos pesos
    Emejor = LinearRegressionE (muestra, labels, wmejor)
    
    # Se hacen 50 iteraciones para buscar el mejor ajuste
    for i in range (1, 50):
        
        # Se calcula el vector de pesos y el error
        w, a = ajusta_PLA(muestra, labels, 100, i)
        E = LinearRegressionE (muestra, labels, w)
        
        # Si el error conseguido es menor, se guarda como el mejor
        if (E < Emejor):
            wmejor = w
            Emejor = E
    
    return wmejor

###############################################################################

# a) Usando los datos 'muestra'
print (" \nApartado a - Lectura de datos" )

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])

# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

# Se muestran los datos para el train
print ("\n -> Datos para el Training")

plt.title ("Training")
plt.scatter(np.squeeze(x[np.where(y == -1),0]), np.squeeze(x[np.where(y == -1),1]), c='r', s=10, label="4")
plt.scatter(np.squeeze(x[np.where(y == 1),0]), np.squeeze(x[np.where(y == 1),1]), c='b', s=10, label="8")

plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')

plt.legend()
plt.show()

input("\n\t--- Pulsar enter para continuar con el siguiente conjunto de datos ---\n")

# Se muestran los datos para el test
print ("\n -> Datos para el Test")

plt.title ("Test")
plt.scatter(np.squeeze(x_test[np.where(y_test == -1),0]), np.squeeze(x_test[np.where(y_test == -1),1]), c='r', s=10, label="4")
plt.scatter(np.squeeze(x_test[np.where(y_test == 1),0]), np.squeeze(x_test[np.where(y_test == 1),1]), c='b', s=10, label="8")

plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')

plt.legend()
plt.show()

input("\n\t--- Pulsar enter para continuar con el siguiente apartado ---\n")

print ("\nApartado b - Ajuste de los datos")

# Se calculan los pesos del ajuste con el algoritmo del bolsillo
pesos = Pocket(x, y)

# Se calculan las constantes de la recta
a = (-(pesos[2] / pesos[1]) / (pesos[2] / pesos[0])) 
b =  (-pesos[2] / pesos[1])

# Se calcula el error en el test y en el train 
errorTest = LinearRegressionE(x, y, pesos)
errorTrain = LinearRegressionE(x_test, y_test, pesos)

# Se muestran los datos con el ajuste y el error generado
print ("\n -> Ajuste de los datos del Train")

plt.title ("Training")
plt.plot([0.1, 0.5], [0.1 * a + b, 0.5 * a + b], c='k', label="prediccion")

plt.scatter(np.squeeze(x[np.where(y == -1),0]), np.squeeze(x[np.where(y == -1),1]), c='r', s=10, label="4")
plt.scatter(np.squeeze(x[np.where(y == 1),0]), np.squeeze(x[np.where(y == 1),1]), c='b', s=10, label="8")

plt.xlabel('Intensidad promedio')

plt.ylabel('Simetria')

plt.legend()
plt.show()

print ("\t -> g(x) = %fx + %f" % (a, b))
print ("\t -> Error Train: ", errorTrain)

input("\n\t--- Pulsar enter para continuar con los datos del Test ---\n")

print ("\n -> Ajuste de los datos del Test")

plt.title ("Test")
plt.plot([0.1, 0.5], [0.1 * a + b, 0.5 * a + b], c='k', label="prediccion")

plt.scatter(np.squeeze(x_test[np.where(y_test == -1),0]), np.squeeze(x_test[np.where(y_test == -1),1]), c='r', s=10, label="4")
plt.scatter(np.squeeze(x_test[np.where(y_test == 1),0]), np.squeeze(x_test[np.where(y_test == 1),1]), c='b', s=10, label="8")

plt.xlabel('Intensidad promedio')

plt.ylabel('Simetria')

plt.legend()
plt.show()

print ("\t -> Error Test: ", errorTest)

input("\n\t--- Pulsar enter para continuar con el siguiente apartado ---\n")

print ("\nApartado c - Verdadero Eout")

# Se calcula la cota del error esperado según el error del train y el error del
# test 
EoutEin = ErrorOut (errorTrain, x, y, pesos)

EoutEtest = ErrorOut (errorTest, x_test, y_test, pesos)

print ("\n -> Cota de Eout basada en Ein:   Eout <= ", EoutEin)
print ("\n -> Cota de Eout basada en Etest: Eout <= ", EoutEtest)
