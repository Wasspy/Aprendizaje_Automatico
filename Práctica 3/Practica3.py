# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: Laura Rabadán Ortega
"""

# Bibliotecas
import time
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix

# Semilla para la generación de números aleatorios
np.random.seed(22)

##################################################################
# Lectura de los datos

# Función para leer los datos desde un fichero de entrada y dividirlos en
# característica y etiquetas
# ENTRADA:
#   -> path: camino del fichero del que se van a extraer los datos
#   -> delimiter: delimitador de datos utilizado en el fuchero
# SALIDA:
#   -> x: matriz de característica de los datos leídos
#   -> y: vector de etiquetas correspondientes a los datos
def LeerDatos (path, delimiter):

    # Se cargan los datos desde el fichero
    x = np.loadtxt(path, delimiter=delimiter)

    # Se guardan las etiquetas de los datos (última columna del conjunto original)
    y = x[:,-1:]

    # Se guardan las características sin las etiquetas
    x = x[:,:-1]

    return x, y

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# División de los datos
# Función para dividir el conjunto de datos en dos subconjuntos equivalentes de
# tamaños 0.8n y 0.2n, siendo n el núemro de datos
# ENTRADA:
#   -> x: conjunto de datos que se van a dividir
#   -> y: etiquetas de los datos que se van a dividir
# SALIDA:
#   -> Se utiliza una función de la biblioteca sklearn la cual divide el conjunto
#      de datos  con sus etiquetas correspondientes pasado como argumento. Se le
#      especifica el tamaño que se quiere que tenga el train y el test
#      (train = len(x)*0.8 ; test = len(x)*0.2), que se quiere que se mezclen los
#      datos antes de dividirlos y que la semilla para ello sea 22.
#      Esta función devuelve 4 conjuntos de datos:
#           -> train_x: conjunto de datos que servirá como train
#           -> test_x: conjunto de datos que servirá como test
#           -> train_y: etiquetas del conjunto de datos de train
#           -> test_y: etiquetas del conjutno de datos de test
def DividirDatos (x, y):

    return train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=22, shuffle=True)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Entrenar la clase encargada de seleccionar las Caracteristicas de los datos.
# LLAMAR UNA ÚNICA VEZ PARA CADA CONJUNTO DE DATOS
# Función que entrena el objeto que se utilizará para seleccionar los datos del conkunto
# de datos.
# ENTRADA:
#   -> datos: conjunto de datos con el que se va a entrenar la clase.
#   -> umbral: umbral que se quiere utilizar (float)
# SALIDA:
#   -> selector: objeto de la clase 'VarianceThreshold' entrenado con los datos
def EntrenaSeleccion (datos, umbral):

    selector = VarianceThreshold(threshold=umbral)
    selector.fit(datos)

    return selector

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Selección de Características
# Función para seleccionar las características que van a interesar para el problema,
# aquellas cuya varianza sea menor a un umbral. Para ello, se utiliza una clase
# de la biblioteca 'sklearn' que hace lo que se ha especificado, inicializandola
# con el umbral que se quiere utilizar y utilizándo un método propio, 'fit_transform'
# que elimina las características del conjunto de datos (sin etiquetas) pasado
# como argumento cuya varianza esté por debajo del umbral especificado.
# ENTRADA:
#   -> selector: objeto que se utiliza para seleccionar las características que
#      se van a eliminar
#   -> datos: conjunto de datos a reducir
# SALIDA:
#   -> x: conjunto de datos sin las características cuya varianza es menor al umbral
def FeatureSelection (selector, datos):

    return selector.transform(datos)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Entrenar la clase encargada de normalizar los datos.
# LLAMAR UNA ÚNICA VEZ PARA CADA CONJUNTO DE DATOS
# Función que entrena el objeto que se utilizará para escalar los datos del conkunto
# de datos.
# ENTRADA:
#   -> datos: conjunto de datos con el que se va a entrenar la clase.
# SALIDA:
#   -> scaler: objeto de la clase 'MinMaxScaler' entrenado con los datos
def EntrenaEscala (datos):

    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(datos)

    return scaler

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Normalización de los datos
# Función para normalizar los datos entre [0,1] en función a los valores del
# train.Para esto, se utiliza una clase de la biblioteca 'sklearn', 'MinMaxScaler',
# la cual entrena con los valores del train y normaliza los valores en función a
# los valores máximos y mínimos de cada característica.
# ENTRADA:
#   -> scaler: escala que se va a utilizar para normalizar los datos
#   -> datos: conjunto de datos a normalizar
# SALIDA:
#   -> x: conjunto de datos normalizado
def Normalizar (scaler, datos):

    return scaler.transform(datos)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Entrenar la clase encargada de transformarlos datos (Polinómica)
# LLAMAR UNA ÚNICA VEZ PARA CADA CONJUNTO DE DATOS
# Función que entrena el objeto que se utilizará para transformar los datos del conjunto
# de datos.
# ENTRADA:
#   -> datos: conjunto de datos con el que se va a entrenar la clase.
#   -> grado: grado de las características.
# SALIDA:
#   -> poly: objeto de la clase 'PolynomialFeatures' entrenado con los datos
def EntrenaPolinomial (datos, grado):

    poly = PolynomialFeatures(degree=grado, interaction_only=False)
    poly.fit(datos)

    return poly

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Transformación de los datos (Polimonial)
# Función para transformar los datoss. Para esto, se utiliza una clase de la
# biblioteca 'sklearn', 'PolynomialFeatures', la cual entrena con los valores del
# train y transforma las características de los datos de la siguiente manera:
#   [x1, x2] => [1, x1, x2, x1**2, x1x2, x2**2]
# ENTRADA:
#   -> poly: objeto de la clase con la que se va a transformar
#   -> datos: conjunto de datos a transformar
# SALIDA:
#   -> x: conjunto de datos transformado
def Polinomial (poly, datos):

    return poly.transform(datos)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Creación de grid de datos
# Función para crear el grid de datos a partir de los parametros pasados por argumento
# y el modelo a utilizar.
# ENTRADA:
#   -> modelo: modelo que va a usarse
#   -> parametros: parametros para la función
#   -> error: función co la que calcular el error
# SALIDA:
#   -> grid: Grid parael modelo que se quiere usar
def CrearGrid (modelo, parametros, error, particiones):

    grid = GridSearchCV(modelo, parametros, scoring=error, n_jobs=2, iid=False,
                   cv=particiones, verbose=0, return_train_score=True)

    return grid

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Transformar función a cuadrática
# Función para transformar un función a cuadrática y eliminar las Características
# nuevas que tengan un único valor
# ENTRADA:
#   -> poly: objeto de la clase con la que se va a transformar
#   -> datos: matriz de datos a transformar
#   -> grado: grado de las características.
# SALIDA:
#   -> x: datos transformados
def TransfCuadratica (poly, train, test, grado):

    x_train = Polinomial(poly, train)
    x_test = Polinomial(poly, test)

    selector = EntrenaSeleccion(x_train, grado)

    x_train = FeatureSelection(selector, x_train)
    x_test = FeatureSelection(selector, x_test)

    line = EntrenaPolinomial(x_train, 1)

    return Polinomial(line, x_train), Polinomial(line, x_test)


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Error en la muestra
# Función para estimar el error en la muestra
# ENTRADA:
#   -> y: etiquetas reales del conjunto
#   -> pred_y: predicción de las etiquetas del conjunto
# SALIDA:
#   -> error: estimación del error
def ErrorIn (y, pred_y):

    error = 0

    for i in range (0, len(y)):
        error += (pred_y[i]-y[i])**2

    return error/len(y)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# Error global para Clasificación
# Función para estimar el error global en el problema de clasificación
# ENTRADA:
#   -> ein: error en la muestra
#   -> len_x: tamaño de la muestra
#   -> tol: tolerancia
# SALIDA:
#   -> error: estimación del error
def ErrorClsf (ein, len_x, tol):

    error = math.log(2/tol)

    error = error/(2*len_x)

    return ein + math.sqrt(error)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

"""

 Práctica 3 - Ajuste de Modelos Lineales

"""

print ('\n\t--- PRÁCTICA 3 - Ajuste de Modelos Lineales ---\n')

"""
    1. Lectura de datos
"""
# Se leen los datos de los dos problemas

# Datos del problema de clasificación (en adelante, digit)
# Datos del train
digit_train_x, digit_train_y = LeerDatos('datos/optdigits.tra', ',')

# Datos del test
digit_test_x, digit_test_y = LeerDatos('datos/optdigits.tes', ',')

print (" Se leyó el fichero de datos 'optdigits'")
print ("\t -> Train: \t%d características ; \t%d ejemplos" % (len(digit_train_x[0]), len(digit_train_y)))
print ("\t -> Test: \t%d características ; \t%d ejemplos" % (len(digit_test_x[0]), len(digit_test_y)))

# Datos del problema de regresión (en adelante, airfoil)
# En este casos no hay dos ficheros distintos para test y train, por tanto se
# dividirán de forma 'manual'
airfoil_x, airfoil_y = LeerDatos('datos/airfoil_self_noise.dat', '\t')

airfoil_train_x, airfoil_test_x, airfoil_train_y, airfoil_test_y = DividirDatos(airfoil_x, airfoil_y)

print ("\n Se leyó el fichero de datos 'airfoil_self_noise'")
print ("\t -> Train: \t%d características ; \t%d ejemplos" % (len(airfoil_train_x[0]), len(airfoil_train_y)))
print ("\t -> Test: \t%d características ; \t%d ejemplos" % (len(airfoil_test_x[0]), len(airfoil_test_y)))

input ("\n Pulsar 'enter' para pasar al preprocesamiento de datos")

"""
    2. Preprocesamiento de datos
"""
# Selección de características interesantes

# Se eliminian las características con una varianza por debajo del umbral 'threshold',
# es decir, en este caso, las características con valores iguales en todos los casos

print ("\n Selección de carterísticas interesantes:")
# Selector para las características de los conjuntos
digit_selector = EntrenaSeleccion(digit_train_x, 0.0)
airfoil_selector = EntrenaSeleccion(airfoil_train_x, 0.0)

# Eliminación de características con valores iguales en el problema de clasificación
fs_digit_train_x = FeatureSelection(digit_selector, digit_train_x)
fs_digit_test_x = FeatureSelection(digit_selector, digit_test_x)

# Eliminación de características con valores iguales en el problema de regresión
fs_airfoil_train_x =  FeatureSelection(airfoil_selector, airfoil_train_x)
fs_airfoil_test_x = FeatureSelection(airfoil_selector, airfoil_test_x)

print ("\tNuevas dimensiones de los datos:")
print ("\t -> 'Digit': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(fs_digit_train_x[0]), len(digit_train_y)))
print ("\t\t - Test: \t%d características ; \t%d ejemplos" % (len(fs_digit_test_x[0]), len(digit_test_y)))
print ("\t -> 'Airfoil': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(fs_airfoil_train_x[0]), len(airfoil_train_y)))
print ("\t\t - Test: \t%d características ; \t%d ejemplos" % (len(fs_airfoil_test_x[0]), len(airfoil_test_y)))

print ("\n Normalización de los datos:")
# Normalización de los datos

print ("\tValores máximos y mínimos del conjunto de train:")
print ("\t -> 'Digit': \n\t\t%f (MAX) ; \t%f (min)" % ((fs_digit_train_x.max(axis=1)).max(), (fs_digit_train_x.min(axis=1)).min()))
print ("\t -> 'Airfoil': \n\t\t%f (MAX) ; \t%f (min)" % (fs_airfoil_train_x.max(axis=1).max(), fs_airfoil_train_x.min(axis=1).min()))
# Escala para los distintos conjuntos de datos
digit_scaler = EntrenaEscala(fs_digit_train_x)
airfoil_scaler = EntrenaEscala(fs_airfoil_train_x)

# Normalización del conjunto de datos 'digit'
nm_digit_train_x = Normalizar(digit_scaler, fs_digit_train_x)
nm_digit_test_x = Normalizar(digit_scaler, fs_digit_test_x)

# Normalización del conjunto de datos 'airfoil'
nm_airfoil_train_x = Normalizar(airfoil_scaler, fs_airfoil_train_x)
nm_airfoil_test_x = Normalizar(airfoil_scaler, fs_airfoil_test_x)

print ("\tNuevos máximos y mínimos del conjunto de train:")
print ("\t -> 'Digit': \n\t\t%f (MAX) ; \t%f (min)" % (nm_digit_train_x.max(axis=1).max(), nm_digit_train_x.min(axis=1).min()))
print ("\t -> 'Airfoil': \n\t\t%f (MAX) ; \t%f (min)" % (nm_airfoil_train_x.max(axis=1).max(), nm_airfoil_train_x.min(axis=1).min()))

input ("Pulsar 'enter' para pasar a la selección de clase de funciones" )

"""
    3. Selección de la clase de Funciones
"""

# Clases de funciones
# Se van a probar 2 clases distintas: la clase lineal y la cuadrática

# Lineales (Se le añade la priemra columna de 1, valor constante de la función)
digit_line = EntrenaPolinomial(nm_digit_train_x, 1)
airfoil_line = EntrenaPolinomial(nm_airfoil_train_x, 1)

ln_digit_train_x = Polinomial(digit_line, nm_digit_train_x)
ln_digit_test_x = Polinomial(digit_line, nm_digit_test_x)

ln_airfoil_train_x = Polinomial(airfoil_line, nm_airfoil_train_x)
ln_airfoil_test_x = Polinomial(airfoil_line, nm_airfoil_test_x)

# Cuadráticas
digit_poly = EntrenaPolinomial(nm_digit_train_x, 2)
d_airfoil_poly = EntrenaPolinomial(nm_airfoil_train_x, 2)
c_airfoil_poly = EntrenaPolinomial(nm_airfoil_train_x, 4)

cd_digit_train_x, cd_digit_test_x = TransfCuadratica (digit_poly, nm_digit_train_x, nm_digit_test_x, 0.0)

cd_airfoil_train_x, cd_airfoil_test_x = TransfCuadratica (d_airfoil_poly, nm_airfoil_train_x, nm_airfoil_test_x, 0.0)
cc_airfoil_train_x, cc_airfoil_test_x = TransfCuadratica (c_airfoil_poly, nm_airfoil_train_x, nm_airfoil_test_x, 0.0)

print ("\n Dimensiones del conjunto de train según la clase de funciones seleccionado:")
print (" -> Clase de Funciones Lineales:")
print ("\t -> 'Digit': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(ln_digit_train_x[0]), len(digit_train_y)))
print ("\t -> 'Airfoil': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(ln_airfoil_train_x[0]), len(airfoil_train_y)))
print (" -> Clase de Funciones Cuadráticas:")
print ("\t -> 'Digit': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(cd_digit_train_x[0]), len(digit_train_y)))
print ("\t -> 'Airfoil': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(cd_airfoil_train_x[0]), len(airfoil_train_y)))
print (" -> Clase de Funciones Cúbicas:")
print ("\t -> 'Airfoil': \n\t\t - Train: \t%d características ; \t%d ejemplos" % (len(cc_airfoil_train_x[0]), len(airfoil_train_y)))
input ("\nPulsar 'enter' para continuar con la elección de modelo")

# Conjunto de validación
# Como validación, para los dos conjutnos de datos, se utiliza la técnica de
# validación cruzada. Se dividirá el conjunto en 10 subconjuntos distintos y se
# entrenará 10 veces, dejando en cada caso un subconjutno apartado de la muestra,
# el cual se usará como conjunto de validación.
kf_validacion = KFold(n_splits=10, shuffle=True, random_state=22)

# Regularización
# Se harán distintas pruebas, aplicando los distintos tipos de regularizacion (L1,
# L2 y Elastic-Net) y el estudio sin aplicarle regularización
# Para ello, se utilizará el GridSearchCV, para estudiar cual es que proporciona
# mejores resultados

# Parámetros
lr_airfoil_parameters = {'fit_intercept':[True], 'normalize':[False], 'copy_X':[True], 'n_jobs':[4]}

lg_digit_parameters = {'penalty':['l2'], 'dual':[False], 'tol':[0.05], 'fit_intercept':[False],
                       'C':[0.9999], 'class_weight':[None], 'random_state':[22], 
                       'solver':['saga'], 'max_iter':[500], 'multi_class':['multinomial'], 
                       'n_jobs':[None]}

pc_digit_parapeters = {'penalty':['none','l2'], 'alpha':[0.0001], 'fit_intercept':[True], 
                       'max_iter':[1000], 'tol':[0.05], 'shuffle':[True], 'eta0':[1], 
                       'n_jobs':[4], 'random_state':[22], 'early_stopping':[True], 
                       'validation_fraction':[0.1], 'n_iter_no_change':[1,5], 
                       'class_weight':[None], 'warm_start':['False']}

# Regresión
ln_airfoil_grid = CrearGrid(LinearRegression(), lr_airfoil_parameters, 'neg_mean_squared_error', 10)

# Función lineal
print("\n Problema de regresión (Regresión Lineal)\n\t-> Función Lineal - Error cuadrático:")

inicio = time.time()
ln_airfoil_grid.fit(ln_airfoil_train_x, airfoil_train_y)
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print("\t\t - Error medio conseguido en el entrenamiento con VC: %f%%" % (-1*ln_airfoil_grid.best_score_))
print("\t\t - Error al estimar la muestra completa: %f%%" % (-1*ln_airfoil_grid.score(ln_airfoil_train_x, airfoil_train_y)))
print("\t\t - Error en cada conjunto de validación (10): ")

contador = 0
for train, test in kf_validacion.split(ln_airfoil_train_x):
    print("\t\t\t - Error en el conjunto %d: %f%%" % (contador, -1*ln_airfoil_grid.score(ln_airfoil_train_x[test,:], airfoil_train_y[test])))
    contador += 1

input ("\n Pulsar 'enter' para seguir el estudio de regresión con la clase de funciones cuadráticas")

# Función cuadrática
print("\n\t-> Función Cuadrática - Error cuadrático:")

cd_airfoil_grid = CrearGrid(LinearRegression(), lr_airfoil_parameters, 'neg_mean_squared_error', 10)

inicio = time.time()
cd_airfoil_grid.fit(cd_airfoil_train_x, airfoil_train_y)
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print("\t\t - Error medio conseguido en el entrenamiento con VC: %f%%" % (-1*cd_airfoil_grid.best_score_))
print("\t\t - Error al estimar la muestra completa: %f%%" % (-1*cd_airfoil_grid.score(cd_airfoil_train_x, airfoil_train_y)))
print("\t\t - Error en cada conjunto de validación (10):")

contador = 0
for train, test in kf_validacion.split(cd_airfoil_train_x):
    print("\t\t\t - Error en el conjunto %d: %f%%" % (contador, -1*cd_airfoil_grid.score(cd_airfoil_train_x[test,:], airfoil_train_y[test])))
    contador += 1

input ("\n Pulsar 'enter' para seguir el estudio de regresión con la clase de funciones cúbicas")

# Función cuadrática
print("\n\t-> Función Cúbica - Error cuadrático:")

cc_airfoil_grid = CrearGrid(LinearRegression(), lr_airfoil_parameters, 'neg_mean_squared_error', 10)

inicio = time.time()
cc_airfoil_grid.fit(cc_airfoil_train_x, airfoil_train_y)
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print("\t\t - Error medio conseguido en el entrenamiento con VC: %f%%" % (-1*cc_airfoil_grid.best_score_))
print("\t\t - Error al estimar la muestra completa: %f%%" % (-1*cc_airfoil_grid.score(cc_airfoil_train_x, airfoil_train_y)))
print("\t\t - Error en cada conjunto de validación (10):")

contador = 0
for train, test in kf_validacion.split(cc_airfoil_train_x):
    print("\t\t\t - Error en el conjunto %d: %f%%" % (contador, -1*cc_airfoil_grid.score(cc_airfoil_train_x[test,:], airfoil_train_y[test])))
    contador += 1

input ("\n Pulsar 'enter' para seguir el estudio con el problema de clasificación")

# Clasificación con 'Accuracy'
pc_ln_digit_grid = CrearGrid(Perceptron(), pc_digit_parapeters, 'accuracy', 5)

# Función lineal
print("\n Problema de clasificación (Perceptron)\n\t-> Función Lineal - 'Accuracy':")

inicio = time.time()
pc_ln_digit_grid.fit(ln_digit_train_x, digit_train_y[:,0])
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print("\t\t - Media de aciertos conseguido en el entrenamiento con VC: %f%%" % (pc_ln_digit_grid.best_score_*100))
print("\t\t - Porcentaje de acierto al estimar la muestra completa: %f%%" % (pc_ln_digit_grid.score(ln_digit_train_x, digit_train_y)*100))
print("\t\t - Porcentaje de acierto en cada conjunto de validación (10):")

contador = 0
for train, test in kf_validacion.split(ln_digit_train_x):
    print("\t\t\t - Porcentaje de acierto en el conjunto %d: %f%%" % (contador, pc_ln_digit_grid.score(ln_digit_train_x[test,:], digit_train_y[test])*100))
    contador += 1

ln_digit_pred_y = pc_ln_digit_grid.predict(ln_digit_train_x)
print ("\t\tMatriz de confusión: \n", confusion_matrix(digit_train_y,ln_digit_pred_y))

input ("\n Pulsar 'enter' para seguir el estudio de clasificación con Perceptron con la clase de funciones cuadráticas")

# Función cuadrática
print("\n\t-> Función Cuadrática - 'Accuracy':")

pc_cd_digit_grid = CrearGrid(Perceptron(), pc_digit_parapeters, 'accuracy', 5)

inicio = time.time()
pc_cd_digit_grid.fit(cd_digit_train_x, digit_train_y[:,0])
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print("\t\t - Media de aciertos conseguido en el entrenamiento con VC: %f%%" % (pc_cd_digit_grid.best_score_*100))
print("\t\t - Porcentaje de acierto al estimar la muestra completa: %f%%" % (pc_cd_digit_grid.score(cd_digit_train_x, digit_train_y)*100))
print("\t\t - Porcentaje de acierto en cada conjunto de validación:")

contador = 0
for train, test in kf_validacion.split(cd_digit_train_x):
    print("\t\t\t - Porcentaje de acierto en el conjunto %d: %f%%" % (contador, pc_cd_digit_grid.score(cd_digit_train_x[test,:], digit_train_y[test])*100))
    contador += 1

cd_digit_pred_y = pc_cd_digit_grid.predict(cd_digit_train_x)
print ("\t\tMatriz de confusión: \n", confusion_matrix(digit_train_y,cd_digit_pred_y))

input ("\n Pulsar 'enter' para seguir el estudio de clasificación con Regresión")

# Clasificación con 'Accuracy'
lr_ln_digit_grid = CrearGrid(LogisticRegression(), lg_digit_parameters, 'accuracy', 5)

# Función lineal
print("\n Problema de clasificación (Regresión Logística)\n\t-> Función Lineal - 'Accuracy':")

inicio = time.time()
lr_ln_digit_grid.fit(ln_digit_train_x, digit_train_y[:,0])
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print ("\t\t - Media de aciertos conseguido en el entrenamiento con VC: %f%%" % (lr_ln_digit_grid.best_score_*100))
print ("\t\t - Porcentaje de acierto al estimar la muestra completa: %f%%" % (lr_ln_digit_grid.score(ln_digit_train_x, digit_train_y)*100))
print ("\t\t - Porcentaje de acierto en cada conjunto de validación:")

contador = 0
for train, test in kf_validacion.split(ln_digit_train_x):
    print("\t\t\t - Porcentaje de acierto en el conjunto %d: %f%%" % (contador, lr_ln_digit_grid.score(ln_digit_train_x[test,:], digit_train_y[test])*100))
    contador += 1

ln_digit_pred_y = lr_ln_digit_grid.predict(ln_digit_train_x)
print ("\t\tMatriz de confusión: \n", confusion_matrix(digit_train_y,ln_digit_pred_y))

input ("\n Pulsar 'enter' para seguir el estudio de clasificación con Regresión Logística con la clase de funciones cuadráticas")

# Función cuadrática
lr_cd_digit_grid = CrearGrid(LogisticRegression(), lg_digit_parameters, 'accuracy', 5)
print("\n\t-> Función Cuadrática - 'Accuracy':")

inicio = time.time()
lr_cd_digit_grid.fit(cd_digit_train_x, digit_train_y[:,0])
fin = time.time()

print ("\t\t - Tiempo: %f s" % (fin-inicio))
print ("\t\t - Media de aciertos conseguido en el entrenamiento con VC: %f%%" % (lr_cd_digit_grid.best_score_*100))
print ("\t\t - Porcentaje de acierto al estimar la muestra completa: %f%%" % (lr_cd_digit_grid.score(cd_digit_train_x, digit_train_y)*100))
print ("\t\t - Porcentaje de acierto en cada conjunto de validación:")

contador = 0
for train, test in kf_validacion.split(cd_digit_train_x):
    print("\t\t\t - Porcentaje de acierto en el conjunto %d: %f%%" % (contador, lr_cd_digit_grid.score(cd_digit_train_x[test,:], digit_train_y[test])*100))
    contador += 1

cd_digit_pred_y = lr_cd_digit_grid.predict(cd_digit_train_x)
print ("\n\t\tMatriz de confusión: \n", confusion_matrix(digit_train_y,cd_digit_pred_y))

input ("\n Pulsar 'enter' para continuar con la estimación del error en la población")

ln_airfoil_pred_y = ln_airfoil_grid.predict(ln_airfoil_train_x)
cd_airfoil_pred_y = cd_airfoil_grid.predict(cd_airfoil_train_x)
cc_airfoil_pred_y = cc_airfoil_grid.predict(cc_airfoil_train_x)

pc_ln_digit_pred_y = pc_ln_digit_grid.predict(ln_digit_train_x)
pc_cd_digit_pred_y = pc_cd_digit_grid.predict(cd_digit_train_x)

lr_ln_digit_pred_y = lr_ln_digit_grid.predict(ln_digit_train_x)
lr_cd_digit_pred_y = lr_cd_digit_grid.predict(cd_digit_train_x)

print ("\n Errores globales estimados para ambos problemas con cada modelo y clase utilizado: ")
print (" -> Regresión: \n\t - Regresión Lineal \n\t\t - Lineal: \tEout =  %f%%" % (ErrorIn(airfoil_train_y, ln_airfoil_pred_y)[0]))
print ("\t\t - Cuadrática: \tEout =  %f%%" % (ErrorIn(airfoil_train_y, cd_airfoil_pred_y)[0]))
print ("\t\t - Cúbica: \tEout =  %f%%" % (ErrorIn(airfoil_train_y, cc_airfoil_pred_y)[0]))
print (" -> Calsificación (Con probabilidad de almenos 0.95 de que...): \n\t - Regresión Logística \n\t\t - Lineal : \tEout <= ", ErrorClsf(ErrorIn(digit_train_y, lr_ln_digit_pred_y)[0], len(digit_train_y), 0.05))
print ("\t\t - Cuadrática: \tEout <= ", ErrorClsf(ErrorIn(digit_train_y, lr_cd_digit_pred_y)[0], len(digit_train_y), 0.05))
print ("\t - Perceptron \n\t\t - Lineal : \tEout <= ", ErrorClsf(ErrorIn(digit_train_y, pc_ln_digit_pred_y)[0], len(digit_train_y), 0.05))
print ("\t\t - Cuadrática: \tEout <= ", ErrorClsf(ErrorIn(digit_train_y, pc_cd_digit_pred_y)[0], len(digit_train_y), 0.05))

input ("\n Pulsar 'enter' para continuar con la comprobación del error en el \n conjunto test con el modelo elegido para cada problema")

print ("\n Modelos elegidos: \n ->  Regresión: \n\t - Clase de funciones: cúbica \n\t - Modelo: Regresión Lineal")
print (" -> Clasificación: \n\t - Clase de funciones: cuadrática \n\t - Modelo: Perceptron")

cc_airfoil_grid.predict(cc_airfoil_test_x)
pc_cd_digit_grid.predict(cd_digit_test_x)

print ("\n Resultados al usar el modelo en el conjunto de test de los problemas:")
print (" -> Regresión: \t%f%% de error en el conjunto de test" % (-1*cc_airfoil_grid.score(cc_airfoil_test_x, airfoil_test_y)))
print (" -> Clasificación: \t%f%% de acierto en el conjunto de test" % (pc_cd_digit_grid.score(cd_digit_test_x, digit_test_y)*100))

print ("\n -> Regresión (cuadrática): \t%f%% de error en el conjunto de test" % (-1*cd_airfoil_grid.score(cd_airfoil_test_x, airfoil_test_y)))

