import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
#Sacar esto, no creo que nos lo dejen usar
from scipy.linalg import solve_triangular

def construye_adyacencia(D,m):
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(A):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    # Chequeanos que la matriz es cuadrada
    if m!=n:
        print('Matriz no cuadrada')
        return

    # Una vez que sabemos que la matriz esta bien, empezamos a calular la Factorizacion
    for j in range(n):
        for i in range(j+1, n):
            l = Ac[i,j]/Ac[j,j]
            Ac[i,j:] = Ac[i,j:] - l * Ac[j,j:]
            Ac[i,j] = l

    
    # Con la factorizacion hecha, asignamos nuestra L y U como corresponda
    L = np.tril(Ac,-1) + np.eye(m)
    U = np.triu(Ac)

    return L, U


# Matriz de Calculo de factorizacion LU con Permutacion no usada en este TP
def calculaLUconP(A):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    P = np.eye(m)

    if m!=n:
        print('Matriz no cuadrada')
        return

    for j in range(n):
        for i in range(j+1, n):
            if(A[j,j] == 0):
                #Chequeo por permutacion de A
                for k in range(j+1,n):
                  if(A[k, j] != 0):
                    fperm = k

                permutarA = Ac[k,:]
                permutarP = P[k,:]
                Ac[k,:] = Ac[j,:]
                P[k,:] = P[j,:]
                Ac[j,:] = permutarA
                P[j,:] = permutarP

            l = Ac[i,j]/Ac[j,j]
            Ac[i,j:] = Ac[i,j:] - l * Ac[j,j:]
            Ac[i,j] = l

    # Falta probar que la permutacion funciona

    L = np.tril(Ac,-1) + np.eye(A.shape[0])
    U = np.triu(Ac)

    return L, U, P

def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    Kinv = calcular_matriz_invertida(calcular_matriz_K(A)) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    #> La inversa de K se calcula mas facil y rapido aprovechando que es diagonal
    C = A.transpose() @ Kinv # Calcula C multiplicando Kinv y A
    return C

def calcular_matriz_K(A):
  # Función para calcular la matriz de Grado para calcular C
  # A: Matriz de adyacencia
  # Retorna la matriz de grado K
  K = np.zeros(A.shape)
  for i in range(A.shape[0]):
    K[i,i] = np.sum(A[i,:])
  return K

#> No vale calcular la inversa asi, usen LU.
def calcular_matriz_invertida(A):
  # Función para calcular la matriz invertida de una Matriz
  # A: Matriz Original
  # Retorna la matriz A pero invertida

  # Calculo de Matriz de Cofactores
  Cof = np.zeros(A.shape)

  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      # Matriz A con Fila i y Columna j borrados
      CofParte = np.delete(np.delete(A,i,0),j,1)

      # Calculo de cada elemento de la matriz de Cofactores
      Cof[i,j] = (-1)**(i+j) * np.linalg.det(CofParte)

  # Calculo de la inversa
  Inv = Cof.transpose() / np.linalg.det(A)

  return Inv

# Correccion de funcion
#< Nueva funcion para calcular la invertida de una matriz digonal eficientemente
def invertida_de_matriz_diagonal(A):
   # Calcula eficientemente la inversa de una matriz diagonal
   # A: Matriz a invertir
   # Retorna A invertida
    n = len(A)
    inversa = []

    for i in range(n):
        fila = []
        for j in range(n):
            if i == j:
                fila.append(1 / A[i][j]) # Asumo que la entrada es correcta, sino deberia chequear que la diagonal no sea 0.
            else:
                fila.append(0)
        inversa.append(fila)

    return inversa

#<
def inversa_desde_LU(A):
    # Función para calcular la matriz invertida a partir de la descomposicion LU de una Matriz
    # A: Matriz a invertir
    # Retorna la matriz A^-1
    L, U = calculaLU(A)
    n = L.shape[0]
    I = np.eye(n)
    inv_A = np.zeros((n, n))

    for i in range(n):
        # Resolvemos L y = e_i
        y = solve_triangular(L, I[:, i], lower=True)
        # Resolvemos U x = y
        x = solve_triangular(U, y)
        # Guardamos la columna resultante de la inversa
        inv_A[:, i] = x

    return inv_A

def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (A.shape[0]/alfa)*(np.eye(A.shape[0]) - (1-alfa) * C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones((N, 1)) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p.transpose()

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # D: Matriz Distancias
    # Retorna la matriz C en versión continua

    A = D.copy()
    C = np.zeros(A.shape)

    #> Esta es la forma desarrollada que esta bien pero es mas lenta. Con numpy quedaba mas rapido y corto usando 
    #> una ecuacion casi identica a la de la C original
    #for i in range(A.shape[0]):
    #  dist_total = 0
    # Calculamos la distancia total de la fila
    #  for j in range(A.shape[1]):
    #    if (i != j):
    #      dist_total += 1/A[i,j]
    #
    # USamos la distancia total para calcular el resto de distancias de la matriz
    #  for j in range(A.shape[1]):
    #    if (i != j):
    #      C[j,i] = (1/A[i,j])/dist_total

    #> una ecuacion casi identica a la de la C original
    #< Ahora evitando los for explicitos usando numpy

    # Lleno la diagonal con inf para evitar dividir por 0
    np.fill_diagonal(A, np.inf)

    # Invertimos todas las distancias excepto la diagonal que queda en 0
    inv_A = 1 / A

    # Calculamos la distancia total de la fila
    dist_total = inv_A.sum(axis=1)

    # Normalizamos cada fila
    # Transponemos porque el valor C[j, i] depende de A[i, j]
    C = (inv_A.T / dist_total).T

    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matriz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    
    B = np.eye(C.shape[0])
    C_exp = C.copy()

    for i in range(0, cantidad_de_visitas-1):
      B += C_exp
      C_exp = C_exp @ C

    return B

def Calcular_v(B, w):
    # Función para calcular Visitantes finales usando LU
    # B: Explicada den Calcular_B
    # w: Matriz de Visitas iniciales
    # Retorna: Un vector v con las visitas de cada museo
    L, U = calculaLU(B) # Calculamos descomposición LU a partir de C y d

    Uv = scipy.linalg.solve_triangular(L,w,lower=True) # Primera inversión usando L
    v = scipy.linalg.solve_triangular(U,Uv) # Segunda inversión usando U
    return v

def Norma1_Matriz(A):
  # Funcion para Calcular La norma 1 de una Matriz
  # A: Matriz a calcular norma
  # Retorna: Valor de la Norma 1 de la matriz

  maximo = 0

  for i in range(A.shape[0]):
    suma = 0
    for j in range(A.shape[1]):
      suma += abs(A[j,i])
    if (suma > maximo):
      maximo = suma

  return maximo

def NumCond1_Matriz(A):
  # Funcion para Calcular El numero de condicion 1 de una Matriz
  # A: Matriz a calcular numero de condicion 1
  # Retorna: Valor del numero de condicion 1 de la matriz

  return Norma1_Matriz(A) * Norma1_Matriz(calcular_matriz_invertida(A))

def GraficoCiudad(A, ranking, museos, barrios):
  # Funcion para graficar ranking con el mapa de la
  # D: Matriz de adyacencia
  # Ranking: Puntuacion de cada ciudad
  # Museos y Barios: Variables para informacion Geografica
  # Retorna: Valor del numero de condicion 1 de la matriz

  G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
  # Construimos un layout a partir de las coordenadas geográficas
  G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}


  factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
  fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
  barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
  pr = ranking# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
  pr = pr/pr.sum() # Normalizamos para que sume 1
  Nprincipales = 3 # Cantidad de principales

  if len(pr.shape) == 2:
    pr = pr[0]

  principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
  labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
  nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
  nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres

  return principales

def GraficoCiudadesPorConexiones(D, conexiones, museos, barrios, titulo, subtitulo, x, y):
  # Funcion para graficar ranking con el mapa de la
  # D: Matriz de adyacencia
  # Ranking: Puntuacion de cada ciudad
  # Museos y Barios: Variables para informacion Geografica
  # Retorna: Valor del numero de condicion 1 de la matriz

  #Creacion unica de variables de los graficos
  puntuaciones = []
  principales = []
  G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
  factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles

  # Crear una figura con x filas y y columnas
  fig, axes = plt.subplots(x, y, figsize=(16, 16))

  # Título principal
  fig.suptitle(titulo, fontsize=16, fontweight='bold')

  for i in range(0,x):
     for j in range(0,y):
      posicion = i+j+(x-1)*i 


      A = construye_adyacencia(D,conexiones[posicion])
      
      G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
      barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=axes[i,j]) # Graficamos Los barrios
      pr = calcula_pagerank(A, 0.2)  #Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar

      pr = pr/pr.sum() # Normalizamos para que sume 1
      puntuaciones.append(pr[0])
      Nprincipales = 3 # Cantidad de principales

      if len(pr.shape) == 2:
        pr = pr[0]


      principales = np.concatenate((principales, np.argsort(pr)[-Nprincipales:]))
      labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos

      nx.draw_networkx(G,G_layout,ax = axes[i,j],node_size = pr*factor_escala,with_labels=False)
      axes[i, j].set_title(subtitulo + str(conexiones[posicion]))

  # Ajustar el espacio entre los subgráficos
  plt.tight_layout(rect=[0, 0, 1, 0.96])

  # Mostrar la figura
  plt.show()

  return puntuaciones,np.array(list(set(principales)))