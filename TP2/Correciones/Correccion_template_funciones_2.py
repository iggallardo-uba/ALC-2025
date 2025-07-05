import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
from scipy.linalg import solve_triangular

from Correccion_template_funciones import *

# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])

# Funciones Auxiliares
def calcular_vector_K(A):
    # Función para calcular el vector de Grado con la funcion de Matriz K
    # A: Matriz de adyacencia
    # Retorna el vector de grado K (Diagonal de la Matriz K)
    return np.diag(calcular_matriz_K(A))

def simetrizacionDeA(A):
    #La funcion recibe la matriz de adyaciencia A de museos y la simetriza
    # A: Matriz de adyacencia de museos
    # Retorna la matriz pero no dirigida, si no simetrica
    return (A+A.transpose())/2

def vector_s(v):
    # funcion para calcular vector s tal que el signo del v es el valor de cada s
    # v = Autovector de la matriz
    # Retorna el vector s

    s = v*(1/np.absolute(v)) # Asumo no valores 0

    return s

def valor_2E(A):
    # Funcion que calcula cantidad de conexiones en A
    # A = Matriz de Adiencencia
    # Retorna el valor E
    
    return sum(sum(A))

def GraficoCiudadModularidad(A, museos, barrios, comunidades, conexiones):
    # Funcion para graficar ranking con el mapa de la
    # D: Matriz de adyacencia
    # Museos y Barios: Variables para informacion Geografica
    # Retorna: Valor del numero de condicion 1 de la matriz

    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}

    node_colors = [0] * len(G.nodes)
    for idx, community in enumerate(comunidades):
        for node in community:
            node_colors[node] = idx  # assign a unique color index per community

    factor_escala = 200 # Escalamos los nodos 10 mil veces para que sean bien visibles
    fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    nx.draw_networkx(G,G_layout,node_size = factor_escala,node_color=node_colors, ax=ax,with_labels=False) # Graficamos red
    ax.set_title("Grafico de Modularidad con m = " + str(conexiones))

def GraficoCiudadCorteLaplaciano(A, museos, barrios, comunidades, conexiones, cortes):
    # Funcion para graficar ranking con el mapa de la
    # D: Matriz de adyacencia
    # Museos y Barios: Variables para informacion Geografica
    # Retorna: Valor del numero de condicion 1 de la matriz

    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}

    node_colors = [0] * len(G.nodes)
    for idx, community in enumerate(comunidades):
        for node in community:
            node_colors[node] = idx  # assign a unique color index per community

    factor_escala = 200 # Escalamos los nodos 10 mil veces para que sean bien visibles
    fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    nx.draw_networkx(G,G_layout,node_size = factor_escala,node_color=node_colors, ax=ax,with_labels=False) # Graficamos red
    ax.set_title("Grafico de Cortes Laplaciano con m = " + str(conexiones) + " con " +str(cortes) + " niveles")

def GraficosCiudadCortesConjunto(D, museos, barrios, m ,cortes, x,y):
    # Crear una figura con x filas y y columnas
    fig, axes = plt.subplots(x, y, figsize=(16, 16))

    # Título principal
    fig.suptitle("Graficos de Cortes con m = " + str(m), fontsize=16, fontweight='bold')

    factor_escala = 200
    
    A = simetrizacionDeA(construye_adyacencia(D, m))

    for i in range(0,x):
        for j in range(0,y):
            posicion = i+j+(x-1)*i 

            G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
            G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
            barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=axes[i,j]) # Graficamos Los barrios

            AnalisisLaplaciano = laplaciano_iterativo(A,cortes[posicion])
            
            node_colors = [0] * len(G.nodes)
            for idx, community in enumerate(AnalisisLaplaciano):
                for node in community:
                    node_colors[node] = idx 

            nx.draw_networkx(G,G_layout,ax = axes[i,j],node_size = factor_escala,node_color=node_colors,with_labels=False)
            axes[i, j].set_title("Grafico de cortes laplaciono con m =" + str(m) + " y nivel " + str(cortes[posicion]))

#> Pueden importar las cosas del template 1, no hace falta copiar y pegar
# Gallar > Noted, despues veo que se queda y que no

# Funciones del TP
def calcula_L(A):
    # Funcion que se encarga de calcular la Matriz Laplaciana
    # A: Matriz de Adyaciencia
    # Retorna solo la matriz Laplaciana

    #Calculamos la matriz K nuestra A
    K = calcular_matriz_K(A)

    #Devolvemos la Matriz L
    return K - A

def calcula_R(A):
    # Funcion para carcular la matriz de Modularidad
    # A: Matriz de Adyanciencia
    # Retorna  la matriz de Modularidad R de A

    #Calculamos nuestra K y E
    K = calcular_vector_K(A)
    div = valor_2E(A)

    #Con todo esto podemos calcular P
    P = np.outer(K,K)/div

    #Devolvemos el R final
    return A-P

# Ver para que estaba
def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    
    s = vector_s(v)
    
    return (s.traspose()@L@s)/4

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    
    #Calculemos s de v
    s = vector_s(v)
    
    # Calcules Q optimizada
    Q = s.transpose()@R@s
    
    return Q

def metpot1(A,tol=1e-8,maxrep=np.inf):
    # Funcion para calcular el primer autovector y autovalor de una matriz A
    # A: Matriz a aplicar el metodo
    # Retorna:
    # - v1: Autovector calculado
    # - l1: Autovalor calculado
    # - _: True si el metodo convergio, False si se paso de repeticiones
    
    # Variables de Inicio
    nrep = 0
    
    # Generamos un vector de partida aleatorio, entre -1 y 1. Despues lo normalizamos
    v = np.random.uniform(low=-1.0, high=1.0, size=A.shape[0]) 
    v = v/np.linalg.norm(v) 

    # Aplicamos el metodo de la Potencia, y calculamos los autovalores correspondientes
    v1 = A@v 
    v1 = v1/np.linalg.norm(v1)

    l = (v.transpose()@A@v)/(v.transpose()@v) 
    l1 = (v1.transpose()@A@v1)/(v1.transpose()@v1)  
    
    # Repetimos el metodo hasta que converga a la tolerancia o se pase de repeticiones
    while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep:
        # Reiniciamos condicionees iniciales
        v = v1 
        l = l1
        
        # Aplicamos el metodo nuevamente
        v1 = A@v 
        v1 = v1/np.linalg.norm(v1) 
        l1 = (v1.transpose()@A@v1)/(v1.transpose()@v1)

        nrep += 1 # Un pasito mas
    
    if not nrep < maxrep:
        print('MaxRep alcanzado')
    
    return v1,l1,nrep<maxrep

def deflaciona(A, v, l, tol=1e-8,maxrep=np.inf):
    # Funcion que deflaciona el autovector V de la matriz A para que su autovalor L pase a ser 0
    # A: Matriz a Deflacionar
    # v: Autovector que queremos deflacionnar
    # l: Valor del autovalor del autovector a deflacionar 
    # Retorna la Matriz A deflacionada
    
    deflA = A - l * np.outer(v, v)

    return deflA

def metpot2(A,tol=1e-8,maxrep=np.inf):
   # Funciona para buscar el segundo autovector y autovalor de la matriz A
   # A: Matriz a aplicar el metodo
   # Retorna el segundo autovector y autovalor
   
   # Calculamos el primer autovalor y autovector
   v,l,_ = metpot1(A)
   
   # Se los sacamos a A para calcular el siguiente
   deflA = deflaciona(A,v,l, tol, maxrep)
   
   # devolvemos los resultados del metodo de la potencia en la Matriz deflacionada
   return metpot1(deflA,tol,maxrep)

def metpotI(A,mu=0.0,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    # A: Matriz a aplicar el metodo
    # mu: El shifteo que realizamos
    
    # Primero calculemos La matriz invertida con su shifteo
    Ainv = inversa_desde_LU(A + np.eye(A.shape[0]) * mu)
    
    #Calculamos con el metodo de Potencia el autovector mas chico y su autovalor
    v,l,_ = metpot1(Ainv, tol, maxrep)
    
    # Reobtenemos el autovalor que le corresponde a la matriz A
    l = 1/l 
    l -= mu
    
    return v,l,_

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Retorna el segundo autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
   # A: Matriz a aplicar el metodo
   # mu: El shifteo que realizamos
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + np.eye(A.shape[0]) * mu # Calculamos la matriz A shifteada en mu
   iX = inversa_desde_LU(X) # La invertimos
   Vinv, Linv, _ = metpot1(iX, tol, maxrep)
   defliX = deflaciona(iX,Vinv, Linv) # La deflacionamos
   v,l,_ =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   # Valen > Aca tienen un bug! Volvieron a sumar mu en vez de restarlo
   # l += mu
   # Gallar > Bueno vamos a restarlo para arreglar el tema
   l -= mu
   return v,l,_

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Funcion que calcula los cortes minimos recursivamente de nuestra matriz de Adyaciencia
    # A: Matriz de Adyaciencia
    # niveles: cuantos cortes va a realizar
    # nombres_s: posiciones o nombres que va recortando el metodo
    # Retorna una lista de conjuntos de nodos que respresentan las comunidades
    
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return [nombres_s]
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        
        #le damos un mu chico o cercano a cero para que calcule los menores autovectores
        v,l,_ = metpotI2(L,0.00001) # Encontramos el segundo autovector de L
        
        #realizemos un formateo de v para que sea mas facil operarlo
        v = np.where(v > 0, 1, -1)
        
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[np.ix_(v == 1, v == 1)] # Asociado al signo positivo
        Am = A[np.ix_(v == -1, v == -1)] # Asociado al signo negativo
        
        # Aplicamos el metodo para cada recorte que hicimos
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe la matriz de adyanciencias y calcula los cortes segun su modularidad
    # A: Matriz de adyaciencias
    # R: Matriz de Modularidad del paso anterior
    # nombres_s:posiciones o nombres que va recortando el metodo
    # Retorna una lista con conjuntos de nodos representando las comunidades

    # Chequeo de si estan bien los inputs
    if A is None and R is None:
        #print('Dame una matriz')
        return(np.nan)
    
    # Si es el primero, calculamos su modularidad
    if R is None:
        R = calcula_R(A)
    
    # Si no damos nombres, creamos los nuestros
    if nombres_s is None:
        nombres_s = range(R.shape[0])
        
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return nombres_s
    
    else:
        # Calculamos el primer autovector y autovalor de R para la modularidad
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        
        # Calculamos la Modularidad base 
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        
        #realizemos un formateo de v para que sea mas facil operarlo
        v = np.where(v > 0, 1, -1)

        # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
        if Q0<=0 or all(v>0) or all(v<0): 
            #print("Modularidad Adecuada")
            
            # Separamos y devolvemos los grupos
            Rp = [nombres_s[i] for i in range(len(v)) if v[i] == 1] # Parte de R asociada a los valores positivos de v
            Rm = [nombres_s[i] for i in range(len(v)) if v[i] == -1] # Parte asociada a los valores negativos de v
            
            return [Rp,Rm]
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            
            # Primero, separamos las comunidades en positivo y negativo
            ComunidadP = [nombres_s[i] for i in range(len(v)) if v[i] == 1] # Parte de R asociada a los valores positivos de v
            ComunidadN = [nombres_s[i] for i in range(len(v)) if v[i] == -1] # Parte asociada a los valores negativos de v
            
            # Segundo, pasamos las matrices de Adyaciencias segun la comunidad
            Apositiva = A[np.ix_(v == 1, v == 1)]
            Anegativa = A[np.ix_(v == -1, v == -1)]
            
            # Tercero, separamos las R Asociadas
            Rp = R[np.ix_(v == 1, v == 1)]
            Rn = R[np.ix_(v == -1, v == -1)]

            # Y calculamos sus autovectores correspondientes
            vPositivo,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vNegativo,lm,_ = metpot1(Rn) # autovector principal de Rm
        
            # Cuarto, formateamos los autovectores para que sea mas facil su operacion
            vp = np.where(vPositivo > 0, 1, -1)
            vn = np.where(vNegativo > 0, 1, -1)
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vn>0) or all(vn<0):
                Q1 += np.sum(Rn[vn>0,:][:,vn>0]) + np.sum(Rn[vn<0,:][:,vn<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return modularidad_iterativo(Apositiva, Rp, ComunidadP) +modularidad_iterativo(Anegativa, Rn, ComunidadN)
            
            
            # Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            
            # Primero, separamos las comunidades en positivo y negativo
            ComunidadP = [nombres_s[i] for i in range(len(v)) if v[i] == 1] # Parte de R asociada a los valores positivos de v
            ComunidadN = [nombres_s[i] for i in range(len(v)) if v[i] == -1] # Parte asociada a los valores negativos de v
            
            # Segundo, pasamos las matrices de Adyaciencias segun la comunidad
            Apositiva = A[np.ix_(v == 1, v == 1)]
            Anegativa = A[np.ix_(v == -1, v == -1)]
            
            # Valen >
            #> Estan calculando dos particiones en el mismo paso y entonces se estan salteando masomenos la mitad 
            #> de los cortes. Por cada paso deberia haber solo una R que parten en 2. Ustedes tienen 3.
            #> El codigo deberia ser casi igual al del laplaciano porque el procedimiento no cambia mucho.

            # Tercero, calculamos las Matrices de modularidad de cada comunidad
            Rpositiva = calcula_R(Apositiva)
            Rnegativa = calcula_R(Anegativa)
            
            # Y calculamos sus autovectores correspondientes
            vPositivo,lp,_ = metpot1(Rpositiva)  # autovector principal de Rp
            vNegativo,lm,_ = metpot1(Rnegativa) # autovector principal de Rm
        
            # Cuarto, formateamos los autovectores para que sea mas facil su operacion
            vPositivo = np.where(vPositivo > 0, 1, -1)
            vNegativo = np.where(vNegativo > 0, 1, -1)

            # Quinto, Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vPositivo>0) or all(vPositivo<0):
               Q1 = np.sum(Rpositiva[vPositivo>0,:][:,vPositivo>0]) + np.sum(Rpositiva[vPositivo<0,:][:,vPositivo<0])
            if not all(vNegativo>0) or all(vNegativo<0):
                Q1 += np.sum(Rnegativa[vNegativo>0,:][:,vNegativo>0]) + np.sum(Rnegativa[vNegativo<0,:][:,vNegativo<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles creados
                return modularidad_iterativo(Apositiva, Rpositiva, ComunidadP) +modularidad_iterativo(Anegativa, Rnegativa, ComunidadN)

