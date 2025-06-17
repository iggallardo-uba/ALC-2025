import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
#Sacar esto, no creo que nos lo dejen usar
from scipy.linalg import solve_triangular

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
def calcular_matriz_K(A):
    # Función para calcular la matriz de Grado para calcular C
    # A: Matriz de adyacencia
    # Retorna la matriz de grado K
    #K = np.zeros(A.shape)
    #for i in range(A.shape[0]):
    #  K[i,i] = np.sum(A[i,:])
    #return K

    # Nuevo
    return np.sum(A, axis=1, keepdims=True)

def simetrizacionDeA(A):
    #La funcion recibe la matriz de adyaciencia A y la simetriza para evitar errores
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

# Funcion de Inversion
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

# Funciones del TP
def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!

    #Calculamos con la nueva A nuestra K
    K = calcular_matriz_K(A)

    #Devolvemos la Matriz L
    return A-K

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!

    #Calculamos nuestra K y E
    K = calcular_matriz_K(A)
    #print("Grado: " + str(K))
    div = valor_2E(A)
    #print("Divisor: " + str(div))

    #Con todo esto podemos calcular P
    P = (K @ K.transpose())/div

    #Devolvemos el R final
    return A-P

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

def metpot1(A,tol=1e-8,maxrep=np.Inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.random.uniform(low=-1.0, high=1.0, size=A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v/np.linalg.norm(v) # Lo normalizamos
   #print(v)
   v1 = A@v # Aplicamos la matriz una vez
   v1 = v1/np.linalg.norm(v1)# normalizamos
   #print(v1)
   l = (v.transpose()@A@v) # Calculamos el autovector estimado
   #print("Valor primer l: " + str(l))
   l1 = (v1.transpose()@A@v1) # Y el estimado en el siguiente paso
   #print("Valor segundo l: " + str(l1))
   
   nrep = 0 # Contador
   
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      #print("repe num: " + str(nrep))
      
      v = v1 # actualizamos v y repetimos
      l = l1
      #print(v)
      #print("Valor primer l: " + str(l))
      v1 = A@v # Calculo nuevo v1
      v1 = v1/np.linalg.norm(v1) # Normalizo
      l1 = v1.transpose()@A@v1 # Calculo autovector
      #print(v1)
      #print("Valor segundo l: " + str(l1))
      nrep += 1 # Un pasito mas
   
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   
   return v1,l1,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.Inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    
    deflA = A - l1 * np.outer(v1, v1) # Sugerencia, usar la funcion outer de numpy

    return deflA

def metpot2(A,tol=1e-8,maxrep=np.Inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   deflA = deflaciona(A, tol, maxrep)
   
   return metpot1(deflA,tol,maxrep)

def metpotI(A,mu,tol=1e-8,maxrep=np.Inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    # Realizamos el shift y vemos LU
    
    M = A - np.eye(A.shape[0]) * mu
    
    Minv = inversa_desde_LU(M)
    
    v,l,_ = metpot1(Minv,tol=tol,maxrep=maxrep)
    
    l = 1/l # Reobtenemos el autovalor correcto
    l -= mu
    
    return v,l,_

def metpotI2(A,mu,tol=1e-8,maxrep=np.Inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + np.eye(A.shape[0]) * mu # Calculamos la matriz A shifteada en mu
   iX = inversa_desde_LU(X) # La invertimos
   defliX = deflaciona(iX,tol=1e-8,maxrep=np.Inf) # La deflacionamos
   v,l,_ =  metpotI(defliX, mu) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(L) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = ... # Asociado al signo positivo
        Am = ... # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        #print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return [nombres_s]
    else:
        print("Primera R:")
        print(R)
        print("autoval: " + str(np.linalg.eig(R)[0]))
        print("autovectores")
        print(np.linalg.eig(R)[1]) 
        
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        
        print("Primer autovector:" + str(v))
        print("Primer Autovalor: " + str(l))
        print("Separacion: " + str(nombres_s))
        
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        print("la que estaba: " + str(Q0))
        
        #Con funcion
        Q0f = calcula_Q(R,v)
        print("funcion: " + str(Q0f))
        
        # Separamos los valores de las comunidades
        v = np.where(v > 0, 1, -1)
        print("Separaciones: " + str(v))
        
        # Valor Modularidad
        #print(Q0)
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            print("Modularidad Adecuada")
            
            # Separamos y devolvemos los grupos
            Rp = [nombres_s[i] for i in range(len(v)) if v[i] == 1] # Parte de R asociada a los valores positivos de v
            Rm = [nombres_s[i] for i in range(len(v)) if v[i] == -1] # Parte asociada a los valores negativos de v
            
            return [Rp,Rm]
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = [nombres_s[i] for i in range(len(v)) if v[i] == 1] # Parte de R asociada a los valores positivos de v
            Rm = [nombres_s[i] for i in range(len(v)) if v[i] == -1] # Parte asociada a los valores negativos de v
            
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
            # Arreglamos los autovectores
            vp = np.where(vp > 0, 1, -1)
            vm = np.where(vm > 0, 1, -1)

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                # Creamos los subniveles
                A1 = A[np.ix_(v == 1, v == 1)]
                A2 = A[np.ix_(v == -1, v == -1)]
                
                # Ahora los analizamos
                return modularidad_iterativo(A1, Rp) +modularidad_iterativo(A2, Rm)

