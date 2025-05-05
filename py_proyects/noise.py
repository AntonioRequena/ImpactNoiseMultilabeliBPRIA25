
import random  # Para usar random.sample
#import numpy as np  # Para manejar arrays si y_train es un array de numpy
from scipy.sparse import issparse  # Para manejar matrices dispersas
import numpy as np

def noise_swap(y_train, noise_perc):
    print('Entrando en swap...')
    #input('Pulsa...')
    # Number of elements (even number):
    number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))
    
    random.seed(41)                             # For reproducibility
    np.random.seed(41)
    # Selecting elements:
    list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)
    #print('list_elements:', list_elements)
    #input('Pulsa una tecla')
    
    
    # Creating output vector:
    y_out = y_train.todense().copy() if issparse(y_train) else y_train.copy()

    # Iterating through the selected pairs:
    for it in range(int(len(list_elements)/2)):
        temp = y_out[list_elements[it]].copy()
     
        #print('y_out[list_element[it]]', y_out[list_elements[it]] )
        #print('y_out[list_element[len(list_elements)-1-it]]', y_out[list_elements[len(list_elements)-1-it]])
        #for el in range(int(len(y_out[list_elements[it]]))):
        #    print('Elemento %d --> %d' % (el, y_out[list_elements[it]][el]))
        #resultado = input('Pulsa una tecla ')
     
        y_out[list_elements[it]] = y_out[list_elements[len(list_elements)-1-it]].copy()
        y_out[list_elements[len(list_elements)-1-it]] = temp.copy()

        #print('y_out[list_element[it]]', y_out[list_elements[it]] )
        #print('y_out[list_element[len(list_elements)-1-it]]', y_out[list_elements[len(list_elements)-1-it]])
        #resultado = input('Pulsa una tecla ')
     
    return y_out



def noise_PUMN(y_train, noise_perc, umbral):
    print('Entrando en PUMN...')
    #input('Pulsa...')

    # Number of elements (even number):
    number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))
    # Supongamos que y_train es una matriz binaria de etiquetas (ejemplos x etiquetas)
    r = umbral  # Umbral de probabilidad
    
    # Copia de los datos originales
    y_train_noisy = y_train.todense().copy() if issparse(y_train) else y_train.copy()
    
    
    random.seed(41)                             # For reproducibility
    np.random.seed(41)
    # Seleccion de elementos:
    list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)
    
    # Aplicación de ruido PUMN
    for i in range(int(len(list_elements))):  # Para cada instancia de la lista de elementos (no del conjunto completo)
    
        #impresion antes del cambio
        #print('Etiquetas ant. del cambio', y_train_noisy[list_elements[i]])

        for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
            p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
            if p > r:  # Condición para aplicar el cambio
                y_train_noisy[list_elements[i], j] = 1 - y_train_noisy[list_elements[i], j]  # Cambia de 0 a 1 o de 1 a 0

        #impresion despues del cambio
        #print('Etiquetas des. del cambio', y_train_noisy[list_elements[i]])
        #input('Pulse una tecla ')

    
    return y_train_noisy
    
    


def noise_add(y_train, noise_perc, umbral):
    print('Entrando en add...')
    #input('Pulsa...')

    # Number of elements (even number):
    number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))
    # Supongamos que y_train es una matriz binaria de etiquetas (ejemplos x etiquetas)
    r = umbral  # Umbral de probabilidad
    
    # Copia de los datos originales
    y_train_noisy = y_train.todense().copy() if issparse(y_train) else y_train.copy()
    
    
    random.seed(41)                             # For reproducibility
    np.random.seed(41)
    # Seleccion de elementos:
    list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)
        
    # Aplicación de ruido Aditivo
    for i in range(int(len(list_elements))):  # Para cada instancia de la lista de elementos (no del conjunto completo)
    
        #impresion antes del cambio
        #print('Etiquetas ant. del cambio', y_train_noisy[list_elements[i]])

        for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
            p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
            if p > r:  # Condición para aplicar el cambio
                if y_train_noisy[list_elements[i], j] == 0:
                    y_train_noisy[list_elements[i], j] = 1 # Cambia de 0 a 1 si la etiqueta es 0

        #impresion despues del cambio
        #print('Etiquetas des. del cambio', y_train_noisy[list_elements[i]])
        #input('Pulse una tecla ')

    
    return y_train_noisy
    

def noise_sub(y_train, noise_perc, umbral):
    print('Entrando en sub...')
    #input('Pulsa...')

    # Number of elements (even number):
    number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))
    # Supongamos que y_train es una matriz binaria de etiquetas (ejemplos x etiquetas)
    r = umbral  # Umbral de probabilidad
    
    # Copia de los datos originales
    y_train_noisy = y_train.todense().copy() if issparse(y_train) else y_train.copy()
    
    
    random.seed(41)                             # For reproducibility
    np.random.seed(41)
    # Seleccion de elementos:
    list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)
        
    # Aplicación de ruido Sustractivo
    for i in range(int(len(list_elements))):  # Para cada instancia de la lista de elementos (no del conjunto completo)
    
        #impresion antes del cambio
        #print('Etiquetas ant. del cambio', y_train_noisy[list_elements[i]])

        for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
            p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
            if p > r:  # Condición para aplicar el cambio
                if y_train_noisy[list_elements[i], j] == 1:
                    y_train_noisy[list_elements[i], j] = 0 # Cambia de 1 a 0 si la etiqueta es 1

        #impresion despues del cambio
        #print('Etiquetas des. del cambio', y_train_noisy[list_elements[i]])
        #input('Pulse una tecla ')

    
    return y_train_noisy


def noise_add_sub(y_train, noise_perc, umbral):
    print('Entrando en add-sub...')
    #input('Pulsa...')

    # Number of elements (even number):
    number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))
    # Supongamos que y_train es una matriz binaria de etiquetas (ejemplos x etiquetas)
    r = umbral  # Umbral de probabilidad
    
    # Copia de los datos originales
    y_train_noisy = y_train.todense().copy() if issparse(y_train) else y_train.copy()
    
    
    random.seed(41)                             # For reproducibility
    np.random.seed(41)
    # Seleccion de elementos:
    list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)
        
    # Aplicación de ruido Aditivo
    for i in range(int(len(list_elements))):  # Para cada instancia de la lista de elementos (no del conjunto completo)
    
        #impresion antes del cambio
        #print('Etiquetas ant. del cambio', y_train_noisy[list_elements[i]])

        for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
            p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
            if p > r:  # Condición para aplicar el cambio
                if y_train_noisy[list_elements[i], j] == 0:
                    y_train_noisy[list_elements[i], j] = 1 # Cambia de 0 a 1 si la etiqueta es 0

        #impresion despues del cambio
        #print('Etiquetas des. del cambio', y_train_noisy[list_elements[i]])
        #input('Pulse una tecla ')

 # Aplicación de ruido Sustractivo
    for i in range(int(len(list_elements))):  # Para cada instancia de la lista de elementos (no del conjunto completo)
    
        #impresion antes del cambio
        #print('Etiquetas ant. del cambio', y_train_noisy[list_elements[i]])

        for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
            p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
            if p > r:  # Condición para aplicar el cambio
                if y_train_noisy[list_elements[i], j] == 1:
                    y_train_noisy[list_elements[i], j] = 0 

        #impresion despues del cambio
        #print('Etiquetas des. del cambio', y_train_noisy[list_elements[i]])
        #input('Pulse una tecla ')


    
    return y_train_noisy


def noise_DAAS(y_train, noise_perc, umbral):
    print('Entrando en DAAS...')
    #input('Pulsa...')

    # Number of elements (even number):
    number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))
    # Supongamos que y_train es una matriz binaria de etiquetas (ejemplos x etiquetas)
    r = umbral  # Umbral de probabilidad
    
    # Copia de los datos originales
    y_train_noisy = y_train.todense().copy() if issparse(y_train) else y_train.copy()
    
    
    random.seed(41)                             # For reproducibility
    np.random.seed(41)
    # Seleccion de elementos:
    list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)
        
    # Aplicación de ruido Aditivo
    for i in range(int(len(list_elements))):  # Para cada instancia de la lista de elementos (no del conjunto completo)
    
        #impresion antes del cambio
        #print('Etiquetas antes del cambio', y_train_noisy[list_elements[i]])
        cambios_add = 0
        for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
            p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
            if p > r:  # Condición para aplicar el cambio
                if y_train_noisy[list_elements[i], j] == 0:
                    cambios_add+=1
                    y_train_noisy[list_elements[i], j] = 1 # Cambia de 0 a 1 si la etiqueta es 0
        #print('Cambios de 0 a 1: ', cambios_add)

        #impresion despues del cambio
        #print('Etiquetas despues add:    ', y_train_noisy[list_elements[i]])
        
        # Aplicación de ruido Sustractivo para igualar el numero de etiquetas modificadas

        while cambios_add > 0:
            for j in range(y_train_noisy[list_elements].shape[1]):  # Para cada etiqueta en la instancia
                p = np.random.uniform(0, 1)  # Genera una muestra de la distribución uniforme          
                if (p > r) & (cambios_add > 0):  # Condición para aplicar el cambio
                    if y_train_noisy[list_elements[i], j] == 1:
                        cambios_add-=1
                        y_train_noisy[list_elements[i], j] = 0 

            #impresion despues del cambio
        #   print('Etiquetas despues sub     ', y_train_noisy[list_elements[i]])
            
        #print('\n')
        #print('Etiquetas definitiva      ', y_train_noisy[list_elements[i]])
        #input('Pulse una tecla ')
    
    return y_train_noisy
    