import pandas as pd
import glob
import os

current_directory = os.getcwd()

path = current_directory + "/Resultados_ruidos"  # Asegúrate de que esta ruta es correcta
print("El directorio actual es:", path)

all_files = glob.glob(path + "/*.csv")

print(f"Archivos encontrados: {all_files}")  # Imprime los archivos encontrados

xxaa=input("¿El directorio es correcto:?")
li = []


for filename in all_files:
    print(f"Leyendo archivo: {filename}")  # Imprime el nombre del archivo que se está leyendo
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

print(f"DataFrames en la lista: {len(li)}")  # Imprime cuántos DataFrames se han añadido a la lista

if li:  # Solo intenta concatenar si la lista no está vacía
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv('Resultados_ruido_swap.csv', index=False)
else:
    print("No se encontraron archivos CSV para concatenar")
