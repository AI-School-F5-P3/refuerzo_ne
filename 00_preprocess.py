import pandas as pd
import os
import matplotlib.pyplot as plt
print('Hola')

ruta="C:/4_F5/021_ML_refuerzo/refuerzo_ne/data/raw/teleCust1000t.csv"

# Carga y preparación de datos con manejo de errores
def cargar_datos(ruta):
    """
    Cargar datos desde un archivo CSV y realizar preprocesamiento inicial.
    
    Parámetros:
    ruta (str): Ruta del archivo CSV a cargar.

    Retorna:
    pd.DataFrame: El DataFrame cargado si se realiza correctamente.
    None: Si ocurre algún error durante la carga.
    """
    try:
        # Intentar cargar los datos
        datos = pd.read_csv(ruta, sep=',')
        print("Datos cargados exitosamente.")
        return datos
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta especificada: {ruta}")
    except pd.errors.EmptyDataError:
        print("Error: El archivo está vacío.")
    except pd.errors.ParserError:
        print("Error: Ocurrió un problema al analizar el archivo. Verifica el formato del CSV.")
    except Exception as e:
        print(f"Error inesperado: {e}")
    
    # En caso de error, devuelve None
    return None

# Ejemplo de uso
datos = pd.DataFrame(cargar_datos(ruta))



#Analiza los datos-------------------------------------------------------------------------

def describe(datos):
    
    descripcion=datos.describe()
    return descripcion

descripcion=describe(datos)
print(f'Breve descripción del datatset:\n{descripcion}')



print(f'Informacion datatset:\n{datos.info()}')

#Contar las filas
def obtener_columnas(datos):
    """
    Devuelve el número de columnas y los nombres de las columnas de un DataFrame.

    Parámetros:
    datos (pd.DataFrame): El DataFrame que contiene los datos.

    Retorna:
    tuple: Un tupla con el número de columnas y una lista de nombres de las columnas.
    """
    # Verifica si el DataFrame está vacío
    if datos.empty:
        print("El DataFrame está vacío.")
        return None, None

    # Obtener el número de columnas y los nombres de las columnas
    num_columnas = datos.shape[1]  # .shape[1] devuelve el número de columnas
    nombres_columnas = datos.columns.tolist()  # .columns devuelve un índice con los nombres de las columnas
    num_filas=datos.shape[0]
    
    return num_columnas, nombres_columnas, num_filas



# Llamamos a la función con los datos cargados
num_columnas, nombres_columnas, num_filas = obtener_columnas(datos)
if num_columnas is not None:
    print(f"Número de columnas: {num_columnas}")
    print(f"Nombres de las columnas: {nombres_columnas}")
    print(f'El número de filas es:{num_filas}')
else:
    print("No se pudo obtener la información de las columnas.")





def contar_valores_por_columna(datos):
    """
    Cuenta la cantidad de valores no nulos por cada columna en un DataFrame.

    Parámetros:
    datos (pd.DataFrame): El DataFrame que contiene los datos.

    Retorna:
    pd.Series: Una serie con los nombres de las columnas como índice y la cantidad de valores no nulos como valores.
    """
    # Verifica si el DataFrame está vacío
    if datos.empty:
        print("El DataFrame está vacío.")
        return None

    # Contar valores no nulos por columna
    conteo = datos.count()
    
    return conteo

# Llamar a la función y mostrar los resultados
conteo_valores = contar_valores_por_columna(datos)
if conteo_valores is not None:
    print("Conteo de valores por columna:")
    print(conteo_valores)
else:
    print("No se pudo obtener el conteo de valores.")







def contar_frecuencias_todas_las_columnas(datos):
    """
    Cuenta la frecuencia de cada valor único en todas las columnas de un DataFrame.

    Parámetros:
    datos (pd.DataFrame): El DataFrame que contiene los datos.

    Retorna:
    dict: Un diccionario donde las claves son los nombres de las columnas y los valores
          son Series de pandas con las frecuencias de los valores únicos.
    """
    # Diccionario para almacenar frecuencias de cada columna
    frecuencias_por_columna = {}

    # Iterar por todas las columnas del DataFrame
    for columna in datos.columns:
        if columna!= 'income':
            frecuencias_por_columna[columna] = datos[columna].value_counts()

    return frecuencias_por_columna


frecuencias_todas = contar_frecuencias_todas_las_columnas(datos)

# Mostrar los resultados
for columna, frecuencias in frecuencias_todas.items():
    print(f"Frecuencias en la columna '{columna}':")
    print(frecuencias)
    print("-" * 30)





def plot_frecuencias_todas_las_columnas(datos, carpeta="graficos_frecuencias"):
    """
    Genera gráficos de barras con las frecuencias de los valores únicos
    para todas las columnas de un DataFrame.

    Parámetros:
    datos (pd.DataFrame): El DataFrame que contiene los datos.
    """
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    for columna in datos.columns:
        # Contar las frecuencias de los valores únicos
        if columna!= 'income':
            frecuencias = datos[columna].value_counts()

            # Crear el gráfico de barras
            plt.figure(figsize=(8, 5))
            frecuencias.plot(kind='bar', color='skyblue', edgecolor='black')

            # Personalizar el gráfico
            plt.title(f"Frecuencias en la columna '{columna}'", fontsize=14)
            plt.xlabel("Valores únicos", fontsize=12)
            plt.ylabel("Frecuencia", fontsize=12)
            plt.xticks(rotation=90, fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    
            # Guardar el gráfico en la carpeta
            archivo = os.path.join(carpeta, f"{columna}_frecuencias.png")
            plt.tight_layout()
            plt.savefig(archivo, dpi=300)
            plt.close()  # Cerrar el gráfico para evitar sobrecarga de memoria
            
            print(f"Gráfico guardado: {archivo}")

# Llamar a la función para generar los gráficos
plot_frecuencias_todas_las_columnas(datos)



#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport #para Python 3.12
# Generar un reporte

def generar_reporte_exploratorio(datos, ruta_salida="reporte_customer.html", carpeta="reportes"):
    """
    Genera un reporte exploratorio de los datos y lo guarda como un archivo HTML.

    Parámetros:
    - datos (pd.DataFrame): El DataFrame con los datos a analizar.
    - ruta_salida (str): El nombre del archivo HTML donde se guardará el reporte.
    - carpeta (str): La carpeta donde se guardará el reporte. Si no existe, se creará.
    """
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Crear el reporte exploratorio
    profile_hate = ProfileReport(datos, title="Reporte Exploratorio")

    # Guardar el reporte en la carpeta especificada
    ruta_completa = os.path.join(carpeta, ruta_salida)
    profile_hate.to_file(ruta_completa)
    
    print(f"Reporte generado y guardado en: {ruta_completa}")

# Ejemplo de uso:
# Asumiendo que `datos` es tu DataFrame con los datos cargados
# ruta_salida es opcional, por defecto es "reporte_customer.html"
generar_reporte_exploratorio(datos)




# Limpieza de datos
if datos is not None:
    print("Llamando a la función clean_data()...")
    # Aquí llamas a la función clean_data() con los datos cargados
    # clean_data(datos)
else:
    print("No se pudieron cargar los datos. Revisa los errores y corrige.")



