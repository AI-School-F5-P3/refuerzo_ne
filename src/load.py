import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_ruta import ruta_datos  # Importa la ruta desde archivo_config.py

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

datos = cargar_datos(ruta_datos)

# Si los datos fueron cargados correctamente, puedes procesarlos aquí
if datos is not None:
    # Por ejemplo, mostrar las primeras filas del dataframe
    print(datos.head())