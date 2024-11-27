import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config_ruta import ruta_datos 
from load import cargar_datos
def describe(datos):
    
    descripcion=datos.describe()
    return descripcion

descripcion=describe(cargar_datos(ruta_datos))
print(f'Breve descripción del datatset:\n{descripcion}')