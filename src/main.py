
import pandas as pd
from generador_informe import generar_informe_correlacion
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generador_informe import generar_informe_correlacion
from config_ruta import ruta_datos 
from load import cargar_datos


# Uso
def main():
    # Cargar datos
    df = pd.DataFrame(cargar_datos(ruta_datos))
    
    # Generar informe
    generar_informe_correlacion(df)

if __name__ == "__main__":
    main()