import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from scipy import stats
import pprint






def generar_informe_correlacion(df):
    """
    Función principal para generar informe de correlación
    """
    # 1. Matriz de Correlación de Pearson
    correlation_matrix = df.select_dtypes(include=[np.number]).corr(method='pearson')
    
    # 2. Gráfico de Mapa de Calor de Correlación
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                vmin=-1, 
                vmax=1, 
                square=True)
    plt.title('Mapa de Calor de Correlación de Pearson')
    plt.tight_layout()
    plt.savefig('matriz_correlacion.png', dpi=300)
    plt.close()

    # 3. Análisis de Correlaciones
    def get_top_correlations(correlation_matrix, n=5):
        """Obtener top correlaciones"""
        # Eliminar diagonal y duplicados
        au_corr = correlation_matrix.unstack()
        labels_to_drop = set((correlation_matrix.index[i], correlation_matrix.columns[j]) 
                              for i in range(len(correlation_matrix.index)) 
                              for j in range(i+1))
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    # 4. Correlaciones Estadísticamente Significativas
    def correlation_significance(df):
        """Calcular valores p para correlaciones"""
        corr_matrix = df.corr()
        p_matrix = np.zeros_like(corr_matrix, dtype=bool)
        
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if i != j:
                    _, p_value = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                    p_matrix[i, j] = p_value < 0.05
        
        return p_matrix

    # Calcular valores p
    
    significant_mask = correlation_significance(df.select_dtypes(include=[np.number]))


    # Generar correlaciones significativas como un diccionario plano
    correlaciones_significativas = {}
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if significant_mask[i, j] and i != j:
                key = f"{correlation_matrix.index[i]} vs {correlation_matrix.columns[j]}"
                correlaciones_significativas[key] = correlation_matrix.iloc[i, j]

    context = {
        'titulo': 'Informe de Correlación de Variables',
        'top_correlaciones_positivas': get_top_correlations(correlation_matrix).to_dict(),
        'top_correlaciones_negativas': get_top_correlations(-correlation_matrix).to_dict(),
        'correlaciones_significativas': correlaciones_significativas,
        'matriz_significancia': significant_mask.tolist(),
        'nombres_columnas': df.select_dtypes(include=[np.number]).columns.tolist(),
        'imagen_matriz': 'matriz_correlacion.png',
        'enumerate': enumerate  # Pasar la función enumerate al contexto
    }

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template_correlacion.html')

    with open('informe_correlacion.html', 'w') as f:
        f.write(template.render(context))

    print("Informe de correlación generado exitosamente.")
    pprint.pprint(context)