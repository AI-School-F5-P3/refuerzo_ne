import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config_ruta import ruta_datos 
from load import cargar_datos

df=cargar_datos(ruta_datos)
# 1. Pearson Correlation Matrix
# Compute correlation matrix for numeric features
correlation_matrix = df.select_dtypes(include=[np.number]).corr(method='pearson')

# 2. Visualize Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,  # Show numerical values
            cmap='coolwarm',  # Color palette
            center=0,  # Center color scale at 0
            vmin=-1, 
            vmax=1,
            square=True)
plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
plt.savefig('C:/4_F5/021_ML_refuerzo/refuerzo_ne/reportes/matriz.png', dpi=300)
plt.show()

# 3. Detailed Correlation Analysis
print("Correlation with Target Variable (custcat):")
correlations_with_target = correlation_matrix['custcat'].sort_values(ascending=False)
print(correlations_with_target)

# 4. Identify Highly Correlated Features
def get_redundant_pairs(df):
    '''Get diagonal and duplicate pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_correlations(correlation_matrix, n=5):
    '''Get top n correlations'''
    au_corr = correlation_matrix.unstack()
    labels_to_drop = get_redundant_pairs(correlation_matrix)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# Print top positive and negative correlations
print("\nTop Positive Correlations:")
print(get_top_correlations(correlation_matrix))

print("\nTop Negative Correlations:")
print(get_top_correlations(-correlation_matrix))

# 5. Statistical Significance of Correlations
from scipy import stats

def correlation_significance(df):
    '''Calculate p-values for correlations'''
    corr_matrix = df.corr()
    p_matrix = np.zeros_like(corr_matrix)
    
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i != j:
                _, p_value = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                p_matrix[i, j] = p_value
    
    return p_matrix

# Calculate and print statistically significant correlations
p_values = correlation_significance(df.select_dtypes(include=[np.number]))
significant_correlations = (p_values < 0.05)


# Crear una máscara para correlaciones significativas
significant_mask = (p_values < 0.05)

# Crear la visualización
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            mask=~significant_mask,  # Invertir la máscara para mostrar solo correlaciones significativas
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Correlaciones Estadísticamente Significativas (p < 0.05)')
plt.tight_layout()

# Guardar la figura
plt.savefig('correlation_significance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Imprimir correlaciones significativas
print("\nCorrelaciones Estadísticamente Significativas:")
significant_corr = df.select_dtypes(include=[np.number]).corr()[significant_mask]
print(significant_corr)
print("\nStatistically Significant Correlations:")
print(significant_correlations)