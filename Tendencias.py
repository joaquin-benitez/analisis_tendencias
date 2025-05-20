import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Objetivo del proyecto
# Analizar las tendencias de compra en función de diferentes variables como categoría, género, temporada y método de pago,
# con el fin de obtener información útil para estrategias comerciales y optimización de ventas.

# Cargar el archivo CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Exploración inicial
def explore_data(df):
    print("\nPrimeras filas del dataset:")
    print(df.head())
    print("\nInformación general:")
    print(df.info())
    print("\nValores nulos:")
    print(df.isnull().sum())
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    print("\nTipos de datos:")
    print(df.dtypes)

def clean_data(df):
    # Eliminar duplicados si existen
    df = df.drop_duplicates()
    
    # Normalizar nombres de columnas (eliminar espacios y usar minúsculas)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Verificar valores nulos y decidir su tratamiento
    df = df.fillna(method='ffill')  # Llenar valores nulos con el anterior válido
    
    return df    



# Análisis exploratorio
def exploratory_analysis(df):
    # Gráfico de distribución de montos de compra
    plt.figure(figsize=(8, 5))
    sns.histplot(df['purchase_amount_(usd)'], bins=30, kde=True, color='blue')
    plt.title("Distribución de Monto de Compra")
    plt.xlabel("Monto en USD")
    plt.ylabel("Frecuencia")
    plt.show()
    
    # Gráfico de compras por categoría
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='category', order=df['category'].value_counts().index, palette='viridis')
    plt.title("Cantidad de Compras por Categoría")
    plt.xlabel("Categoría")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=45)
    plt.show()
    
    # Gráfico de compras por género
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='gender', palette='coolwarm')
    plt.title("Distribución de Compras por Género")
    plt.xlabel("Género")
    plt.ylabel("Cantidad de Compras")
    plt.show()
    
    # Relación entre temporada y monto de compra
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='season', y='purchase_amount_(usd)', palette='pastel')
    plt.title("Distribución del Monto de Compra por Temporada")
    plt.xlabel("Temporada")
    plt.ylabel("Monto de Compra (USD)")
    plt.show()
    
    # Método de pago más utilizado
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='payment_method', order=df['payment_method'].value_counts().index, palette='magma')
    plt.title("Métodos de Pago Más Utilizados")
    plt.xlabel("Método de Pago")
    plt.ylabel("Cantidad de Compras")
    plt.xticks(rotation=45)
    plt.show()

# Generación de conclusiones
def generate_conclusions(df):
    print("\nConclusiones del Análisis:")

    # Categoría de productos más comprada
    top_category = df['category'].value_counts().idxmax()
    print(f"1. La categoría de productos más comprada es '{top_category}', indicando una alta demanda en ese segmento.")

    # Género con más compras
    top_gender = df['gender'].value_counts().idxmax()
    print(f"2. El género que realiza más compras es '{top_gender}', lo que sugiere estrategias de marketing específicas.")

    # Temporada con mayor gasto promedio
    season_spending = df.groupby('season')['purchase_amount_(usd)'].mean()
    top_season = season_spending.idxmax()
    print(f"3. La temporada con el mayor gasto promedio es '{top_season}', lo que puede ayudar a planificar promociones en esas fechas.")

    # Método de pago más utilizado
    top_payment_method = df['payment_method'].value_counts().idxmax()
    print(f"4. El método de pago más utilizado es '{top_payment_method}', lo que sugiere priorizar su disponibilidad en plataformas de venta.")

    # Rango de montos de compra
    min_purchase = df['purchase_amount_(usd)'].min()
    max_purchase = df['purchase_amount_(usd)'].max()
    print(f"5. La distribución de montos de compra muestra que la mayoría de las compras están en un rango de ${min_purchase:.2f} a ${max_purchase:.2f} USD.")

# Archivo de datos
data_file = "c:/Users/Admin/Desktop/Programacion/data2/shopping_trends.csv"
df = load_data(data_file)
explore_data(df)
df = clean_data(df) 
print(df)    
exploratory_analysis(df)
generate_conclusions(df)


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Variables relevantes para segmentar clientes
features = df[['gender', 'season', 'payment_method', 'category', 'purchase_amount_(usd)']]

# 2. Codificación de variables categóricas
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(features[['gender', 'season', 'payment_method', 'category']])

# Combinar variables codificadas con el monto de compra
import numpy as np
X = np.concatenate([encoded, features[['purchase_amount_(usd)']].values], axis=1)

# 3. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means con número de clusters = 4 (podés ajustarlo)
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Análisis por cluster
print("\nCantidad de clientes por cluster:")
print(df['cluster'].value_counts())

print("\nGasto promedio por cluster:")
print(df.groupby('cluster')['purchase_amount_(usd)'].mean())

# 6. Visualización con reducción de dimensiones (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['cluster'], palette='Set2')
plt.title("Segmentación de Clientes (PCA + K-Means)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n=== Clasificación: Ajuste de hiperparámetros para Código Promocional ===")

# Variables predictoras y objetivo
X_class = df[['age', 'gender', 'season', 'payment_method', 'category', 'preferred_payment_method']]
y_class = df['promo_code_used'].map({'Yes': 1, 'No': 0})

# Codificación variables categóricas
X_class_encoded = pd.get_dummies(X_class, drop_first=True)

# División entrenamiento/test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X_class_encoded, y_class, test_size=0.3, random_state=42, stratify=y_class
)

# Definición del modelo base
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Grid de parámetros para explorar
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Búsqueda con validación cruzada
grid_search = GridSearchCV(estimator=clf,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

# Entrenamos
grid_search.fit(X_train, y_train)

print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor accuracy en CV: {grid_search.best_score_:.4f}")

# Evaluación con los mejores parámetros en test
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

print("\nAccuracy en test:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Visualización del árbol
plt.figure(figsize=(20, 10))
plot_tree(best_clf, feature_names=X_class_encoded.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Árbol de Decisión Ajustado - ¿Usará Código Promocional?")
plt.show()

