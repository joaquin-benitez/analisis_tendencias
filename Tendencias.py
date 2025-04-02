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


