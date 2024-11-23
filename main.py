import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datos
df = pd.read_csv('./car_prices.csv')

# Preprocesamiento de datos
categorical_columns = ['transmission', 'color', 'body', 'interior']
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

df = pd.concat([df[numerical_columns], df[categorical_columns]], axis=1)

# Manejo de valores faltantes
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].mean())

# Codificación de variables categóricas
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Separar características y variable objetivo
X = df_encoded.drop(columns=['sellingprice'])
y = df_encoded['sellingprice']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar el modelo
modelo_Reglineal = LinearRegression()
modelo_Reglineal.fit(X_train, y_train)

# Aplicación de Streamlit
st.title('Predicción de Precios de Autos')

st.header('Ingrese los parámetros del vehículo')

# Obtener valores únicos para las variables categóricas
unique_values = {}
for col in categorical_columns:
    unique_values[col] = df[col].unique().tolist()

# Crear entradas para las variables numéricas
input_data = {}
for col in numerical_columns:
    if col != 'sellingprice':
        input_data[col] = st.number_input(f'Ingrese {col}', value=float(df[col].mean()))

# Crear menús desplegables para las variables categóricas
for col in categorical_columns:
    input_data[col] = st.selectbox(f'Seleccione {col}', unique_values[col])

# Botón para realizar la predicción
if st.button('Predecir'):
    # Crear un DataFrame con los datos de entrada
    input_df = pd.DataFrame([input_data])

    # Codificar las variables categóricas de la entrada
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

    # Alinear las columnas con las del conjunto de entrenamiento
    input_df_encoded = input_df_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Realizar la predicción
    prediction = modelo_Reglineal.predict(input_df_encoded)

    st.subheader('Precio de venta estimado:')
    st.write(f'${prediction[0]:,.2f}')
