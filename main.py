# main.py

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

# Calcular la media y desviación estándar del precio de venta
mean_price = y_train.mean()
std_price = y_train.std()

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

    estimacion = prediction[0]
    desviaciones = (estimacion - mean_price) / std_price

    st.subheader('Precio de venta estimado:')
    st.write(f'${estimacion:,.2f}')

    # Mostrar el rango normal de precio
    st.write(f"El rango normal de precio de este vehículo es entre ${round(estimacion - std_price, 2):,.2f} y ${round(estimacion + std_price, 2):,.2f}")

    # Mostrar la recomendación basada en desviaciones
    if desviaciones > 3:
        st.write("🚫 El automóvil está **muy por encima** de su valor de mercado; se recomienda **no comprar**.")
    elif desviaciones > 2:
        st.write("⚠️ El automóvil está **significativamente por encima** de su valor de mercado; se recomienda **no comprar**.")
    elif desviaciones > 1:
        st.write("ℹ️ El automóvil se encuentra **por encima** de su valor de mercado; se recomienda **negociar el precio**.")
    elif desviaciones > -1:
        st.write("✅ El automóvil se encuentra en el **precio promedio** de mercado; comprar el vehículo observando su condición.")
    elif desviaciones > -2:
        st.write("👍 El automóvil se encuentra **por debajo** de su precio de mercado; se recomienda **comprar el vehículo**.")
    else:
        st.write("🏆 El precio del automóvil se encuentra **significativamente por debajo** del precio del mercado; se recomienda comprarlo posteriormente a una **inspección técnica**.")
