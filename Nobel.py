import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Premios Nobel")

st.image(
    "no_2.jpg",
    caption="Símbolo de los Premios Nobel basado en su creador, Alfred Nobel"
)

st.header("Clasificación de texto")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("df_nobel_final.csv", encoding="latin-1")

nobel = cargar_datos()

# Validar columnas necesarias
columnas_requeridas = {"Text", "Label"}

if not columnas_requeridas.issubset(nobel.columns):
    st.error("El archivo debe contener las columnas 'Text' y 'Label'.")
    st.stop()

# Eliminar datos vacíos
nobel = nobel.dropna(subset=["Text", "Label"])

X = nobel["Text"].astype(str)
y = nobel["Label"]

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Convertir texto a valores numéricos
vectorizador = TfidfVectorizer(
    lowercase=True,
    stop_words=None
)

X_train_vectorizado = vectorizador.fit_transform(X_train)
X_test_vectorizado = vectorizador.transform(X_test)

# Probar diferentes valores de K
resultados = {}

max_k = min(5, len(X_train))

for k in range(1, max_k + 1):
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train_vectorizado, y_train)

    predicciones = modelo.predict(X_test_vectorizado)
    accuracy = accuracy_score(y_test, predicciones)

    resultados[k] = accuracy

# Elegir el mejor valor de K
mejor_k = max(resultados, key=resultados.get)

modelo_final = KNeighborsClassifier(n_neighbors=mejor_k)
modelo_final.fit(X_train_vectorizado, y_train)

st.write(f"Mejor valor de K: {mejor_k}")
st.write(f"Exactitud del modelo: {resultados[mejor_k]:.2%}")

# Entrada del usuario
texto_usuario = st.text_input("Introduce el texto que deseas evaluar:")

if st.button("Predecir"):
    if texto_usuario.strip() == "":
        st.warning("Introduce un texto antes de realizar la predicción.")
    else:
        texto_vectorizado = vectorizador.transform([texto_usuario])
        prediccion = modelo_final.predict(texto_vectorizado)[0]

        nombres_premios = {
            0: "Physics",
            1: "Medicine",
            2: "Peace",
            3: "Literature",
            4: "Chemistry",
            5: "Economics"
        }

        # Si Label es numérico, usa el diccionario.
        # Si ya contiene texto, muestra directamente la etiqueta.
        if isinstance(prediccion, (int, float)):
            resultado = nombres_premios.get(
                int(prediccion),
                "Categoría desconocida"
            )
        else:
            resultado = str(prediccion)

        st.subheader("Predicción")
        st.success(resultado)

