import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



st.write(''' # Premios Nobel ''')
st.image("no_2.jpg", caption="Simbolo de Premios Nobel en base a su creador Alfred Nobel")

st.header('Texto')

def user_input_features():
  # Entrada
  texto_limpio = st.text_input("Introduce el texto a evaluar")

  user_input_data = {'Text': texto_limpio}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df_nob = user_input_features()

nobel =  pd.read_csv('df_nobel_final.csv', encoding='latin-1')
x = nobel.Text
y = nobel.Label

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

k_list = range(1,6,1)

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    modelo_knn = knn.fit(x_train, y_train)

    y_pred = modelo_knn.predict(x_test)
  

    mse = mean_squared_error(y_test, y_pred)
    
    knn_dict[k] = mse
    

#vect = CountVectorizer()
#x_dtm = vect.fit_transform(x)

#nb = MultinomialNB()
#nb.fit(x_dtm, y)

#df_dtm = vect.transform(df['Text'])

#prediction = nb.predict(df_dtm)


st.subheader('Predicción')
if y_pred == 0:
  st.write('Physics')
elif y_pred == 1:
  st.write('Medicine')
elif y_pred == 2:
  st.write('Peace')
elif y_pred == 3:
  st.write('Literature')
elif y_pred == 4:
  st.write('Chemistry')
elif y_pred == 5:
  st.write('Economics')
else:
  st.write('Sin Predicción')
