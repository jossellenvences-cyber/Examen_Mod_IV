import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier


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

vect = CountVectorizer()
x_dtm = vect.fit_transform(x)

nb = DecisionTreeClassifier()
nb.fit(x_dtm, y)

#df_dtm = vect.transform(df['Text'])
#prediction = nb.predict(df_dtm)

classifier = DecisionTreeClassifier(max_depth=8, criterion='texto_limpio', min_samples_leaf=10, max_features=7, random_state=0)
classifier.fit(x, y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('Physics')
elif prediction == 1:
  st.write('Medicine')
elif prediction == 2:
  st.write('Peace')
elif prediction == 3:
  st.write('Literature')
elif prediction == 4:
  st.write('Chemistry')
elif prediction == 5:
  st.write('Economics')
else:
  st.write('Sin Predicción')
