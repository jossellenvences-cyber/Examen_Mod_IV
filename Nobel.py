import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Nobel Prizes")

st.image("no_2.jpg", caption="Nobel Prize symbol based on its creator, Alfred Nobel")

st.header("Text Classification")

@st.cache_data
def load_data():
    return pd.read_csv("df_nobel_final.csv", encoding="latin-1")

nobel = load_data()

columnas = {"Text", "Label"}

x = nobel["Text"].astype(str)
y = nobel["Label"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Test different K values
results = {}

max_k = min(5, len(x_train))

for k in range(1, max_k + 1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_vec, y_train)

    predictions = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    results[k] = accuracy

# Select the best K value
best_k = max(results, key=results.get)

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(x_train_vec, y_train)

#st.write(f"Best K value: {best_k}")
#st.write(f"Model accuracy: {results[best_k]:.2%}")

# User input
user_text = st.text_input("Enter the text you want to evaluate:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text before making a prediction.")
    else:
        user_text_vectorized = vectorizer.transform([user_text])
        prediction = final_model.predict(user_text_vectorized)[0]

        #prize_categories = {
            #0: "Physics",
            #1: "Medicine",
            #2: "Peace",
            #3: "Literature",
            #4: "Chemistry",
            #5: "Economics"

             prediction == 0:
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

        }

        # If labels are numeric, use the dictionary.
        # If labels are already text, display them directly.
        if isinstance(prediction, (int, float)):
            result = prize_categories.get(
                int(prediction),
                "Unknown category"
            )
        else:
            result = str(prediction)

        st.subheader("Prediction")
        st.success(result)




        
