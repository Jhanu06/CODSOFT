#Hello welcome this is Jhansi Madugula
#This is a task in ML internship provided by CODSOFT,this is task -1 movie generation (April-2024)!!

#The below code provides the predicted data in model_evalution file by providing the training data of movies along with bargraphs after prediction


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

try:
    with tqdm(total=50, desc="Loading Data") as pbar:
        data = pd.read_csv('C://Users//yaswa//OneDrive//Desktop//vscode//CodSoft//movie_plots.csv', header=None, names=['MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading data: {e}")

X = data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_tfidf)
    pbar.update(50)

def generate_movie_plot(model, vectorizer, seed_text, max_length=100):
    for _ in range(max_length):
        seed_vector = vectorizer.transform([seed_text])
        next_word_prob = model.predict_proba(seed_vector)[0]
        next_word = np.random.choice(model.classes_, p=next_word_prob)
        seed_text += " " + next_word
        if next_word == '<EOS>':
            break
    return seed_text

generated_plots = []
for _ in range(10):
    plot = generate_movie_plot(naive_bayes, tfidf_vectorizer, "start_token")
    generated_plots.append(plot)

for i, plot in enumerate(generated_plots, 1):
    print(f"Generated Plot {i}: {plot}")

with open("generated_plots.txt", "w", encoding="utf-8") as output_file:
    for plot in generated_plots:
        output_file.write(f"{plot}\n")
