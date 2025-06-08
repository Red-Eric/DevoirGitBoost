import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import words
import random
import string

##### devoir git 



nltk.download('words')

english_words = set(word.lower() for word in words.words())

def generate_nonsense_words(n, min_len=4, max_len=8):
    nonsense_words = set()
    alphabet = string.ascii_lowercase
    while len(nonsense_words) < n:
        length = random.randint(min_len, max_len)
        word = ''.join(random.choices(alphabet, k=length))
        if word not in english_words:
            nonsense_words.add(word)
    return list(nonsense_words)

def train_classifier():
    real_words = random.sample(list(english_words), 1000)
    nonsense_words = generate_nonsense_words(1000)

    words_list = real_words + nonsense_words
    labels = [1] * len(real_words) + [0] * len(nonsense_words) 
    df = pd.DataFrame({'word': words_list, 'label': labels})
    df.to_csv('word_dataset.csv', index=False)

    data = pd.read_csv('word_dataset.csv')
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
    X = vectorizer.fit_transform(data['word'])
    y = data['label']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create
    classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # Eval
    predictions = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

    return classifier, vectorizer

# Prediction
def detect_word(word, model, vectorizer):
    features = vectorizer.transform([word.lower()])
    prediction = model.predict(features)[0]
    return "Real word" if prediction == 1 else "Nonsense word"

#
if __name__ == '__main__':
    print("Training the model on real and nonsense words... XD")
    model, vectorizer = train_classifier()

    print("\nWord Detector is ready! Type a word to test, or 'quit' to exit.")
    while True:
        user_input = input("Enter a word: ").strip()
        if user_input.lower() == 'quit':
            print("kkkkkkkkkk")
            break
        if not user_input.isalpha():
            print("Invalid input. Please enter only letters.")
            continue
        result = detect_word(user_input, model, vectorizer)
        print(f"Result: {result}\n")
