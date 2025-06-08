## Nonsense Word Detector

Ce projet utilise un réseau de neurones simple pour déterminer si un mot est un mot anglais réel ou un mot inventé (non présent dans le dictionnaire).

---

### Prérequis

Installe les dépendances Python :

```bash
pip install pandas scikit-learn nltk
```

---

### 🚀 Lancer le projet

```bash
python main.py
```

> Le modèle s'entraîne automatiquement à chaque exécution.

---

### Exemple 

```
Training the model on real and nonsense words...
Accuracy: 95.10%

Word Detector is ready! Type a word to test, or 'quit' to exit.
Enter a word: house
Result: Real word

Enter a word: blarfit
Result: Nonsense word
```

---

### Remarques

* Les faux mots sont générés de manière aléatoire et peuvent parfois ressembler à de vrais mots.
* Ce projet peut être une bonne base pour entraîner un détecteur de mots valides dans n'importe quelle langue.
