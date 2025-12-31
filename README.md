# 🧠 IA Multimodale de Détection de Détresse Psychologique

Ce projet utilise le **Deep Learning** pour analyser la santé mentale à travers deux vecteurs : le texte (NLP) et l'audio (Prosodie). Actuellement, le module textuel est entièrement fonctionnel et calibré pour une utilisation de précision.

## 🚀 État du Projet : Module Textuel (Finalisé)

Le modèle textuel est conçu pour distinguer une émotion négative passagère (colère, nostalgie) d'une détresse psychologique réelle (anhédonie, risque suicidaire).

### 🛠️ Spécifications Techniques
- **Architecture :** Réseau de neurones récurrents **Bi-LSTM** (Bidirectional Long Short-Term Memory).
- **Embeddings :** Utilisation de **GloVe** (Global Vectors for Word Representation) pour une compréhension sémantique profonde.
- **Optimisation :** - Pondération des classes (**Class Weights**) à 0.4 pour la détresse afin de limiter les faux positifs.
  - Seuil de décision personnalisé à **0.75** pour garantir une haute confiance avant alerte.

### 📊 Performance et Validation
Le modèle a été validé sur des cas complexes pour tester sa résilience :
- **Nostalgie :** Identifié comme **SAIN** (Score 0.63).
- **Colère externe :** Identifié comme **SAIN** (Score 0.54).
- **Détresse/Vide émotionnel :** Identifié comme **DÉTRESSE** (Score 0.83).

## 📂 Structure du Dépôt
- `predict.py` : Script interactif pour tester l'IA en temps réel.
- `modeling.py` : Architecture du réseau de neurones.
- `model_mental_health_deep.h5` : Le modèle entraîné (cerveau de l'IA).
- `requirements.txt` : Liste des dépendances (TensorFlow, Scikit-learn, etc.).

> **Note sur les données :** Les fichiers de données brutes (CSV) et les vecteurs GloVe (.txt) ne sont pas inclus dans ce dépôt en raison de leur taille (800MB+). Le modèle `.h5` pré-entraîné est prêt à l'emploi.

