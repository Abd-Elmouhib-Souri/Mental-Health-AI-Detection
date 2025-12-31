import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight  # <-- NOUVEAU

def train_and_evaluate(df):
    print("\n--- 🏆 Étape 3 : Deep Learning ELITE avec Intelligence GloVe ---")
    
    # 1. Paramètres de configuration
    MAX_WORDS = 20000
    MAX_LEN = 100
    EMBEDDING_DIM = 100

    # 2. Tokenisation
    print("🔡 Tokenisation du texte...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(df['cleaned_text'].astype(str))
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'].astype(str))
    X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    y = df['label'].values

    # 3. SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("⚖️ Application du SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 4. GLOVE
    print("📚 Chargement des vecteurs GloVe...")
    embeddings_index = {}
    with open('glove.6B.100d.txt', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i < MAX_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # 5. ARCHITECTURE
    print("🏗️ Construction du cerveau textuel...")
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 6. CALCUL DES POIDS DE CLASSE (POUR CALMER L'IA)
    # On donne 1.0 au "Sain" et seulement 0.4 à la "Détresse"
    # Cela oblige l'IA à être sûre à 250% avant de mettre un score élevé.
    custom_weights = {0: 1.0, 1: 0.4}
    print(f"⚖️ Application des poids de classe : {custom_weights}")

    # 7. ENTRAÎNEMENT
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("🏋️ Début de l'entraînement intensif...")
    model.fit(
        X_train_res, 
        y_train_res, 
        epochs=15, 
        batch_size=256, 
        validation_split=0.1, 
        callbacks=[es],
        class_weight=custom_weights  # <-- AJOUTÉ ICI
    )

    # 8. SAUVEGARDE
    print("💾 Sauvegarde du modèle ELITE...")
    model.save('model_mental_health_deep.h5')
    joblib.dump(tokenizer, 'tokenizer_deep.pkl')
    
    return model, tokenizer