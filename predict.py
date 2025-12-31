import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess_text import clean_text

def predict_tool():
    # 1. Chargement avec gestion d'erreurs
    print("🧠 Chargement du modèle Deep Learning ELITE...")
    try:
        model = tf.keras.models.load_model('model_mental_health_deep.h5')
        tokenizer = joblib.load('tokenizer_deep.pkl')
    except Exception as e:
        print(f"❌ Erreur : Impossible de charger les fichiers. {e}")
        return

    print("\n--- 🤖 IA DE SANTÉ MENTALE PRÊTE ---")
    print("(Tapez 'quitter' pour sortir)")

    while True:
        user_input = input("\n✍️  Entrez un texte à analyser : ")
        
        if user_input.lower() == 'quitter':
            break

        # 2. Prétraitement rigoureux
        cleaned = clean_text(user_input)
        
        # Transformation en séquence
        seq = tokenizer.texts_to_sequences([cleaned])
        
        # Padding (doit être identique à l'entraînement : 100)
        padded = pad_sequences(seq, maxlen=100)

        # 3. Prédiction
        prediction = model.predict(padded, verbose=0)[0][0]

        # 4. Interprétation avec SEUIL DE SÉCURITÉ (0.75)
        # On passe à 0.75 pour éviter les faux positifs sur les phrases de transition
        seuil = 0.75 
        
        if prediction > seuil:
            # Calcul de la confiance (Détresse)
            confiance = prediction * 100
            print(f"⚠️  RÉSULTAT : DÉTRESSE (Confiance: {confiance:.2f}%)")
            print("💡 Conseil : Le modèle détecte des marqueurs de vulnérabilité émotionnelle.")
        else:
            # Calcul de la confiance (Sain)
            # Plus le score est proche de 0, plus la confiance "Sain" est élevée
            confiance = (1 - prediction) * 100
            print(f"✅ RÉSULTAT : SAIN / NEUTRE (Confiance: {confiance:.2f}%)")
            print("💡 Conseil : Le message semble stable ou positif.")
        
        # Petit debug pour toi (à supprimer plus tard si tu veux)
        print(f"DEBUG - Score brut : {prediction:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    predict_tool()