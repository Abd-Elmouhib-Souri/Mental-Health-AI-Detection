import os
import pandas as pd
import ssl
from data_loader import load_all_data
from preprocess_text import clean_text
from modeling import train_and_evaluate

# Correctif pour les erreurs de téléchargement SSL (NLTK/Certificats)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def main():
    output_file = os.path.join("data", "reddit_cleaned_master.csv")
    correction_file = os.path.join("data", "corrections.csv")
    
    # --- PHASE 1 & 2 : CHARGEMENT OU NETTOYAGE ---
    if os.path.exists(output_file):
        print(f"♻️ Fichier nettoyé trouvé ! Chargement de {output_file}...")
        df = pd.read_csv(output_file)
    else:
        print("🚀 Premier lancement : Nettoyage massif des données...")
        data = load_all_data() 
        df = data.get('text')
        
        if df is None or len(df) == 0:
            print("❌ Erreur : Aucun texte trouvé.")
            return

        df['cleaned_text'] = df['text'].apply(clean_text)
        df[['cleaned_text', 'label']].to_csv(output_file, index=False)
        print(f"💾 Données nettoyées sauvegardées dans : {output_file}")

    # --- PHASE DE CONSOLIDATION ---
    print("🔍 Filtrage et unification des labels...")
    df = df.dropna(subset=['cleaned_text', 'label'])
    
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    distress_keywords = ['1', 'drug', 'alcohol', 'trauma', 'stress', 'personality', 'early life']
    
    df['label'] = df['label'].apply(lambda val: 1 if any(k in val for k in distress_keywords) else 0)
    
    if len(df) > 500000:
        print("⚡ Limitation du dataset Reddit à 500,000 lignes...")
        df = df.sample(n=500000, random_state=42)

    # --- PHASE D'INJECTION DES CORRECTIONS (RÉSILIENCE) ---
    if os.path.exists(correction_file):
        print("🎯 Injection du dataset de correction pour le 'Zéro Faute'...")
        df_corr = pd.read_csv(correction_file)
        
        # On multiplie par 3000 pour que ces 50 exemples pèsent environ 150 000 lignes
        # Cela force le LSTM à accorder une importance capitale au mot "BUT"
        df_corr_repeated = pd.concat([df_corr] * 3000, ignore_index=True)
        
        # Nettoyage rapide des corrections au cas où
        df_corr_repeated['cleaned_text'] = df_corr_repeated['text'].apply(clean_text)
        
        # Fusion avec le dataset principal
        df = pd.concat([df, df_corr_repeated[['cleaned_text', 'label']]], ignore_index=True)
        print(f"✅ Injection terminée. Taille totale du dataset : {len(df)} lignes.")

    # --- PHASE 3 : ENTRAÎNEMENT ---
    train_and_evaluate(df)
    
    print("\n✅ Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()