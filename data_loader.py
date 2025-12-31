import os
import pandas as pd
import glob

# Chemin racine où se trouvent tes dossiers de données
BASE_DATA_PATH = os.path.join("data", "Multimodel_Dataset")

def load_all_data():
    paths = {
        "audio": os.path.join(BASE_DATA_PATH, "Audio_Dataset"),
        "eeg": os.path.join(BASE_DATA_PATH, "EEG Data"),
        "video": os.path.join(BASE_DATA_PATH, "Video_Dataset"),
        "text": os.path.join(BASE_DATA_PATH, "Original Reddit Data")
    }
    
    loaded_data = {}
    print("--- 🚀 Phase 1 : Collecte Multimodale ---")
    
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"❌ Dossier {key.upper()} introuvable : {path}")
            continue

        if key == "text":
            # On inclut twitter_joy.csv s'il est dans le dossier racine de la data ou dans Original Reddit Data
            # Pour être sûr de le trouver, on cherche dans BASE_DATA_PATH aussi
            all_csv = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            
            # Vérification si twitter_joy est un niveau au dessus
            extra_joy = os.path.join(BASE_DATA_PATH, "twitter_joy.csv")
            if os.path.exists(extra_joy) and extra_joy not in all_csv:
                all_csv.append(extra_joy)

            df_list = []
            for f in all_csv:
                try:
                    fname = os.path.basename(f).lower()
                    
                    # --- CAS SPÉCIFIQUE : TWITTER JOY ---
                    if "twitter_joy" in fname:
                        print(f"🌈 Extraction des données de bonheur : {fname}")
                        cols = ["target", "ids", "date", "flag", "user", "text"]
                        # On charge en spécifiant l'encoding car Sentiment140 est en latin-1
                        temp_df = pd.read_csv(f, encoding='latin-1', names=cols)
                        # On garde UNIQUEMENT les positifs (target 4) et on force le label 0
                        temp_df = temp_df[temp_df['target'] == 4][['text']]
                        temp_df['label'] = 0
                        df_list.append(temp_df)
                        
                    # --- CAS GÉNÉRAL : REDDIT ---
                    else:
                        temp_df = pd.read_csv(f)
                        temp_df.columns = [c.lower().strip() for c in temp_df.columns]
                        target_col = 'selftext' if 'selftext' in temp_df.columns else 'text' if 'text' in temp_df.columns else None
                        
                        if target_col:
                            temp_df = temp_df.rename(columns={target_col: 'text'})
                            keywords_distress = ['dep', 'anx', 'sw', 'suicide', 'lone', 'hopeless']
                            if 'label' not in temp_df.columns:
                                temp_df['label'] = 1 if any(word in fname for word in keywords_distress) else 0
                            df_list.append(temp_df[['text', 'label']])
                            
                except Exception as e:
                    print(f"⚠️ Erreur sur {os.path.basename(f)} : {e}")

            if df_list:
                loaded_data[key] = pd.concat(df_list, ignore_index=True)
                print(f"✅ TEXTE : {len(loaded_data[key])} lignes chargées (Reddit + Twitter Joy).")
        
        else:
            files = [f for f in os.listdir(path) if not f.startswith('.')]
            loaded_data[key] = files
            print(f"✅ {key.upper()} : {len(files)} fichiers détectés.")
            
    return loaded_data