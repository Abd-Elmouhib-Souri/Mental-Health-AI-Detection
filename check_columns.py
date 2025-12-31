import os
import glob
import pandas as pd

# On définit le chemin vers le dossier texte
path_text = os.path.join("data", "Multimodel_Dataset", "Original Reddit Data")

# On cherche TOUS les fichiers .csv récursivement
all_csv = glob.glob(os.path.join(path_text, "**", "*.csv"), recursive=True)

if not all_csv:
    print("❌ Aucun fichier CSV trouvé ! Vérifie que le dossier 'data' est bien dans 'ML Project'.")
else:
    # On prend le premier fichier trouvé pour inspecter les colonnes
    first_file = all_csv[0]
    print(f"🔍 Analyse du fichier : {first_file}")
    
    try:
        df = pd.read_csv(first_file, nrows=5) # On lit seulement 5 lignes pour aller vite
        print("\n✅ Colonnes trouvées :")
        print(df.columns.tolist())
    except Exception as e:
        print(f"❌ Erreur lors de la lecture : {e}")