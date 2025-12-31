import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Téléchargement des ressources nécessaires (à faire une seule fois)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text):
    """ Nettoyage complet d'un texte """
    if not isinstance(text, str):
        return ""
    
    # 1. Mise en minuscule
    text = text.lower()
    
    # 2. Suppression des URLs (http/https)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Suppression de la ponctuation et des caractères spéciaux
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenization (couper le texte en mots)
    words = word_tokenize(text)
    
    # 5. Suppression des "Stopwords" (les mots inutiles)
    stop_words = set(stopwords.words('english')) # Reddit est principalement en anglais
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)

# --- Test du script ---
if __name__ == "__main__":
    test_phrase = "I am so sad today... Check this link: https://help.com #depressed"
    print(f"Avant : {test_phrase}")
    print(f"Après : {clean_text(test_phrase)}")