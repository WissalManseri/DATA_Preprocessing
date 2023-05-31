import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

# Chemin du jeu de données
# chemin/vers/votre/jeu/de/donnees
data_path = "DATSET/"

# Créer un détecteur de visages
#'chemin/vers/le/classificateur/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Définir les paramètres de l'ACP
n_components = 100

# Initialiser l'objet PCA
pca = PCA(n_components=n_components)

# Parcourir les fichiers du jeu de données
for folder in os.listdir(data_path):
    for file in os.listdir(os.path.join(data_path, folder)):
        # Charger l'image
        img = cv2.imread(os.path.join(data_path, folder, file))

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Redimensionner l'image
        resized = cv2.resize(gray, (100, 100))

        # Normaliser les couleurs
        normalized = cv2.equalizeHist(resized)

        # Détecter les visages
        faces = face_cascade.detectMultiScale(normalized)

        # Extraire le premier visage détecté
        (x, y, w, h) = faces[0]

        # Aligner le visage
        aligned = cv2.resize(normalized[y:y+h, x:x+w], (100, 100))

        # Appliquer l'ACP aux données
        flattened = aligned.reshape(1, -1)
        pca.fit(flattened)
        transformed = pca.transform(flattened)
        transformed = transformed.reshape(-1)

        # Enregistrer l'image prétraitée
        cv2.imwrite(os.path.join(data_path, folder, "preprocessed_" + file), transformed)
        
