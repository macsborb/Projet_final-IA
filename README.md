# Projet de Détection de Chiens avec YOLOv3

Ce projet utilise le modèle YOLOv3 pour détecter des chiens dans des images. Il redimensionne ensuite les images contenant des chiens pour qu'elles soient carrées, avec une taille maximale de 244x244 pixels, et les enregistre dans un sous-dossier `resize` de chaque dossier d'image.

## Prérequis

Avant de commencer, assurez-vous de suivre ces étapes :

1. **Télécharger le modèle YOLOv3** :
    - Téléchargez le fichier `yolov3.weights` à partir du lien suivant : [Télécharger YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights).
    - Placez ce fichier dans le même dossier que votre script Python.

2. **Fichier de configuration YOLOv3** :
    - Pas besoin de téléchargez le fichier de configuration `yolov3.cfg` mais si il y a un probleme avec mes fichiers vous pouvez le téléchargez à partir du lien suivant : [Télécharger YOLOv3 CFG](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).
    - Placez ce fichier dans le même dossier que votre script Python.

3. **Fichier `coco.names`** :
    - Pas besoin de téléchargez le fichier `coco.names` mais si il y a un probleme avec mes fichiers vous pouvez le téléchargez à partir du lien suivant : [Télécharger coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names).
    - Placez ce fichier dans le même dossier que votre script Python.

4. **Dataset des Images** : 
    - Vous devez disposer d'un dossier contenant des sous-dossiers d'images de chiens. L'arborescence du projet doit être structurée comme suit :

    ```
    Projet/
    ├── Images/
    │   └── dossier_des_races_de_chien/
    │       ├── image1.jpg
    │       ├── image2.png
    │       └── ...
    ├── yolov3.weights
    ├── yolov3.cfg
    ├── coco.names
    └── script.py
    ```

    - Le dossier `Images` contient des sous-dossiers, où chaque sous-dossier représente une catégorie de chien. Assurez-vous que les images à traiter se trouvent dans ces sous-dossiers.

6. **Installation des dépendances** :
    - Vous devez installer les bibliothèques nécessaires pour faire fonctionner ce projet. Utilisez le fichier `requirements.txt` ou installez les dépendances manuellement avec la commande suivante :
    
    ```bash
    pip install opencv-python numpy Pillow
    ```

## Utilisation

1. Clonez ce dépôt et placez les fichiers nécessaires dans les bons dossiers.
2. Exécutez le script Python. Ce script va parcourir tous les dossiers du dossier `Images`, détecter les chiens dans les images, les découper dans un carré, les redimensionner à une taille maximale de 244x244 pixels et enregistrer les images dans un sous-dossier `resize`.

    ```bash
    python script.py
    ```

3. Une fois le traitement terminé, vous trouverez les images redimensionnées dans un sous-dossier `resize` à l'intérieur de chaque dossier d'image.

    Exemple de structure après traitement :

    ```
    Projet/
    ├── Images/
    │   └── dossier_des_races_de_chien/
    │       ├── image1.jpg
    │       ├── image2.png
    │       └── ...
    │       └── resize/
    │           ├── image1_resized.jpg
    │           ├── image2_resized.jpg
    │           └── ...
    ├── yolov3.weights
    ├── yolov3.cfg
    ├── coco.names
    └── script.py
    ```

## Remarque

- **Le modèle YOLOv3 (`yolov3.weights`) n'est pas inclus dans ce dépôt en raison de sa taille.** Vous devez le télécharger manuellement.
- Assurez-vous que les fichiers nécessaires (`yolov3.weights`, `yolov3.cfg`, `coco.names`) se trouvent dans le même dossier que le script Python.

## Aide

Si vous avez des questions ou des problèmes avec l'installation ou l'exécution du projet, n'hésitez pas à ouvrir une **issue** sur GitHub, et je vous aiderai avec plaisir.

---

Merci d'utiliser ce projet de détection d'objets avec YOLOv3 !
