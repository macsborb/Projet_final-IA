import cv2
import numpy as np
import os

def load_yolo():
    # Chargement du modèle YOLOv3
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Chargement des classes
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def detect_dog(image_path, net, classes, output_dir):
    # Chargement de l'image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    print(f"Image loaded with shape: {image.shape}")
    
    # Conversion de l'image pour YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Récupération des couches de sortie
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Détection d'objets
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Parcours des sorties de YOLO
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75 and classes[class_id] == "dog":  # Rechercher uniquement les chiens
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordonnées du rectangle englobant
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Application de la suppression des doublons avec Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) + " " + str(round(confidences[i], 2))
            print(f"Dog detected: {label} at ({x}, {y}, {w}, {h})")

            # Redimensionner l'image à la taille du carré autour du chien
            # Utilisation de la plus grande dimension (w ou h) pour obtenir un carré
            side_length = max(w, h)
            
            # Calcul du carré centré sur le chien
            x_centered = max(x - (side_length - w) // 2, 0)
            y_centered = max(y - (side_length - h) // 2, 0)
            
            # Découpe du carré autour du chien
            cropped_image = image[y_centered:y_centered + side_length, x_centered:x_centered + side_length]

            # Redimensionner l'image pour une taille maximale de 244x244 tout en gardant les proportions
            max_size = 244
            aspect_ratio = width / height

            if width > height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)

            resized_image = cv2.resize(cropped_image, (new_width, new_height))

            # Enregistrer l'image redimensionnée dans le dossier de sortie
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, resized_image)
            print(f"Resized image saved to {output_path}")

    else:
        print("No dog detected.")


def process_images_in_directory(directory, net, classes):
    for root, dirs, files in os.walk(directory):  # Parcours récursif des dossiers
        # Créer un sous-dossier "resize" dans chaque dossier où les images seront enregistrées
        output_dir = os.path.join(root, 'resize')
        os.makedirs(output_dir, exist_ok=True)  # Crée le dossier "resize" si nécessaire
        
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):  # Filtrer uniquement les images
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}...")
                detect_dog(image_path, net, classes, output_dir)


# Chargement du modèle et des classes
net, classes = load_yolo()

# Dossier contenant les images
directory = "Images"  # Remplacez par votre chemin de dossier

# Traitement des images dans le dossier
process_images_in_directory(directory, net, classes)
