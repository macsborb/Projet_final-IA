import streamlit as st
#import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédicteur de Race de Chien",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

device = torch.device("cpu")  # Change to "cuda" if using GPU

##########################################
# Chargement des noms de classes
##########################################
class_names_file = r"C:\Users\20200337\Desktop\PROGRAMMATION\02_DOG_RACE_PREDICTION\dog_dataset_no_aug.h5"
with h5py.File(class_names_file, 'r') as f:
    class_names = f['class_names'][:]
    class_names = [cn.decode('utf-8') if isinstance(cn, bytes) else cn for cn in class_names]
class_names = ['Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua', 'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short-haired_pointer', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog', 'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'Shih-Tzu', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset', 'beagle', 'black-and-tan_coonhound', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dhole', 'dingo', 'flat-coated_retriever', 'giant_schnauzer', 'golden_retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound', 'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier', 'soft-coated_wheaten_terrier', 'standard_poodle', 'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wire-haired_fox_terrier']
num_classes = len(class_names)

##########################################
# Définition des modèles
##########################################
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=120):
        super(ResNet50Classifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class ResNet152Model(nn.Module):
    def __init__(self, num_classes=120):
        super(ResNet152Model, self).__init__()
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=120):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7,7))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

##########################################
# Modèle d'ensemble
##########################################
class EnsembleModel(nn.Module):
    def __init__(self, models_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.softmax = nn.Softmax(dim=1)
        
        if weights is None:
            self.weights = [1.0 / len(models_list)] * len(models_list)
        else:
            assert len(weights) == len(models_list), "Le nombre de poids doit correspondre au nombre de modèles."
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def forward(self, x):
        weighted_outputs = 0
        for weight, model in zip(self.weights, self.models):
            out = model(x)  # logits
            out = self.softmax(out)
            weighted_outputs += weight * out
        return weighted_outputs

##########################################
# Chargement des checkpoints de modèles
##########################################
model_dir = r"C:\Users\20200337\Desktop\PROGRAMMATION\04_DOG_PREDICTION_STREAMLIT\MODELS"

resnet50_path = os.path.join(model_dir, "resnet50_epoch_0025_valloss_0.5121.pth")
resnet152_original_path = os.path.join(model_dir, "resnet152_model_epoch_0040_valloss_0.4799.pth")
cnn_path = os.path.join(model_dir, "CNN_model_epoch_0030_valloss_2.9044.pth")

@st.cache_resource(show_spinner=False)
def load_all_models():
    # ResNet50
    model_resnet50 = ResNet50Classifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(resnet50_path, map_location=device)
    model_resnet50.load_state_dict(checkpoint['model_state_dict'])
    model_resnet50.eval()

    # ResNet152 (Original)
    model_resnet152_orig = ResNet152Model(num_classes=num_classes).to(device)
    checkpoint = torch.load(resnet152_original_path, map_location=device)
    model_resnet152_orig.load_state_dict(checkpoint['model_state_dict'])
    model_resnet152_orig.eval()

    # CNN
    model_cnn = CustomCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(cnn_path, map_location=device)
    model_cnn.load_state_dict(checkpoint['model_state_dict'])
    model_cnn.eval()

    weights = [0.5, 0.5]  # Ajustez les poids si nécessaire
    ensemble = EnsembleModel([model_resnet50, model_resnet152_orig], weights=weights).to(device)
    ensemble.eval()

    return model_resnet50, model_resnet152_orig, model_cnn, ensemble

model_resnet50, model_resnet152_orig, model_cnn, ensemble = load_all_models()

##########################################
# Transformations pour l'image
##########################################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

##########################################
# Interface Streamlit
##########################################
st.title("🐾 Prédicteur de Race de Chien")
st.markdown("""
Bienvenue sur le **Prédicteur de Race de Chien** ! Téléversez une photo de votre chien, choisissez le modèle, et nous prédirons sa race.
""")

# Ajout des explications sur les races de chiens prédictibles
st.markdown("### Voici toutes les races de chiens que nous pouvons prédire:")
st.markdown("""
- Shih Tzu
- Affenpinscher
- Lévrier Afghan
- Chien de chasse africain
- Airedale Terrier
- Staffordshire Terrier Américain
- Coonhound And-Tan
- Appenzeller
- Terrier Australien
- Basenji
- Basset
- Beagle
- Terrier de Bedlington
- Bouvier Bernois
- Épagneul de Blenheim
- Bloodhound
- Blue Tick Coonhound
- Border Collie
- Border Terrier
- Lévrier Russe
- Bouledogue Boston
- Bouvier des Flandres
- Boxer
- Griffon Brabantais
- Briard
- Épagneul Breton
- Mastiff
- Cairn Terrier
- Cardigan Welsh Corgi
- Retriever de la baie de Chesapeake
- Chihuahua
- Chow Chow
- Clumber Spaniel
- Retriever à poil dur
- Terrier Wheaten à poil dur
- Épagneul Cocker
- Collie
- Dandie Dinmont Terrier
- Dhole
- Dingo
- Dobermann
- Foxhound Anglais
- Setter Anglais
- Springer Spaniel Anglais
- Entlebucher Sennenhund
- Husky Esquimau
- Retriever à poil plat
- Bouledogue Français
- Berger Allemand
- Schnauzer Géant
- Golden Retriever
- Setter Gordon
- Dogue Allemand
- Berger des Pyrénées
- Grand Chien de Montagne Suisse
- Groenendael
- Fox Terrier à poil dur
- Braque Pointer à poil dur
- Lévrier Ibicenco
- Setter Irlandais
- Terrier Irlandais
- Épagneul d'eau Irlandais
- Dogue Irlandais
- Lévrier Italien
- Épagneul Japonais
- Keeshond
- Kelpie
- Terrier Kerry Bleu
- Komondor
- Kuvasz
- Labrador Retriever
- Terrier Lakeland
- Leonberg
- Lhasa Apso
- Malamute de l'Alaska
- Berger Malinois
- Bichon Maltais
- Xoloitzcuintli
- Épagneul Caniche Nain
- Caniche Toy
- Schnauzer miniature
- Terre-Neuve
- Terrier Norfolk
- Élan Norvégien
- Terrier de Norwich
- Berger Anglais
- Otterhound
- Papillon
- Pékinois
- Welsh Pembroke Corgi
- Poméranien
- Carlin
- Redbone Coonhound
- Rhodesian Ridgeback
- Rottweiler
- Saint-Bernard
- Saluki
- Samoyède
- Schipperke
- Terrier écossais
- Scottish Deerhound
- Sealyham Terrier
- Berger des Shetland
- Husky Sibérien
- Silky Terrier
- Staffordshire Bull Terrier
- Caniche Standard
- Schnauzer Standard
- Épagneul Sussex
- Mastiff tibétain
- Terrier tibétain
- Caniche Toy
- Terrier Toy
- Vizsla
- Walker Hound
- Weimaraner
- Springer Spaniel gallois
- Terrier blanc de West Highland
- Whippet
- Terrier Yorkshire
""")

# Ajout des accuracies des modèles
st.markdown("### Accuracy des modèles:")
st.markdown("""
- **Accuracy CNN**: 0.2300
- **Accuracy ResNet50**: 0.84
- **Accuracy ResNet152**: 0.83
- **Accuracy Ensemble**: 0.94
""")

# Choix du modèle
model_choice = st.radio(
    "Choisissez le modèle de prédiction :",
    ("CNN", "ResNet50", "ResNet152", "Ensemble des Resnet")
)

uploaded_file = st.file_uploader(
    "Téléchargez une photo de votre chien",
    type=["png", "jpg", "jpeg"],
    help="Choisissez un fichier image (png, jpg, jpeg) de votre chien."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Image du chien téléversée.', use_container_width=True)
    except Exception as e:
        st.error("Erreur lors du chargement de l'image. Veuillez vous assurer que le fichier est une image.")
        st.stop()

if uploaded_file is not None:
    # Bouton de prédiction
    if st.button("Prédire la race", use_container_width=True):
        with st.spinner('Prédiction en cours...'):
            # Sélection du modèle pré-chargé
            if model_choice == "ResNet50":
                model = model_resnet50
                apply_softmax = True
            elif model_choice == "ResNet152":
                model = model_resnet152_orig
                apply_softmax = True
            elif model_choice == "Ensemble des Resnet":
                model = ensemble
                apply_softmax = False  # Ensemble already applies Softmax
            else:
                model = model_cnn
                apply_softmax = True

            # Prétraitement de l'image
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Prédiction
            with torch.no_grad():
                output = model(input_tensor)
                if apply_softmax:
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                else:
                    probs = output.cpu().numpy()[0]  # Already probabilities

            # Tri des probabilités
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            sorted_breeds = np.array(class_names)[sorted_indices]

            # Afficher top 5
            top_k = 5 if len(class_names) > 5 else len(class_names)
            top_breeds = sorted_breeds[:top_k]
            top_probs = sorted_probs[:top_k]

            # Affichage du graphique
            fig, ax = plt.subplots(figsize=(8, 6))
            y_pos = np.arange(len(top_breeds))
            bars = ax.barh(y_pos, top_probs, color='skyblue')
            bars[0].set_color('red')  # La plus élevée en rouge
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_breeds)
            ax.invert_yaxis()  # Meilleure lisibilité
            ax.set_xlabel('Probabilité')
            ax.set_title('Prédictions du modèle')
            for i, v in enumerate(top_probs):
                ax.text(v + 0.01, i + 0.1, f"{v*100:.2f}%", color='black', va='center')
            st.pyplot(fig)

st.markdown("---")
st.markdown("Développé par Auguste DOYET, Robbie Blanc et Natal HOUSSET")
