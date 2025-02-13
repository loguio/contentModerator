# Content Moderator

## Auteurs

- **Bourse Marius**
- **Croizier Jules**

## Description

Ce projet est une application web développée avec Streamlit et hébergée sur une instance EC2 d'AWS. Il permet d'analyser des images et vidéo.

## Prérequis

Avant de lancer le projet, assurez-vous d'avoir :

- Une instance EC2 fonctionnelle
- Python 3 installé sur l'instance
- Git installé sur l'instance
- Un environnement virtuel Python (optionnel mais recommandé)

## Installation

### 1. Connexion à l'instance EC2

Accédez à votre instance EC2

### 2. Mise à jour et installation des dépendances

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
```

### 3. Clonage du projet

```sh
git clone https://github.com/loguio/contentModerator.git
cd contentModerator
```

### 4. Installation des dépendances

```sh
pip install -r requirements.txt
```

## Exécution de l'application

Lancez l'application Streamlit avec la commande :

```sh
streamlit run SupDeVinci_ModerationApp/app.py --server.port 8501 --server.enableCORS false --server.headless true
```

Notre server actuel permet de ne pas avoir besoin de lancer cette commande au lancement du serveur EC2.

## Configuration du pare-feu AWS

- Ouvrez le port **8501** dans le groupe de sécurité de votre instance EC2 pour accéder à l'application depuis un navigateur.

## Accès à l'application

Ouvrez un navigateur et accédez à :

```
http://54.221.26.80:8501/
```
