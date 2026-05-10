# Synthèse vocale pour le Darija marocain avec DiffWave

Ce dépôt contient le code du projet GenAI consacré à la synthèse vocale du Darija marocain à l’aide du modèle DiffWave et du dataset DODa.

Le projet suit un pipeline complet :

1. prétraitement des données audio ;
2. extraction des spectrogrammes Mel ;
3. entraînement de DiffWave from scratch ;
4. génération d’audios ;
5. évaluation des sorties obtenues.

Le projet a été principalement développé sur Google Colab, afin de bénéficier d’un environnement simple à partager et d’un accès à un GPU pour l’entraînement du modèle. L’entraînement d’un modèle de diffusion audio étant coûteux en calcul, l’utilisation de Colab a permis de dépasser les limites d’une exécution locale classique sur nos machines personnelles.

---

## Organisation du dépôt

```text
.
├── notebooks/
│   ├── 01_preprocessing_doda.ipynb
│   ├── 02_diffwave_from_scratch_training_generation.ipynb
│   └── 03_evaluation_results.ipynb
│
├── results/
│   ├── figures/
│   ├── csv/
│   └── generated_audios/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Le projet repose principalement sur :

- **PyTorch** pour l’entraînement du modèle ;
- **torchaudio**, **librosa** et **soundfile** pour le traitement audio ;
- **datasets** pour charger DODa depuis Hugging Face ;
- l’implémentation officielle de **DiffWave** disponible sur GitHub.

---

## Exécution du projet

Les notebooks doivent être exécutés dans l’ordre suivant :

```text
01_preprocessing_doda.ipynb
02_diffwave_from_scratch_training_generation.ipynb
03_evaluation_results.ipynb
```

Chaque notebook peut être lancé indépendamment une fois que les sorties du notebook précédent sont disponibles.

### `01_preprocessing_doda.ipynb`

Ce notebook prépare les données utilisées par le modèle :

- chargement du dataset DODa depuis Hugging Face ;
- reconstruction du mapping entre indices, locuteurs et genres ;
- filtrage des voix masculines ;
- analyse exploratoire du corpus ;
- conversion des audios en mono ;
- rééchantillonnage à 22 050 Hz ;
- normalisation des signaux ;
- extraction des spectrogrammes Mel ;
- sauvegarde des fichiers prétraités ;
- création des splits `train`, `validation` et `test`.

À la fin de cette étape, le dossier suivant est généré :

```text
data_preprocessed/
├── wavs/
├── mels/
├── train.csv
├── val.csv
├── test.csv
└── summary.json
```

### `02_diffwave_from_scratch_training_generation.ipynb`

Ce notebook contient la partie principale du projet :

- chargement des données prétraitées ;
- création des `Dataset` et `DataLoader` PyTorch ;
- initialisation de DiffWave avec des poids aléatoires ;
- entraînement du modèle **from scratch** ;
- sauvegarde des checkpoints ;
- génération d’audios à partir des spectrogrammes Mel du jeu de test ;
- sauvegarde des sorties audio et des figures associées.

Le modèle n’est donc pas chargé depuis un checkpoint pré-entraîné : il apprend uniquement à partir des données DODa préparées dans le premier notebook.

### `03_evaluation_results.ipynb`

Ce notebook regroupe les éléments d’évaluation :

- visualisation des courbes d’apprentissage ;
- comparaison des waveforms originales et générées ;
- comparaison des spectrogrammes Mel ;
- écoute des audios générés ;
- analyse qualitative des résultats.

---

## Données

Les données audio prétraitées ne sont pas incluses directement dans le dépôt GitHub car elles sont trop volumineuses.

Le notebook de prétraitement permet de les générer automatiquement à partir du dataset DODa.  
Une fois le prétraitement terminé, les fichiers attendus sont :

```text
data_preprocessed/
├── wavs/        # audios prétraités
├── mels/        # spectrogrammes Mel sauvegardés au format .npy
├── train.csv
├── val.csv
├── test.csv
└── summary.json
```

### Utilisation sous Google Colab

Lors de l’entraînement, le projet manipule plusieurs milliers de petits fichiers audio et Mel spectrograms.  
La lecture directe de ces fichiers depuis Google Drive peut ralentir fortement les itérations d’entraînement.

Pour cette raison, les données prétraitées sont d’abord stockées dans Google Drive, puis copiées temporairement dans l’espace local de Colab (`/content`) avant l’entraînement. Cette étape permet d’accélérer les accès disque tout en conservant les checkpoints et les résultats sur Drive.

En pratique :

- les **données d’entraînement** sont utilisées depuis `/content/data_preprocessed` ;
- les **checkpoints**, **figures** et **résultats** sont sauvegardés sur Google Drive afin de ne pas être perdus à la fermeture de la session Colab.

---

## Résultats fournis

Le dépôt contient un dossier `results/` permettant de consulter directement les principales sorties sans relancer tout l’entraînement :

```text
results/
├── figures/
├── csv/
└── generated_audios/
```

Ce dossier regroupe notamment :

- les courbes de loss ;
- les comparaisons entre signaux originaux et générés ;
- les comparaisons de spectrogrammes Mel ;
- les fichiers CSV associés aux expériences ;
- quelques exemples d’audios générés.

Les checkpoints et les données complètes ne sont pas versionnés afin d’éviter d’alourdir le dépôt. Les résultats essentiels sont néanmoins fournis afin de permettre une consultation rapide du projet, même sans réexécuter les notebooks.

---

## Configuration audio utilisée

| Paramètre | Valeur |
|---|---:|
| Fréquence d’échantillonnage | 22 050 Hz |
| Nombre de bandes Mel | 80 |
| `n_fft` | 1024 |
| `hop_length` | 256 |
| `win_length` | 1024 |
| Fréquence maximale | 8000 Hz |

---

## Remarques

- Le modèle implémenté dans ce dépôt correspond à la branche **from scratch** du projet.
- Les fichiers volumineux tels que les données prétraitées, les checkpoints et les sorties complètes ne sont pas destinés à être suivis par Git.
- Les principaux résultats sont fournis dans `results/` afin de faciliter la consultation du projet.
- Le dépôt a été organisé pour permettre de relancer chaque étape séparément et de reproduire le pipeline complet.
- L’utilisation de Google Colab est recommandée pour l’entraînement, car la génération audio par diffusion demande davantage de puissance de calcul qu’une exécution CPU classique.

---

## Références

- Kong et al. (2021), *DiffWave: A Versatile Diffusion Model for Audio Synthesis*
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*
- Bidry et al. (2025), *DODa — Moroccan Darija Speech Dataset*
- van den Oord et al. (2016), *WaveNet: A Generative Model for Raw Audio*

