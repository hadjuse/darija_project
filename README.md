# Synthèse vocale pour le Darija marocain avec DiffWave

Ce dépôt contient le travail réalisé dans le cadre du projet **GenAI** sur la synthèse vocale du **Darija marocain** à l’aide du modèle **DiffWave** et du dataset **DODa**.

Le projet suit un pipeline complet :

1. prétraitement des données audio ;
2. entraînement d’un modèle DiffWave **from scratch** ;
3. fine-tuning d’un modèle DiffWave pré-entraîné sur LJSpeech ;
4. génération d’audios ;
5. évaluation et comparaison des différentes approches.

Le projet a été développé principalement sur **Google Colab**, afin de bénéficier d’un accès à un **GPU** pour l’entraînement. Les modèles de diffusion audio étant coûteux en calcul, l’utilisation de Colab a permis de réaliser des entraînements plus longs que sur nos machines personnelles.

---

## Organisation du dépôt

```text
.
├── notebooks/
│   ├── 01_preprocessing_doda.ipynb
│   ├── 02_diffwave_from_scratch_training_generation.ipynb
│   ├── 03_diffwave_training_fine_tuned_10k.ipynb
│   └── 04_evaluation_results.ipynb
│
├── results/
│   ├── results_10000_steps/
│   │   ├── figures/
│   │   ├── generated_audios/
│   │   └── results_csv/
│   │
│   ├── results_50000_steps/
│   │   ├── checkpoints_pt/
│   │   ├── figures/
│   │   ├── generated_audios/
│   │   └── results_csv/
│   │
│   └── results_fine_tuning_10000/
│       ├── figures/
│       ├── generated_audios/
│       └── results_csv/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset utilisé

Le projet utilise le dataset **DODa — Moroccan Darija Speech Dataset**, disponible sur Hugging Face.

DODa contient des échantillons audio en Darija marocain accompagnés de transcriptions. Dans notre projet, nous avons conservé uniquement les voix masculines afin de réduire la variabilité acoustique pendant l’entraînement.

Après filtrage, le corpus utilisé contient :

- **8012 exemples** ;
- environ **6,26 heures d’audio** ;
- **3 locuteurs masculins** : M1, M2 et M3.

Les données sont ensuite réparties en trois ensembles :

- **6489** exemples pour l’entraînement ;
- **721** exemples pour la validation ;
- **802** exemples pour le test.

Lien vers le dataset :  
[DODa — Moroccan Darija Speech Dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/hadjuse/darija_project.git
cd darija_project
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Le projet repose principalement sur :

- **PyTorch** pour l’entraînement des modèles ;
- **torchaudio**, **librosa** et **soundfile** pour le traitement audio ;
- **datasets** pour le chargement du dataset DODa depuis Hugging Face ;
- l’implémentation de **DiffWave** disponible sur GitHub.

---

## Exécution du projet

Les notebooks doivent être exécutés dans l’ordre suivant :

```text
01_preprocessing_doda.ipynb
02_diffwave_from_scratch_training_generation.ipynb
03_diffwave_training_fine_tuned_10k.ipynb
04_evaluation_results.ipynb
```

### `01_preprocessing_doda.ipynb`

Ce notebook prépare les données utilisées par les modèles :

- chargement du dataset DODa depuis Hugging Face ;
- reconstruction du mapping entre indices, locuteurs et genres ;
- filtrage des voix masculines ;
- analyse exploratoire du corpus ;
- conversion des audios en mono ;
- rééchantillonnage à 22 050 Hz ;
- normalisation des signaux ;
- extraction des spectrogrammes Mel ;
- création des splits `train`, `validation` et `test`.

À la fin de cette étape, un dossier de données prétraitées est généré :

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

Ce notebook entraîne DiffWave **from scratch**, c’est-à-dire à partir de poids initialisés aléatoirement, sans utiliser de modèle pré-entraîné.

Il contient :

- le chargement des données prétraitées ;
- la création des `Dataset` et `DataLoader` PyTorch ;
- l’entraînement du modèle ;
- la sauvegarde des checkpoints ;
- la génération d’audios à partir des spectrogrammes Mel du jeu de test ;
- la sauvegarde des courbes de loss, des spectrogrammes et des waveforms.

Deux entraînements from scratch ont été conservés dans les résultats :

- un entraînement à **10 000 steps** ;
- un entraînement à **50 000 steps**.

### `03_diffwave_training_fine_tuned_10k.ipynb`

Ce notebook entraîne une seconde version du modèle par **fine-tuning**.

Contrairement au modèle from scratch, celui-ci part d’un checkpoint DiffWave pré-entraîné sur **LJSpeech**, puis est adapté au dataset DODa pendant **10 000 steps**.

Cette branche permet de comparer :

- un modèle qui apprend uniquement à partir du Darija ;
- un modèle qui bénéficie d’une connaissance préalable de la parole acquise sur l’anglais.

### `04_evaluation_results.ipynb`

Ce notebook regroupe l’évaluation finale du projet.

Il permet notamment de :

- comparer les courbes d’apprentissage ;
- écouter les audios générés ;
- comparer les waveforms ;
- comparer les spectrogrammes Mel ;
- analyser les différences entre l’approche **from scratch** et l’approche **fine-tuning**.

---

## Données prétraitées

Les données audio prétraitées ne sont pas incluses dans le dépôt GitHub, car elles sont volumineuses.

Elles sont générées automatiquement par le notebook de prétraitement à partir du dataset DODa. Après exécution, l’organisation attendue est la suivante :

```text
data_preprocessed/
├── wavs/        # fichiers audio prétraités
├── mels/        # spectrogrammes Mel au format .npy
├── train.csv
├── val.csv
├── test.csv
└── summary.json
```

### Utilisation sous Google Colab

Le projet manipule plusieurs milliers de petits fichiers audio et de spectrogrammes. Lors des premiers essais, la lecture directe depuis Google Drive ralentissait fortement l’entraînement.

Pour cette raison, les données prétraitées sont stockées sur Google Drive, puis copiées temporairement dans l’espace local de Colab (`/content`) avant l’entraînement. Cette organisation permet :

- de conserver les données et les résultats sur Drive ;
- d’accélérer les accès disque pendant les entraînements ;
- d’éviter de dépendre uniquement des ressources locales de nos ordinateurs.

---

## Résultats disponibles

Le dossier `results/` contient les principaux résultats obtenus au cours du projet, afin qu’ils puissent être consultés sans avoir à relancer tous les entraînements.

```text
results/
├── results_10000_steps/
├── results_50000_steps/
└── results_fine_tuning_10000/
```

Chaque dossier contient, selon les expériences :

- les figures générées ;
- quelques audios produits par le modèle ;
- les fichiers CSV contenant les losses ;
- pour le modèle from scratch à 50 000 steps, les checkpoints sauvegardés.

Les résultats permettent notamment de comparer :

- l’évolution du modèle from scratch entre **10 000** et **50 000 steps** ;
- le modèle **from scratch** avec le modèle **fine-tuné**.

Les données complètes et les fichiers les plus volumineux ne sont pas versionnés dans le dépôt afin de conserver une structure légère.

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

## Ressources utilisées

Les principales ressources utilisées pour ce projet sont :

- l’implémentation officielle de **DiffWave** :  
  [lmnt-com/diffwave](https://github.com/lmnt-com/diffwave)

- l’article original présentant le modèle :  
  [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://openreview.net/forum?id=a-xFK8Ymz5J)

- le cours Hugging Face sur les modèles de diffusion audio, utilisé pour mieux comprendre le principe de génération audio et le rôle des spectrogrammes Mel :  
  [Hugging Face Diffusion Course — Unit 4](https://huggingface.co/learn/diffusion-course/unit4/3)

- le dataset utilisé :  
  [DODa — Moroccan Darija Speech Dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)

---

## Remarques

- Les résultats du modèle from scratch à **50 000 steps** correspondent au meilleur entraînement obtenu dans nos expériences.
- Le fine-tuning a été réalisé à partir d’un modèle pré-entraîné sur **LJSpeech**, puis adapté sur DODa.
- Les principaux résultats sont fournis dans `results/` afin de faciliter la consultation du projet.
- L’utilisation de **Google Colab avec GPU** est recommandée pour réexécuter les entraînements.

---

## Références

- Kong et al. (2021), *DiffWave: A Versatile Diffusion Model for Audio Synthesis*
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*
- Bidry et al. (2025), *DODa — Moroccan Darija Speech Dataset*
- van den Oord et al. (2016), *WaveNet: A Generative Model for Raw Audio*

