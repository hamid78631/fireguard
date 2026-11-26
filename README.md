# 🔥 FireGuard — Détection précoce de feu et de fumée avec alertes automatiques

> ⚠️ **Avertissement obligatoire.** FireGuard est un **prototype pédagogique**. Il **ne remplace en aucun cas** un système de détection incendie certifié (détecteurs de fumée, alarmes conformes aux normes). Cette phrase doit figurer dans le rapport, dans la démo et dans la soutenance.

Ce document est **la référence unique de l'équipe**. En cas de doute sur un nom de fichier, un format ou une règle, **c'est ce README qui fait foi**. Si une règle doit changer, on modifie ce README **par une Pull Request** pour que tout le monde soit au courant.

---

## Sommaire

1. [Le projet](#1-le-projet)
2. [Fonctionnement global](#2-fonctionnement-global)
3. [Technologies et rôle de chacune](#3-technologies-et-rôle-de-chacune)
4. [Arborescence (obligatoire)](#4-arborescence-obligatoire)
5. [Installation pas à pas](#5-installation-pas-à-pas)
6. [Données, modèles et vidéos](#6-données-modèles-et-vidéos)
7. [Répartition des rôles](#7-répartition-des-rôles)
8. [Contrats entre les modules](#8-contrats-entre-les-modules)
9. [Règles Git](#9-règles-git)
10. [Conventions de code](#10-conventions-de-code)
11. [Avancement et planning](#11-avancement-et-planning)
12. [Définition de « terminé »](#12-définition-de-terminé)
13. [Problèmes fréquents](#13-problèmes-fréquents)
14. [Crédits et sources](#14-crédits-et-sources)

---

## 1. Le projet

**Sujet (énoncé) :** construire un modèle de vision qui détecte les signes visuels de feu ou de fumée dans des images de caméra, et qui confirme les détections sur **plusieurs images consécutives** pour limiter les fausses alertes. Quand le seuil est atteint, le système **sauvegarde une preuve**, **enregistre l'événement** et **envoie une notification urgente** aux contacts désignés. Le prototype doit être testé sur des vidéos variées.

**Livrable :** un prototype de détection vidéo avec alertes automatiques, preuves des événements et **évaluation des faux positifs**.

---

## 2. Fonctionnement global

```
 Caméra / fichier vidéo
          │
          ▼
 [OpenCV] lecture image par image
          │
          ▼
 [YOLO] détection feu / fumée + confiance       ◀── models/fireguard_best.pt   (P1)
          │
          ▼
 [Filtre temporel] N images positives sur M ?   ── non ──▶ image suivante
          │ oui (et pas en cooldown)                                         (P2)
          ▼
 Événement confirmé
   ├── preuve : image annotée + clip vidéo  → evidence/
   ├── journal : base SQLite                → evidence/events.db
   └── message JSON publié sur MQTT         → topic fireguard/alerts
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 [FastAPI] consultation   [n8n] workflow d'alerte                         (P3 / P4)
 des événements et              │
 des preuves                    ├── Telegram (message + photo)
                                └── Email
```

---

## 3. Technologies et rôle de chacune

| Technologie | Rôle dans FireGuard | Responsable |
|---|---|---|
| **Python 3.12** | Langage de tout le projet | Tous |
| **YOLO (Ultralytics)** | Détection d'objets : *quoi* (feu/fumée) et *où* (boîte) | P1 |
| **OpenCV** | Lecture vidéo, dessin des boîtes, sauvegarde des preuves | P2 |
| **SQLite** | Journal des événements (fichier unique, sans serveur) | P2 |
| **FastAPI** | API web pour consulter les événements et les preuves | P3 |
| **MQTT (Mosquitto)** | Messagerie légère entre le détecteur et les alertes | P3 |
| **Docker / Docker Compose** | Lancer Mosquitto, l'API et n8n avec une seule commande | P3 |
| **n8n** | Workflow d'automatisation des alertes | P4 |
| **Telegram Bot API / Email** | Canaux de notification | P4 |
| **Google Colab** | Entraînement du modèle sur GPU gratuit | P1 |

---

## 4. Arborescence (obligatoire)

**Tout le monde utilise exactement cette arborescence.** On ne crée pas de dossier ailleurs et on ne renomme rien sans PR.

Légende : ✅ versionné dans Git · 🚫 ignoré par Git (local seulement) · 👤 responsable

```
fireguard/
├── README.md                    ✅  ce document (référence de l'équipe)
├── requirements.txt             ✅  dépendances Python, avec versions
├── .env.example                 ✅  modèle de configuration (sans secrets)
├── .env                         🚫  ta configuration locale (secrets)
├── .gitignore                   ✅
├── docker-compose.yml           ✅  👤 P3 (à créer, étape 8)
│
├── configs/                     ✅  fichiers de configuration
│   └── dfire.yaml               ✅  👤 P1  dataset pour YOLO (à créer)
│
├── data/
│   ├── dfire/                   🚫  dataset D-Fire extrait (voir §6.1)
│   │   ├── D-Fire/
│   │   │   ├── train/images/  train/labels/
│   │   │   ├── test/images/   test/labels/
│   │   │   ├── train_fold1.txt  (généré par make_split.py)
│   │   │   └── val_fold1.txt    (généré par make_split.py)
│   │   └── 5-fold cross validation/
│   └── videos/                  📁  vidéos de test (les .mp4 sont 🚫)
│       ├── fire/                    vidéos avec du feu
│       ├── smoke/                   vidéos avec de la fumée
│       ├── negative/                vidéos pièges SANS feu (soleil, bougies, phares…)
│       └── ground_truth.csv     ✅  👤 P4  vérité terrain (voir §8.6)
│
├── models/                      🚫  poids des modèles (.pt), partagés par Google Drive
│   └── fireguard_best.pt            👤 P1  modèle de référence
│
├── evidence/                    🚫  sorties du système : preuves + base SQLite
│
├── notebooks/                   ✅
│   └── train_colab.ipynb        ✅  👤 P1  notebook d'entraînement
│
├── src/                         ✅  code Python (lancé avec python -m src.xxx)
│   ├── video_reader.py              démo étape 2 (lecture vidéo)
│   ├── detect_demo.py               démo étape 3 (YOLO COCO)
│   ├── dataset_check.py         👤 P1  statistiques et vérification des labels
│   ├── make_split.py            👤 P1  génération train/val (à créer)
│   ├── common/                  👤 P2 + relecture P3  code partagé
│   │   ├── config.py                lecture du .env
│   │   └── schemas.py               format de l'événement (§8.2)
│   ├── pipeline/                👤 P2  détection temps réel
│   │   ├── detector.py              appel du modèle YOLO
│   │   ├── temporal_filter.py       filtre N sur M + cooldown
│   │   ├── evidence.py              sauvegarde image + clip
│   │   ├── event_store.py           écriture SQLite
│   │   ├── publisher.py             publication MQTT
│   │   └── run.py                   point d'entrée : python -m src.pipeline.run
│   ├── api/                     👤 P3
│   │   └── main.py                  application FastAPI
│   └── evaluation/              👤 P4
│       └── evaluate_events.py       évaluation des fausses alertes
│
├── tests/                       ✅  tests automatiques (au minimum le filtre temporel)
├── mosquitto/config/            ✅  👤 P3  mosquitto.conf
├── n8n/workflows/               ✅  👤 P4  workflows n8n exportés en .json
└── docs/
    ├── figures/                 ✅  schémas, courbes, captures pour le rapport
    └── report/                  ✅  rapport final
```

> Les fichiers `.gitkeep` sont des fichiers vides. Ils servent uniquement à ce que Git conserve les dossiers vides, pour que l'arborescence soit identique chez tout le monde après un `git clone`. **Ne pas les supprimer.**
>
> Les fichiers `__init__.py` (vides) font de `src` et de ses sous-dossiers des **packages Python**. Ils sont nécessaires pour les imports entre modules, par exemple `from src.common.schemas import ...`.

---

## 5. Installation pas à pas

Toutes les commandes se tapent dans **PowerShell**, depuis la **racine du projet**.

### 5.1 Prérequis (une seule fois par machine)

| Outil | Version | Vérification |
|---|---|---|
| Python | **3.12.x** (pas 3.13) | `python --version` |
| Git | récent | `git --version` |
| Docker Desktop | récent (P3 et P4 au minimum) | `docker --version` |
| Compte Google | pour Colab et le Drive partagé | — |

### 5.2 Récupérer le projet

```powershell
git clone https://github.com/hamid78631/fireguard.git
```

```powershell
cd fireguard
```

### 5.3 Environnement virtuel et dépendances

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Le terminal doit maintenant afficher `(.venv)` au début de la ligne. **Il faut réactiver le venv à chaque nouveau terminal.**

```powershell
pip install -r requirements.txt
```

### 5.4 Vérifier l'installation

```powershell
python -c "import ultralytics, cv2, torch; print('YOLO', ultralytics.__version__, '| OpenCV', cv2.__version__, '| CUDA:', torch.cuda.is_available())"
```

Résultat attendu : `YOLO 8.4.216 | OpenCV 5.1.0 | CUDA: False` (ou `True` si tu as un GPU NVIDIA).

### 5.5 Configuration locale

```powershell
Copy-Item .env.example .env
```

Remplis ensuite les valeurs dans `.env`. **Ne commite jamais `.env`.**

### 5.6 Modèle de démonstration (COCO)

Ce modèle sert uniquement aux démos des étapes 2 et 3 :

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

```powershell
Move-Item yolo11n.pt models\
```

### 5.7 Tester

```powershell
python -m src.detect_demo
```

Une fenêtre s'ouvre avec la webcam et des boîtes autour des objets. Touche `q` pour quitter.

---

## 6. Données, modèles et vidéos

Ces fichiers sont **trop lourds pour Git**. Voici où les trouver et **où exactement les ranger**.

### 6.1 Dataset D-Fire (P1 obligatoire, les autres facultatif)

1. Télécharger **`D-Fire.zip` (2,83 Go)** depuis le **1er lien** du README officiel : https://github.com/gaiasd/DFireDataset (lien « D-Fire dataset (only images and labels) »).
2. Extraire **dans `data/dfire/`**, pour obtenir **exactement** `data/dfire/D-Fire/train/...` et `data/dfire/D-Fire/test/...`.
3. Depuis le lien « Training, validation and test sets » du même README, télécharger le dossier **`5-fold cross validation`** et le placer dans `data/dfire/5-fold cross validation/`.
4. Vérifier les comptages :

| Dossier | Fichiers attendus |
|---|---|
| `data/dfire/D-Fire/train/images` | **17 221** |
| `data/dfire/D-Fire/train/labels` | **17 221** |
| `data/dfire/D-Fire/test/images` | **4 306** |
| `data/dfire/D-Fire/test/labels` | **4 306** |

**Statistiques du dataset** (obtenues avec `python -m src.dataset_check`) :

| | Train | Test |
|---|---|---|
| Images négatives (aucun objet) | 7 833 | 2 005 |
| Classe 0 | 9 550 boîtes / 8 444 images | 2 315 boîtes / 2 081 images |
| Classe 1 | 11 814 boîtes / 4 707 images | 2 878 boîtes / 1 115 images |

### 6.2 Dossier Google Drive partagé

Un dossier Drive unique, **`FireGuard`**, partagé avec les 4 membres. Lien : **_à compléter par le propriétaire du dépôt_**.

```
FireGuard (Google Drive)/
├── datasets/     D-Fire.zip + train_fold1.txt + val_fold1.txt + dfire.yaml  (pour Colab)
├── models/       tous les modèles entraînés (voir 6.3)
├── videos/       les vidéos de test, avec les MÊMES sous-dossiers que data/videos/
└── runs/         résultats bruts des entraînements Colab
```

### 6.3 Modèles

- Le modèle **de référence** utilisé par le pipeline s'appelle toujours **`models/fireguard_best.pt`**.
- Chaque modèle entraîné est archivé sur le Drive avec ce nom : `fireguard_v<numéro>_<architecture>_e<epochs>.pt`, par exemple `fireguard_v1_yolo11n_e50.pt`.
- Quand P1 annonce un nouveau modèle de référence, chacun le télécharge depuis le Drive et le **renomme** en `models/fireguard_best.pt`.
- Le fichier utilisé est défini par la variable `MODEL_PATH` dans `.env`.

### 6.4 Vidéos de test

- **Sources autorisées :** vidéos de surveillance D-Fire (lien « Some surveillance videos » du README officiel), Pexels, Pixabay. Toujours noter la source dans `ground_truth.csv`.
- **Rangement :** `data/videos/fire/`, `data/videos/smoke/` ou `data/videos/negative/`. Une vidéo qui contient feu **et** fumée va dans `fire/`.
- **Nom des fichiers :** `<categorie>_<numéro sur 2 chiffres>_<description>.mp4`, en minuscules, sans espaces ni accents. Exemples : `fire_01_cuisine.mp4`, `negative_03_coucher_soleil.mp4`.
- **Qualité :** 720p maximum, et durée entre 10 secondes et 2 minutes.
- **Objectif minimum :** 10 vidéos `fire`, 5 vidéos `smoke`, 10 vidéos `negative`.

---

## 7. Répartition des rôles

| Rôle | Membre | Blocs | Branches typiques |
|---|---|---|---|
| **P1 — Modèle de vision** | _à compléter_ | Étape 4 | `feat/dataset-split`, `feat/training` |
| **P2 — Pipeline de détection** | _à compléter_ | Étapes 5 et 6 | `feat/temporal-filter`, `feat/evidence` |
| **P3 — API, MQTT, Docker** | _à compléter_ | Étapes 7 et 8 | `feat/api`, `feat/mqtt-docker` |
| **P4 — Alertes et évaluation** | _à compléter_ | Étapes 9 et 10 | `feat/n8n-alerts`, `feat/evaluation` |

### P1 — Modèle de vision

- [ ] Vérifier visuellement le sens des classes 0 et 1 avec `dataset_check.py`, puis mettre à jour le §8.1.
- [ ] Écrire `src/make_split.py` (génère `train_fold1.txt` et `val_fold1.txt` à partir du pli 1).
- [ ] Écrire `configs/dfire.yaml`.
- [ ] Créer le dossier Drive `datasets/` et y envoyer le zip, les deux listes et le yaml.
- [ ] Écrire `notebooks/train_colab.ipynb` (extraction, entraînement, sauvegarde sur le Drive).
- [ ] Entraîner **au moins 2 configurations** (par exemple yolo11n et yolo11s, ou deux nombres d'epochs) et les comparer.
- [ ] Évaluer sur le **test set** : précision, rappel, mAP50, mAP50-95 **par classe**, matrice de confusion.
- [ ] Publier `fireguard_best.pt` sur le Drive et l'annoncer à l'équipe.
- [ ] Copier les courbes utiles dans `docs/figures/`.
- **Rapport :** sections « Données » et « Modèle ».

### P2 — Pipeline de détection

- [ ] `src/common/config.py` : lecture du `.env`.
- [ ] `src/common/schemas.py` : l'événement du §8.2, **relu par P3 et P4**.
- [ ] `src/pipeline/detector.py` : renvoie les détections d'une image.
- [ ] `src/pipeline/temporal_filter.py` : fenêtre glissante N sur M + cooldown. **Avec des tests dans `tests/`.**
- [ ] `src/pipeline/evidence.py` : image annotée + clip (quelques secondes avant et après).
- [ ] `src/pipeline/event_store.py` : table SQLite du §8.4.
- [ ] `src/pipeline/publisher.py` : publication MQTT du §8.3.
- [ ] `src/pipeline/run.py` : assemble le tout.
- [ ] Démarrer **tout de suite** avec `models/yolo11n.pt`, puis passer à `fireguard_best.pt` quand P1 le livre (simple changement de `MODEL_PATH`).
- **Rapport :** section « Logique anti-fausses alertes ».

### P3 — API, MQTT, Docker

- [ ] `mosquitto/config/mosquitto.conf` et Mosquitto dans Docker.
- [ ] `src/api/main.py` : les routes du §8.5.
- [ ] `docker-compose.yml` : Mosquitto, API et n8n.
- [ ] Tester l'API avec de **faux événements** sans attendre P2 (insérer des lignes SQLite à la main, publier sur MQTT à la main).
- [ ] Tenir à jour la section 5 (Installation) de ce README.
- **Rapport :** section « Architecture ».

### P4 — Alertes et évaluation

- [ ] Créer le bot Telegram et définir les contacts. Le token va **uniquement** dans `.env` et dans les identifiants n8n.
- [ ] Workflow n8n : MQTT → message Telegram avec photo + email → appel `POST /events/{id}/notified`.
- [ ] Exporter le workflow dans `n8n/workflows/fireguard_alerts.json` (**sans identifiants**).
- [ ] Collecter les vidéos de test (§6.4) et remplir `data/videos/ground_truth.csv` (§8.6).
- [ ] `src/evaluation/evaluate_events.py` : métriques du §8.7, **avec et sans** filtre temporel.
- **Rapport :** sections « Évaluation » et « Limites ».

---

## 8. Contrats entre les modules

**Ces formats sont figés.** Les modifier exige une PR relue par **tous** les membres concernés.

### 8.1 Classes du modèle

| Numéro | Nom |
|---|---|
| `0` | `smoke` |
| `1` | `fire` |

> ⚠️ **À confirmer par P1** avec la vérification visuelle (`python -m src.dataset_check`). Les statistiques vont dans ce sens : environ 1,1 boîte par image pour la classe 0 (un panache), environ 2,5 pour la classe 1 (plusieurs foyers). Mais ce n'est pas une preuve. Retirer cet avertissement une fois la vérification faite.

### 8.2 Événement d'alerte (JSON)

```json
{
  "event_id": "cam01_20261126T143205",
  "timestamp": "2026-11-26T14:32:05+01:00",
  "camera_id": "cam01",
  "classes": ["fire", "smoke"],
  "max_confidence": 0.87,
  "frames_confirmed": 8,
  "window_size": 10,
  "evidence_image": "evidence/cam01_20261126T143205.jpg",
  "evidence_clip": "evidence/cam01_20261126T143205.mp4"
}
```

| Champ | Type | Règle |
|---|---|---|
| `event_id` | texte | `<camera_id>_<AAAAMMJJ>T<HHMMSS>`, unique |
| `timestamp` | texte | ISO 8601 **avec** fuseau horaire |
| `camera_id` | texte | identique à `CAMERA_ID` du `.env` |
| `classes` | liste de textes | valeurs possibles : `"fire"`, `"smoke"` |
| `max_confidence` | réel entre 0 et 1 | confiance maximale sur la fenêtre |
| `frames_confirmed` | entier | nombre d'images positives dans la fenêtre |
| `window_size` | entier | taille de la fenêtre (`WINDOW_SIZE`) |
| `evidence_image` / `evidence_clip` | texte | chemin **relatif à la racine**, avec des `/` |

### 8.3 MQTT

| Paramètre | Valeur |
|---|---|
| Broker | Mosquitto, `localhost:1883` |
| Topic | `fireguard/alerts` |
| QoS | `1` |
| Contenu | le JSON du §8.2, encodé en UTF-8 |

### 8.4 Base SQLite (`evidence/events.db`)

```sql
CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    camera_id         TEXT NOT NULL,
    classes           TEXT NOT NULL,      -- liste JSON, ex. '["fire","smoke"]'
    max_confidence    REAL NOT NULL,
    frames_confirmed  INTEGER NOT NULL,
    window_size       INTEGER NOT NULL,
    evidence_image    TEXT,
    evidence_clip     TEXT,
    notified          INTEGER NOT NULL DEFAULT 0   -- 0 = non, 1 = oui
);
```

P2 **écrit** les événements. P3 les **lit** et met à jour `notified`.

### 8.5 API FastAPI (port 8000)

| Méthode | Route | Réponse |
|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/events?limit=50` | liste d'événements (§8.2), du plus récent au plus ancien |
| `GET` | `/events/{event_id}` | un événement, ou une erreur 404 |
| `GET` | `/evidence/{event_id}/image` | l'image JPEG de la preuve |
| `GET` | `/evidence/{event_id}/clip` | le clip MP4 de la preuve |
| `POST` | `/events/{event_id}/notified` | met `notified = 1` (appelé par n8n) |

### 8.6 Vérité terrain (`data/videos/ground_truth.csv`)

```csv
video,categorie,danger_present,debut_s,source,remarques
fire/fire_01_cuisine.mp4,fire,1,4.0,Pexels,feu de poêle
negative/negative_03_coucher_soleil.mp4,negative,0,,Pexels,ciel orange
```

| Colonne | Règle |
|---|---|
| `video` | chemin relatif à `data/videos/` |
| `categorie` | `fire`, `smoke` ou `negative` |
| `danger_present` | `1` s'il y a du feu ou de la fumée à un moment de la vidéo, sinon `0` |
| `debut_s` | seconde où le feu ou la fumée devient **visible** ; vide si `danger_present = 0` |
| `source` | site d'origine |
| `remarques` | texte libre, sans virgule |

### 8.7 Définitions pour l'évaluation (P4)

| Mesure | Définition |
|---|---|
| **Vraie alerte** | alerte dans une vidéo où `danger_present = 1`, déclenchée à un instant ≥ `debut_s` |
| **Fausse alerte** | alerte dans une vidéo `negative`, **ou** avant `debut_s` |
| **Détection manquée** | vidéo avec `danger_present = 1` sans aucune alerte |
| **Délai de détection** | instant de la 1re vraie alerte − `debut_s` (en secondes) |
| **Taux de fausses alertes** | fausses alertes ÷ durée totale des vidéos, **par heure** |

Chaque mesure est calculée **avec** le filtre temporel et **sans** (alerte dès la première image positive), pour prouver l'utilité du filtre.

---

## 9. Règles Git

1. **On ne pousse jamais directement sur `main`.** Tout passe par une Pull Request.
2. **Avant de commencer une tâche**, on part toujours d'un `main` à jour :

```powershell
git switch main
```

```powershell
git pull
```

```powershell
git switch -c feat/nom-de-la-tache
```

3. **Noms de branches :** `<type>/<sujet>` en minuscules, avec des tirets. Types : `feat` (fonctionnalité), `fix` (correction), `docs` (documentation), `chore` (configuration, outils).
4. **Messages de commit :** `<type>: <ce qui change>`, en français. Exemple : `feat: filtre temporel N sur M avec cooldown`. Un commit = un changement cohérent.
5. **Pull Request :**
   - **Titre** : comme un message de commit.
   - **Description** : *quoi*, *pourquoi*, *comment tester*.
   - **Au moins 1 relecture** par un membre d'**un autre rôle**.
   - C'est l'**auteur** qui fusionne, après approbation, puis il supprime la branche.
6. **Ne jamais commiter** : `.env`, les données, les vidéos, les modèles `.pt` ou le contenu de `evidence/`. Le `.gitignore` les bloque ; **ne jamais utiliser `git add -f`** pour forcer.
7. **Une nouvelle dépendance** s'ajoute dans `requirements.txt`, avec sa version, **dans la même PR**.

---

## 10. Conventions de code

- **Python 3.12**, style PEP 8 : 4 espaces, `snake_case` pour les fonctions et les variables, `MAJUSCULES` pour les constantes.
- **Noms en français, sans accents** (par exemple `lire_labels`, `seuil_confiance`). Commentaires en français.
- **On lance toujours depuis la racine du projet, en module** : `python -m src.pipeline.run`, et jamais `python src/pipeline/run.py`.
- **Chemins** : toujours relatifs à la racine (`models/...`, `evidence/...`), avec `pathlib.Path`.
- **Aucune valeur en dur** pour les réglages ou les secrets : tout passe par `.env` (via `src/common/config.py`). Toute nouvelle variable s'ajoute aussi dans `.env.example`.
- **Chaque fichier exécutable** se termine par `if __name__ == "__main__":`.

---

## 11. Avancement et planning

| Étape | Contenu | Responsable | État |
|---|---|---|---|
| 1 | Environnement, venv, Git | Tous | ✅ fait |
| 2 | Lecture vidéo OpenCV (`video_reader.py`) | Tous | ✅ fait |
| 3 | Première détection YOLO COCO (`detect_demo.py`) | Tous | ✅ fait |
| 4 | Dataset, entraînement, métriques | P1 | 🔄 en cours (dataset vérifié, split à faire) |
| 5 | Filtre temporel + cooldown | P2 | ⏳ à faire |
| 6 | Preuves + journal SQLite | P2 | ⏳ à faire |
| 7 | API FastAPI | P3 | ⏳ à faire |
| 8 | MQTT + Docker Compose | P3 | ⏳ à faire |
| 9 | n8n + Telegram/email | P4 | ⏳ à faire |
| 10 | Évaluation des faux positifs + rapport | P4 (+ tous) | ⏳ à faire |

**Date de rendu : _à compléter_**

| Semaine | Objectif |
|---|---|
| S1 | Tout le monde installé (§5). Contrats (§8) validés. Chacun démarre son bloc avec des données factices. |
| S2 | Développement en parallèle. P1 entraîne, P4 collecte les vidéos. |
| S3 | **Intégration** : chaîne complète vidéo → alerte Telegram. Démo interne. |
| S4 | Évaluation, rapport, préparation de la soutenance. |

**Qui met à jour ce tableau ?** Chaque responsable modifie la ligne de son étape dans la PR qui la termine.

---

## 12. Définition de « terminé »

Une tâche est terminée seulement si **toutes** ces conditions sont remplies :

- [ ] le code fonctionne depuis un **clone propre** en suivant ce README ;
- [ ] il respecte l'arborescence (§4) et les contrats (§8) ;
- [ ] aucun secret, aucune donnée et aucun modèle n'est commité ;
- [ ] les nouvelles dépendances sont dans `requirements.txt` et les nouvelles variables dans `.env.example` ;
- [ ] la PR est relue et approuvée par un autre membre ;
- [ ] le tableau du §11 est à jour.

---

## 13. Problèmes fréquents

| Symptôme | Cause | Solution |
|---|---|---|
| `Activate.ps1 ... l'exécution de scripts est désactivée` | La politique d'exécution de PowerShell bloque le script | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis réessayer |
| L'invite affiche `C:\...>` sans `PS` | Tu es dans **cmd**, pas dans PowerShell | Utiliser `.venv\Scripts\activate.bat`, ou ouvrir PowerShell |
| Le script se lance mais n'affiche rien | Le fichier n'est **pas enregistré** (taille 0) | Ctrl+S dans l'éditeur ; activer *Auto Save* dans VS Code |
| `ModuleNotFoundError: No module named 'src'` | Lancé depuis le mauvais dossier ou sans `-m` | Se placer à la racine et lancer `python -m src.xxx` |
| `ModuleNotFoundError: ultralytics` | Le venv n'est pas activé | `.\.venv\Scripts\Activate.ps1` |
| « Impossible d'ouvrir la source : 0 » | La webcam est utilisée par une autre application | Fermer Teams, Zoom, etc. |
| La vidéo défile trop vite | La boucle lit les images aussi vite que possible | Normal ; YOLO ralentit la boucle |

---

## 14. Crédits et sources

- **D-Fire dataset** : https://github.com/gaiasd/DFireDataset. **Citer l'article de référence indiqué dans leur README.**
- Images préfixées `PublicDataset` : issues de **HPWREN** (UC San Diego), du **Center for Wildfire Research** (FESB, Université de Split) et de **FireNet** (Jadon et al., 2019). Ces sources doivent être créditées dans le rapport.
- **Ultralytics YOLO** : https://docs.ultralytics.com
