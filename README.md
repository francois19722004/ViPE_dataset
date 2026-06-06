\# ViPE Dataset Visualization



Visualisation des annotations du dataset \[ViPE](https://github.com/nv-tlabs/vipe) (Video Pose Engine for 3D Geometric Perception) de NVIDIA : cartes de profondeur métriques, intrinsèques caméra et trajectoires 3D.



Projet réalisé dans le cadre du cours de Vision par Ordinateur — ENPC, 2A.



\## Structure du projet



```

.

├── README.md

├── requirements.txt

├── visualize\_depth.py          # Visualise 6 depth maps d'une séquence

├── visualize\_full\_v3.py        # 3 figures séparées : depth, intrinsèques, trajectoire

└── data/                       # ⚠️ NON COMMITÉ (voir section Dataset)

&#x20;   ├── meta.parquet

&#x20;   └── payload/

&#x20;       └── dpsp-012acfb6/

&#x20;           ├── depth.tar       # \~35 GB → contient des .zip de fichiers .exr

&#x20;           ├── pose.tar        # \~14 MB → contient des .npz

&#x20;           └── intrinsics.tar  # \~10 MB → contient des .npz + .txt

```



\## Prérequis



\- Python 3.10+

\- Un compte \[Hugging Face](https://huggingface.co/) avec un token d'accès (permission Read)

\- \~40 GB d'espace disque pour un shard du dataset



\## Installation



\### 1. Cloner le repo



```bash

git clone https://github.com/TON\_USERNAME/TON\_REPO.git

cd TON\_REPO

```



\### 2. Créer un environnement virtuel



\*\*Avec uv (recommandé) :\*\*

```bash

uv init

uv venv

\# Linux/Mac

source .venv/bin/activate

\# Windows PowerShell

.venv\\Scripts\\Activate.ps1

```



\*\*Ou avec venv classique :\*\*

```bash

python -m venv .venv

\# Linux/Mac

source .venv/bin/activate

\# Windows PowerShell

.venv\\Scripts\\Activate.ps1

```



\### 3. Installer les dépendances



```bash

pip install -r requirements.txt

```



\### 4. Se connecter à Hugging Face



```bash

pip install huggingface\_hub\[cli]

huggingface-cli login

```



Coller le token quand demandé. Le token se crée sur https://huggingface.co/settings/tokens (permission \*\*Read\*\*).



> \*\*Note :\*\* selon la version, la commande peut être `huggingface-cli` ou `hf`. Essayer les deux si l'une ne marche pas.



\## Téléchargement du dataset



Le dataset complet fait 3.5 TB. On ne télécharge qu'\*\*un seul shard\*\* (\~35 GB).



\### Étape 1 : Métadonnées + poses + intrinsèques (\~25 MB, rapide)



```bash

hf download nvidia/vipe-dynpose-100kpp \\

&#x20; payload/dpsp-012acfb6/pose.tar \\

&#x20; payload/dpsp-012acfb6/intrinsics.tar \\

&#x20; meta.parquet \\

&#x20; --repo-type dataset \\

&#x20; --local-dir ./data

```



\### Étape 2 : Depth maps (\~35 GB, long)



```bash

hf download nvidia/vipe-dynpose-100kpp \\

&#x20; payload/dpsp-012acfb6/depth.tar \\

&#x20; --repo-type dataset \\

&#x20; --local-dir ./data

```



\### Étape 3 : Extraire les fichiers tar



Les fichiers `.tar` doivent être extraits dans le même dossier.



\*\*Windows :\*\* clic droit sur chaque `.tar` → Extraire ici (ou avec 7-Zip)



\*\*Linux/Mac :\*\*

```bash

cd data/payload/dpsp-012acfb6

tar -xf pose.tar

tar -xf intrinsics.tar

tar -xf depth.tar

```



Après extraction, la structure doit être :



```

data/payload/dpsp-012acfb6/

├── depth/

│   ├── 36a3a713-69ad-4f3c-abf2-c3620290eae....zip

│   ├── 36a5fd27-16fa-4ac6-8e0d-52c0d4a370f4....zip

│   └── ...  (1000 fichiers .zip)

├── pose/

│   ├── 36a3a713-69ad-4f3c-abf2-c3620290eae....npz

│   └── ...  (1000 fichiers .npz)

├── intrinsics/

│   ├── 36a3a713-69ad-4f3c-abf2-c3620290eae....npz

│   ├── 36a3a713-69ad-4f3c-abf2-c3620290eae....txt

│   └── ...  (2000 fichiers : .npz + .txt)

├── depth.tar

├── pose.tar

└── intrinsics.tar

```



Chaque séquence est identifiée par un UUID. Le même UUID apparaît dans `depth/`, `pose/` et `intrinsics/`.



\## Format des données



C'est le point critique : NVIDIA utilise un encodage spécifique, pas du PNG ou du NPZ classique.



\### Depth maps



```

depth/{uuid}.zip

&#x20; └── 00000.exr      ← OpenEXR, canal "Z", float16

&#x20; └── 00001.exr

&#x20; └── ...

```



Chaque `.zip` contient une séquence vidéo. Chaque `.exr` est une frame. Le canal `"Z"` contient la profondeur métrique en mètres, encodée en \*\*float16\*\*. Il faut la librairie `OpenEXR` pour les lire.



\### Poses caméra



```

pose/{uuid}.npz

&#x20; ├── "data"  →  array (N, 4, 4) float64  —  matrices SE(3) cam-to-world

&#x20; └── "inds"  →  array (N,) int  —  indices des frames

```



\### Intrinsèques caméra



```

intrinsics/{uuid}.npz

&#x20; ├── "data"  →  array (N, 4) float64  —  \[fx, fy, cx, cy] en pixels

&#x20; └── "inds"  →  array (N,) int  —  indices des frames



intrinsics/{uuid}\_camera.txt

&#x20; → type de caméra par frame (PINHOLE, UNIFIED, etc.)

```



\## Utilisation



\### Visualiser les depth maps (6 frames d'une séquence)



```bash

python visualize\_depth.py

```



Génère `depth\_viz\_XXXXXXXX.png`.



\### Visualiser depth + intrinsèques + trajectoire



```bash

python visualize\_full\_v3.py

```



Génère 3 fichiers :

\- `1\_depth\_map.png` : carte de profondeur colorée (colormap turbo)

\- `2\_intrinsics.png` : matrice K superposée sur la depth map

\- `3\_trajectory.png` : trajectoire 3D de la caméra



\## Référence



```bibtex

@article{huang2025vipe,

&#x20; title={ViPE: Video Pose Engine for 3D Geometric Perception},

&#x20; author={Huang, Jiahui and Zhou, Qunjie and Rabeti, Hesam and others},

&#x20; journal={arXiv preprint arXiv:2508.10934},

&#x20; year={2025}

}

```



\## Liens



\- \[Paper (arXiv)](https://arxiv.org/abs/2508.10934)

\- \[Code source ViPE (GitHub)](https://github.com/nv-tlabs/vipe)

\- \[Dataset Dynpose-100K++ (Hugging Face)](https://huggingface.co/datasets/nvidia/vipe-dynpose-100kpp)

\- \[Documentation ViPE](https://nv-tlabs.github.io/vipe/)

