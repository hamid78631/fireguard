import random
from collections import Counter
from pathlib import Path

import cv2

DATASET = Path("data/dfire/D-Fire")
COULEURS = {0: (255, 0, 0), 1: (0, 255, 255)}  # BGR : classe 0 bleu, classe 1 jaune


def lire_labels(chemin_txt):
    objets = []
    for ligne in chemin_txt.read_text().splitlines():
        if ligne.strip():
            cls, xc, yc, w, h = ligne.split()
            objets.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return objets


def statistiques(split):
    dossier = DATASET / split / "labels"
    nb_boites = Counter()
    nb_images = Counter()
    vides = 0

    for txt in dossier.glob("*.txt"):
        objets = lire_labels(txt)
        if not objets:
            vides += 1
        for o in objets:
            nb_boites[o[0]] += 1
        for cls in {o[0] for o in objets}:
            nb_images[cls] += 1

    print(f"--- {split} ---")
    print(f"Images sans aucun objet (négatives) : {vides}")
    for cls in sorted(nb_boites):
        print(f"Classe {cls} : {nb_boites[cls]} boîtes dans {nb_images[cls]} images")


def yolo_vers_pixels(xc, yc, w, h, largeur, hauteur):
    x1 = int((xc - w / 2) * largeur)
    y1 = int((yc - h / 2) * hauteur)
    x2 = int((xc + w / 2) * largeur)
    y2 = int((yc + h / 2) * hauteur)
    return x1, y1, x2, y2


def afficher_exemples(split, n=10):
    images = list((DATASET / split / "images").glob("*.jpg"))
    for chemin_img in random.sample(images, n):
        chemin_txt = DATASET / split / "labels" / (chemin_img.stem + ".txt")
        img = cv2.imread(str(chemin_img))
        hauteur, largeur = img.shape[:2]

        for cls, xc, yc, w, h in lire_labels(chemin_txt):
            x1, y1, x2, y2 = yolo_vers_pixels(xc, yc, w, h, largeur, hauteur)
            couleur = COULEURS.get(cls, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(img, f"classe {cls}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

        cv2.imshow("Labels - touche = suivante, q = quitter", img)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    statistiques("train")
    statistiques("test")
    afficher_exemples("train")