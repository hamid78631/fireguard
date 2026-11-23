import time 

import cv2 

SOURCE=0 #0 = webcam 

def main() :
    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print(f"Impossible d'ouvrir la source vidéo : {SOURCE}")
        return 

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Source ouverte : {largeur}x{hauteur} à {fps_video:.1f} FPS")

    num_frame = 0 
    t_precedent = time.time()

    while True : 
        ok ,frame = cap.read()

        if not ok : 
            print("Fin de la vidéo ou caméra déconnectée")
            break
        num_frame += 1

        t_actuel = time.time()
        fps_reel = 1 / max(t_actuel - t_precedent, 1e-6)
        t_precedent = t_actuel 

        texte = f"Frame {num_frame} - FPS réel : {fps_reel:.1f}"
        cv2.putText(frame , texte , (10, 30 ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0 , 255 , 0) , 2)
        cv2.imshow("Fireguard - lecture vidéo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()