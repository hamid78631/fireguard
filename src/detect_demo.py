import time

import cv2
from ultralytics import YOLO

SOURCE = 0
MODEL_PATH = "models/yolo11n.pt"
CONF_MIN = 0.5


def main():
    model = YOLO(MODEL_PATH)
    print(f"Modèle chargé : {len(model.names)} classes connues")
    print("Exemples :", list(model.names.values())[:10])

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la source : {SOURCE}")
        return

    t_precedent = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=CONF_MIN, verbose=False)
        r = results[0]

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        t_actuel = time.time()
        fps_reel = 1 / max(t_actuel - t_precedent, 1e-6)
        t_precedent = t_actuel
        cv2.putText(frame, f"{fps_reel:.1f} FPS | {len(r.boxes)} objets",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FireGuard - detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()