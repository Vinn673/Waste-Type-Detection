from ultralytics import YOLO
import cv2

# Load model hasil training (ganti sesuai nama file)
model = YOLO("runs/detect/train/weights/best.pt")

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak ditemukan!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi
    results = model(frame, stream=True)

    # Gambar bounding box
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Gambar kotak dan tulisan
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Deteksi Sampah (Webcam)", frame)

    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
