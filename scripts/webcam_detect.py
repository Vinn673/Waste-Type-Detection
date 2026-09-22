from ultralytics import YOLO
import cv2

# path model hasil training
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated = results[0].plot()

    cv2.imshow("Deteksi Sampah - YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
