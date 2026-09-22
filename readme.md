# AI Detektor Sampah

Sistem deteksi sampah otomatis menggunakan YOLO (Ultralytics) dan OpenCV untuk mengidentifikasi dan mengklasifikasikan jenis sampah secara real-time.

## Struktur Project

```text
AI Detektor Sampah/
├── main.py             # Menu utama (Training & Deteksi Webcam)
├── requarement.txt     # Daftar dependencies Python
├── yolo11n.pt          # Model YOLOv11 nano default
├── yolov8n.pt          # Model YOLOv8 nano alternatif
├── best.pt             # Model hasil training kustom (jika ada)
├── scripts/            # Skrip pendukung (training, deteksi, dll)
└── dataset/            # Dataset untuk training
```

## Persyaratan Sistem

- Python 3.8+
- Webcam (untuk mode deteksi real-time)

## Instalasi

1. Clone repository atau unduh project ini.
2. Install dependencies yang dibutuhkan:
   ```bash
   pip install -r requarement.txt
   ```

## Cara Penggunaan

Jalankan skrip utama `main.py`:
```bash
python main.py
```

Anda akan diberikan dua pilihan mode:
1. **Training (`scripts/train.py`)**: Melatih model YOLO dengan dataset kustom.
2. **Webcam Detect (`scripts/webcam_detect.py`)**: Menjalankan deteksi sampah secara langsung menggunakan kamera web.

## Dependencies

- `ultralytics` (YOLO)
- `opencv-python` (Computer Vision)
- `numpy`
- `tqdm`
