print("1. Train")
print("2. Webcam detect")

pil = input("Pilih mode: ")

if pil == "1":
    import scripts.train
elif pil == "2":
    import scripts.webcam_detect
else:
    print("Pilihan tidak valid.")
