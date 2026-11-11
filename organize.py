import os
import shutil

# Ganti path ini sesuai dengan lokasi folder dataset test kamu yang berisi foto-foto tersebut
# Ingat gunakan \\ untuk Windows atau / 
dataset_path = r"C:\Users\nuril\OneDrive\Dokumen\Semester 7\downsyndrome\backend\tests\downsyndrome_dataset\test"

# Buat folder DS dan NORMAL
ds_dir = os.path.join(dataset_path, "DS")
normal_dir = os.path.join(dataset_path, "NORMAL")
os.makedirs(ds_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)

moved_ds = 0
moved_normal = 0

# Pindahkan file berdasarkan nama depannya
for filename in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, filename)
    
    # Lewati jika itu adalah folder
    if os.path.isdir(file_path):
        continue
        
    if filename.lower().startswith("down"):
        shutil.move(file_path, os.path.join(ds_dir, filename))
        moved_ds += 1
    elif filename.lower().startswith("healty") or filename.lower().startswith("healthy"):
        shutil.move(file_path, os.path.join(normal_dir, filename))
        moved_normal += 1

print(f"Selesai! {moved_ds} gambar dipindah ke folder DS, {moved_normal} gambar dipindah ke folder NORMAL.")