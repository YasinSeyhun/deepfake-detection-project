import os
import shutil
import kagglehub
from tqdm import tqdm

def download_ffhq():
    print("FFHQ veri seti indiriliyor...")
    
    # Veri setini indir
    dataset_path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
    print(f"Veri seti şu konuma indirildi: {dataset_path}")
    
    # Hedef klasör
    target_dir = "data/ffhq"
    
    # Hedef klasörü temizle
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # Görüntüleri kopyala
    print("Görüntüler kopyalanıyor...")
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.copy2(src_path, dst_path)
    
    print("Veri seti başarıyla indirildi ve hazırlandı!")

if __name__ == "__main__":
    download_ffhq() 