import os
import torch
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from PIL import Image
import io

from models.stylegan import StyleGAN
from models.detector import TripletLossDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Upload klasörünü oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model yükleme
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = StyleGAN().to(device)
detector = TripletLossDetector().to(device)

# Model ağırlıklarını yükle
generator.load_state_dict(torch.load('checkpoints/generator_epoch_100.pt', map_location=device))
detector.load_state_dict(torch.load('checkpoints/detector_epoch_100.pt', map_location=device))

generator.eval()
detector.eval()

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Rastgele latent vektör oluştur
        z = torch.randn(1, 512).to(device)
        
        # Görüntü oluştur
        with torch.no_grad():
            fake_image = generator(z)
        
        # Görüntüyü PIL formatına dönüştür
        fake_image = fake_image.squeeze(0).cpu()
        fake_image = (fake_image + 1) / 2
        fake_image = transforms.ToPILImage()(fake_image)
        
        # Görüntüyü byte dizisine dönüştür
        img_byte_arr = io.BytesIO()
        fake_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return jsonify({
            'success': True,
            'image': img_byte_arr.decode('latin1')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        })
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        })
    
    if file and allowed_file(file.filename):
        try:
            # Görüntüyü oku ve dönüştür
            image = Image.open(file.stream).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Tespit yap
            with torch.no_grad():
                prediction, _ = detector(image_tensor)
                probability = prediction.item()
            
            return jsonify({
                'success': True,
                'probability': probability,
                'is_fake': probability > 0.5
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        'success': False,
        'error': 'Invalid file type'
    })

if __name__ == '__main__':
    app.run(debug=True) 