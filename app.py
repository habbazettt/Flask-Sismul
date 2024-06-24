import os
from flask import Flask, render_template, request, redirect, url_for, send_file, abort
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import pywt
import soundfile as sf
import cv2

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'static/uploads/'
COMPRESSED_FOLDER = 'static/compressed/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(COMPRESSED_FOLDER):
    os.makedirs(COMPRESSED_FOLDER)

ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg'},
    'audio': {'mp3', 'wav'},
    'video': {'mp4', 'avi', 'mkv'}
}

def allowed_file(filename, filetype):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[filetype]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file and allowed_file(file.filename, 'image'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Compress image using DCT and DWT
        compressed_dct_path = compress_image_dct(filepath)
        compressed_dwt_path = compress_image_dwt(filepath)

        return render_template('index.html', message='Image uploaded and compressed successfully',
                               compressed_dct_path=compressed_dct_path, compressed_dwt_path=compressed_dwt_path)

    return render_template('index.html', message='Invalid file type')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['audio']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file and allowed_file(file.filename, 'audio'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Compress audio using DCT and DWT
        compressed_dct_path = compress_audio_dct(filepath)
        compressed_dwt_path = compress_audio_dwt(filepath)

        return render_template('index.html', message='Audio uploaded and compressed successfully',
                               compressed_dct_path=compressed_dct_path, compressed_dwt_path=compressed_dwt_path)

    return render_template('index.html', message='Invalid file type')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['video']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file and allowed_file(file.filename, 'video'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Compress video using DCT and DWT
        compressed_dct_path = compress_video_dct(filepath)
        compressed_dwt_path = compress_video_dwt(filepath)

        return render_template('index.html', message='Video uploaded and compressed successfully',
                               compressed_dct_path=compressed_dct_path, compressed_dwt_path=compressed_dwt_path)

    return render_template('index.html', message='Invalid file type')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

def compress_image_dct(image_path, quality=60):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply DCT
    dct_image = cv2.dct(np.float32(image_gray))

    # Quantize DCT coefficients
    quantized_dct_image = np.round(np.divide(dct_image, quality)) * quality

    # Inverse DCT
    idct_image = cv2.idct(quantized_dct_image)

    # Save compressed image
    compressed_image_path = os.path.join(COMPRESSED_FOLDER, os.path.basename(image_path)[:-4] + "_compressed_dct.jpg")
    cv2.imwrite(compressed_image_path, idct_image.astype(np.uint8))

    return compressed_image_path

def compress_image_dwt(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply DWT
    coeffs = pywt.dwt2(image_gray, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Quantize LL coefficients (can be adjusted)
    LL /= 32.0
    LH = np.zeros_like(LH)
    HL = np.zeros_like(HL)
    HH = np.zeros_like(HH)

    # Inverse DWT
    idwt_image = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    # Save compressed image
    compressed_image_path = os.path.join(COMPRESSED_FOLDER, os.path.basename(image_path)[:-4] + "_compressed_dwt.jpg")
    cv2.imwrite(compressed_image_path, idwt_image.astype(np.uint8))

    return compressed_image_path

def compress_audio_dct(audio_path, quality=60):
    output_path = os.path.join(COMPRESSED_FOLDER, os.path.basename(audio_path)[:-4] + "_compressed_dct.wav")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Apply DCT
    dct_coeffs = np.fft.rfft(y)

    # Quantize DCT coefficients
    quantized_dct_coeffs = np.round(dct_coeffs / quality) * quality

    # Reconstruct audio using inverse DCT
    y_compressed = np.fft.irfft(quantized_dct_coeffs)

    # Save compressed audio
    sf.write(output_path, y_compressed, sr)

    return output_path

def compress_audio_dwt(audio_path):
    output_path = os.path.join(COMPRESSED_FOLDER, os.path.basename(audio_path)[:-4] + "_compressed_dwt.wav")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Apply DWT
    coeffs = pywt.wavedec(y, 'db1', level=1)
    compressed_coeffs = [np.round(coef) for coef in coeffs]

    # Reconstruct audio using inverse DWT
    y_compressed = pywt.waverec(compressed_coeffs, 'db1')

    # Save compressed audio
    sf.write(output_path, y_compressed, sr)

    return output_path

def compress_video_dct(video_path, quality=60):
    output_path = os.path.join(COMPRESSED_FOLDER, os.path.basename(video_path)[:-4] + "_compressed_dct.mp4")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video at '{video_path}'")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply DCT
        dct_frame = cv2.dct(np.float32(frame_gray))

        # Quantize DCT coefficients
        quantized_dct_frame = np.round(np.divide(dct_frame, quality)) * quality

        # Inverse DCT
        idct_frame = cv2.idct(quantized_dct_frame)

        # Write the compressed frame
        out.write(idct_frame.astype(np.uint8))

    # Release everything if job is finished
    cap.release()
    out.release()

    return output_path

def compress_video_dwt(video_path):
    output_path = os.path.join(COMPRESSED_FOLDER, os.path.basename(video_path)[:-4] + "_compressed_dwt.mp4")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video at '{video_path}'")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply DWT
        coeffs = pywt.dwt2(frame_gray, 'haar')
        LL, (LH, HL, HH) = coeffs

        # Quantize LL coefficients (can be adjusted)
        LL /= 32.0
        LH = np.zeros_like(LH)
        HL = np.zeros_like(HL)
        HH = np.zeros_like(HH)

        # Inverse DWT
        idwt_frame = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

        # Convert back to BGR for writing to video
        compressed_frame = cv2.cvtColor(idwt_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Write the compressed frame
        out.write(compressed_frame)

    # Release everything if job is finished
    cap.release()
    out.release()

    return output_path

if __name__ == "__main__":
    app.run(debug=True)

