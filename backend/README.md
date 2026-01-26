# RhythmMamba Backend

Flask backend for video-based heart rate extraction using RhythmMamba model.

## Features

- **Video Upload**: Accepts MP4, AVI, WebM, MOV formats
- **Face Detection**: Automatic face region extraction using Haar Cascade
- **Preprocessing**: Frame extraction, cropping, resizing, normalization
- **Inference**: RhythmMamba model for rPPG signal extraction
- **FFT Analysis**: Heart rate calculation from rPPG waveform
- **REST API**: Easy integration with frontend

## Installation

### 1. Install Python Dependencies

The backend requires the same environment as the training code. If you already have the `rhythm` conda environment set up:

```bash
# Activate the conda environment
conda activate rhythm

# Install Flask dependencies
cd backend
pip install -r requirements.txt
```

### 2. Verify Model Checkpoint

The backend automatically searches for the trained model checkpoint:
- `../../../experiment0/user/PreTrainedModels/.../UBFC_UBFC_UBFC_RhythmMamba_Epoch29.pth`
- `../checkpoints/`
- `../results/`

Make sure your trained model is accessible from one of these locations.

## Usage

### Starting the Server

From the `backend` directory in WSL:

```bash
cd /mnt/<backend path>
conda activate rhythm
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

#### 2. Video Inference
```bash
POST /api/infer
Content-Type: multipart/form-data
Field: video (file)
```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/api/infer \
  -F "video=@test_video.mp4"
```

**Response:**
```json
{
  "success": true,
  "hr": 72.5,
  "hr_conf": 0.92,
  "quality": 0.89,
  "snr": 10.2,
  "waveform": [0.1, 0.2, ...],
  "processing_time": 2.3,
  "fps": 30.0,
  "model_metrics": {
    "mae": 0.54,
    "rmse": 0.94,
    "pearson": 0.99
  }
}
```

#### 3. Model Information
```bash
GET /api/model-info
```

**Response:**
```json
{
  "model": "RhythmMamba",
  "architecture": "Mamba SSM with Vision Transformer",
  "input_size": "128x128",
  "chunk_length": 160,
  "checkpoint": "UBFC_UBFC_UBFC_RhythmMamba_Epoch29.pth",
  "device": "cuda:0",
  "metrics": {
    "mae": 0.54,
    "rmse": 0.94,
    "pearson": 0.99,
    "snr": 10.2
  }
}
```

## Architecture

### Video Processing Pipeline

1. **Upload**: Receive video file via POST request
2. **Frame Extraction**: Extract all frames using OpenCV
3. **Face Detection**: Detect face region using Haar Cascade
4. **Crop & Resize**: Crop face region, resize to 128×128
5. **Normalization**: Standardize pixel values (mean=0, std=1)
6. **Chunking**: Split into 160-frame chunks
7. **Inference**: Pass through RhythmMamba model
8. **FFT Analysis**: Calculate heart rate from rPPG signal
9. **Response**: Return HR, waveform, and quality metrics

### Model Details

- **Input**: Video clips (160 frames, 128×128 RGB)
- **Output**: rPPG signal (photoplethysmogram)
- **Architecture**: Mamba SSM encoder with ViT-inspired patching
- **Parameters**: 144-dim embeddings, 12 layers, 20% dropout
- **Training**: 30 epochs on UBFC-rPPG dataset (10 subjects)
- **Performance**: MAE 0.54 bpm, RMSE 0.94 bpm, Pearson 0.99

## Troubleshooting

### Model Not Found
If you see "WARNING: No trained model checkpoint found":
- Check that training completed successfully
- Verify checkpoint path in `find_latest_checkpoint()` function
- Update the search paths to match your directory structure

### CUDA Out of Memory
If you encounter GPU memory issues:
- The backend processes one video at a time
- Reduce chunk size or process fewer chunks
- Use CPU inference: modify `device = torch.device('cpu')`

### Face Not Detected
If the model can't detect faces:
- Ensure video has clear frontal face visibility
- Check lighting conditions
- Verify `haarcascade_frontalface_default.xml` exists in `dataset/` directory
- The backend falls back to center crop if no face detected

## Development

### Adding New Features

- **Webcam Support**: Extend `/api/infer` to accept WebRTC streams
- **Batch Processing**: Process multiple videos simultaneously
- **Real-time Streaming**: WebSocket endpoint for live video
- **Additional Metrics**: Add respiration rate, SpO2 estimation

### Testing

Test the API with a sample video:

```bash
# Using the UBFC dataset video
curl -X POST http://localhost:5000/api/infer \
  -F "video=@../Data/DATASET_2/subject1/vid.avi" \
  -o response.json

# Check the response
cat response.json | python -m json.tool
```

## Production Deployment

For production use:

1. **Use Gunicorn**: Replace Flask development server
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Enable HTTPS**: Use nginx reverse proxy with SSL

3. **Add Authentication**: Implement API key or OAuth

4. **Rate Limiting**: Prevent abuse with flask-limiter

5. **Monitoring**: Add logging and error tracking

## License

Same as main project (MIT License)
