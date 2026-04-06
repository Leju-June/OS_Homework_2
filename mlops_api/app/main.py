from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from app.model_service import analyze_face_from_image

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="얼굴 인식 및 나이 예측 API",
    description="가벼운 얼굴 인식 모델(DeepFace)을 이용한 나이 예측 MLOps 서버입니다.",
    version="1.0.0"
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Age Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #6b7280;
            --border-color: #e5e7eb;
            --error: #ef4444;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .container {
            background-color: var(--card-bg);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            width: 100%;
            max-width: 480px;
            text-align: center;
        }
        h1 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 1.5rem;
            background-color: #fafafa;
        }
        .upload-area:hover {
            border-color: var(--primary);
            background-color: #eff6ff;
        }
        .upload-area.active {
            border-color: var(--primary);
            background-color: #eff6ff;
        }
        .upload-area svg {
            width: 48px;
            height: 48px;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }
        .upload-area span {
            display: block;
            font-weight: 500;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }
        .upload-area small {
            color: var(--text-muted);
        }
        .file-input {
            display: none;
        }
        .btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.2s ease;
        }
        .btn:hover {
            background-color: var(--primary-hover);
        }
        .btn:disabled {
            background-color: #93c5fd;
            cursor: not-allowed;
        }
        #preview-container {
            display: none;
            margin-bottom: 1.5rem;
            position: relative;
        }
        #preview-image {
            max-width: 100%;
            border-radius: 8px;
            max-height: 250px;
            object-fit: cover;
        }
        .remove-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0,0,0,0.5);
            color: white;
            border: none;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .remove-btn:hover {
            background: rgba(0,0,0,0.8);
        }
        #result-container {
            display: none;
            margin-top: 1.5rem;
            padding: 1.5rem;
            background-color: #f8fafc;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        .result-age {
            font-size: 3rem;
            font-weight: 700;
            color: var(--primary);
        }
        .result-text {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }
        #error-message {
            color: var(--error);
            font-size: 0.9rem;
            margin-top: 1rem;
            display: none;
        }
        .spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Face Age & Gender Predictor</h1>
    <p class="subtitle">AI 기반 얼굴 나이 및 성별 분석 서비스</p>

    <div class="upload-area" id="drop-zone">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <span>클릭하거나 이미지를 끌어다 놓으세요</span>
        <small>PNG, JPG (최대 5MB)</small>
        <input type="file" id="file-input" class="file-input" accept="image/*">
    </div>

    <div id="preview-container">
        <img id="preview-image" src="" alt="미리보기">
        <button class="remove-btn" id="remove-btn">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
        </button>
    </div>

    <button id="predict-btn" class="btn" disabled>
        <span id="btn-text">나이 분석하기</span>
        <div class="spinner" id="spinner"></div>
    </button>

    <div id="error-message"></div>

    <div id="result-container">
        <div class="result-text">분석된 예측 결과</div>
        <div class="result-age">
            <span id="gender-value">-</span>, <span id="age-value">0</span>세
        </div>
    </div>
</div>

<script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');
    const resultContainer = document.getElementById('result-container');
    const ageValue = document.getElementById('age-value');
    const genderValue = document.getElementById('gender-value');
    const errorMessage = document.getElementById('error-message');

    let currentFile = null;

    // 드래그 앤 드롭 이벤트
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('active');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // 클릭하여 업로드
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            showError('이미지 파일만 업로드 가능합니다.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            previewImage.src = reader.result;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'block';
            predictBtn.disabled = false;
            hideError();
            resultContainer.style.display = 'none';
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        dropZone.style.display = 'block';
        previewContainer.style.display = 'none';
        predictBtn.disabled = true;
        resultContainer.style.display = 'none';
        hideError();
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.style.display = 'block';
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        predictBtn.disabled = true;
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        hideError();
        resultContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || '분석 중 오류가 발생했습니다.');
            }

            ageValue.textContent = result.predicted_age;
            genderValue.textContent = result.predicted_gender;
            resultContainer.style.display = 'block';

        } catch (error) {
            showError(error.message);
        } finally {
            predictBtn.disabled = false;
            btnText.style.display = 'block';
            spinner.style.display = 'none';
        }
    });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_TEMPLATE

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    # 파일 확장자/타입 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 업로드된 이미지 파일 읽기
        contents = await file.read()
        
        # 모델 서버를 통한 나이 및 성별 예측
        prediction = analyze_face_from_image(contents)
        
        return {
            "filename": file.filename, 
            "predicted_age": int(prediction['age']),
            "predicted_gender": prediction['gender'],
            "message": "성공적으로 분석되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 서버 내부 오류가 발생했습니다: {str(e)}")

# 개발 시 로컬 테스트용
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
