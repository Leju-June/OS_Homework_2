from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from app.model_service import analyze_face_from_image

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="얼굴형 측정 API",
    description="MediaPipe Face Mesh를 이용한 얼굴형(계란형, 둥근형 등) 판별 MLOps 서버입니다.",
    version="1.0.0"
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Shape Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --card-bg: rgba(255, 255, 255, 0.85); /* Glassmorphism 배경 */
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border-color: rgba(255, 255, 255, 0.5);
            --error: #ef4444;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            /* 트렌디한 캐주얼 웹사이트 스타일 (동적 그라데이션) */
            background: linear-gradient(-45deg, #ff9a9e, #fecfef, #a1c4fd, #c2e9fb);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .container {
            background-color: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 3rem 2.5rem;
            border-radius: 24px;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1), 0 10px 20px -10px rgba(0,0,0,0.05);
            border: 1px solid var(--border-color);
            width: 100%;
            max-width: 480px;
            text-align: center;
            /* 팝업 등장 애니메이션 */
            animation: popIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            transform: scale(0.95);
            opacity: 0;
        }
        @keyframes popIn {
            to { transform: scale(1); opacity: 1; }
        }
        h1 {
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
            color: #111827;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 2.5rem;
            font-weight: 500;
        }
        .upload-area {
            border: 2px dashed rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 1.5rem;
            background-color: rgba(255, 255, 255, 0.6);
        }
        .upload-area:hover, .upload-area.active {
            border-color: var(--primary);
            background-color: rgba(99, 102, 241, 0.05);
            transform: translateY(-2px);
        }
        .upload-area svg {
            width: 54px;
            height: 54px;
            color: var(--primary);
            margin-bottom: 1rem;
            opacity: 0.8;
            transition: transform 0.3s ease;
        }
        .upload-area:hover svg {
            transform: scale(1.1);
        }
        .upload-area span {
            display: block;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }
        .upload-area small {
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        .file-input { display: none; }
        .btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.875rem 1.5rem;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        .btn:hover {
            background-color: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
        }
        .btn:disabled {
            background-color: #a5b4fc;
            cursor: not-allowed;
            transform: translateY(0);
            box-shadow: none;
        }
        #preview-container {
            display: none;
            margin-bottom: 1.5rem;
            position: relative;
            animation: fadeIn 0.4s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        #preview-image {
            max-width: 100%;
            border-radius: 12px;
            max-height: 260px;
            object-fit: cover;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .remove-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.6);
            backdrop-filter: blur(4px);
            color: white;
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        .remove-btn:hover { background: rgba(0,0,0,0.9); }
        #result-container {
            display: none;
            margin-top: 2rem;
            padding: 2rem 1.5rem;
            background: linear-gradient(135deg, #f8fafc, #ffffff);
            border-radius: 16px;
            border: 1px solid rgba(0,0,0,0.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.02);
            animation: popIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .result-age {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #d946ef);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        .result-text {
            color: var(--text-muted);
            font-size: 1rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }
        #error-message {
            color: var(--error);
            font-size: 0.95rem;
            font-weight: 500;
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
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Face Shape Analyzer</h1>
    <p class="subtitle">AI 기반 얼굴형 분류 서비스</p>

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
        <span id="btn-text">얼굴형 측정하기</span>
        <div class="spinner" id="spinner"></div>
    </button>

    <div id="error-message"></div>

    <div id="result-container">
        <div class="result-text">얼굴형 및 성별 판별 결과</div>
        <div class="result-age">
            <span id="gender-value">-</span>성, <span id="shape-value">-</span>
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
    const shapeValue = document.getElementById('shape-value');
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

            shapeValue.textContent = result.predicted_shape;
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
        
        # 모델 서버를 통한 얼굴형 예측
        prediction = analyze_face_from_image(contents)
        
        return {
            "filename": file.filename, 
            "predicted_shape": prediction['face_shape'],
            "predicted_gender": prediction.get('predicted_gender', '알 수 없음'),
            "message": "성공적으로 분석되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 서버 내부 오류가 발생했습니다: {str(e)}")

# 개발 시 로컬 테스트용
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
