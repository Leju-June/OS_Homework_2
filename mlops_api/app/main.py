from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from app.model_service import predict_age_from_image

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="얼굴 인식 및 나이 예측 API",
    description="가벼운 얼굴 인식 모델(DeepFace)을 이용한 나이 예측 MLOps 서버입니다.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "환영합니다! /predict 경로를 통해 이미지를 업로드하고 나이를 예측해보세요!"
    }

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    # 파일 확장자/타입 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 업로드된 이미지 파일 읽기
        contents = await file.read()
        
        # 모델 서버를 통한 나이 예측
        predicted_age = predict_age_from_image(contents)
        
        return {
            "filename": file.filename, 
            "predicted_age": int(predicted_age),
            "message": "성공적으로 분석되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 서버 내부 오류가 발생했습니다: {str(e)}")

# 개발 시 로컬 테스트용
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
