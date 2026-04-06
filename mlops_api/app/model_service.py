import cv2
import numpy as np
from deepface import DeepFace

def analyze_face_from_image(image_bytes: bytes) -> dict:
    """
    이미지 바이트를 입력받아 DeepFace 모델을 사용하여 나이와 성별을 예측합니다.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다. 올바른 이미지 파일인지 확인하세요.")
        
    results = DeepFace.analyze(img, actions=['age', 'gender'], enforce_detection=False)
    
    if isinstance(results, list):
        target = results[0]
    else:
        target = results
        
    predicted_age = target['age']
    predicted_gender = target.get('dominant_gender', 'Unknown')
    
    # UI 표시를 위한 변환
    if predicted_gender == "Man":
        predicted_gender = "남성"
    elif predicted_gender == "Woman":
        predicted_gender = "여성"
        
    return {
        "age": predicted_age,
        "gender": predicted_gender
    }
