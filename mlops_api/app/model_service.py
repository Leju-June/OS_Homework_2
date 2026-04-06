import cv2
import numpy as np
from deepface import DeepFace

def predict_age_from_image(image_bytes: bytes) -> int:
    """
    이미지 바이트를 입력받아 DeepFace 모델을 사용하여 나이를 예측합니다.
    """
    # 1. 파일 바이트를 numpy 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 2. OpenCV를 사용하여 이미지 디코딩
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다. 올바른 이미지 파일인지 확인하세요.")
        
    # 3. 모델을 사용하여 나이 예측
    # 처음 실행 시 weights 파일(약 패키지 크기)이 다운로드될 수 있습니다.
    # enforce_detection=False 옵션은 얼굴을 명확히 찾지 못하더라도 오류 없이 진행하도록 합니다.
    results = DeepFace.analyze(img, actions=['age'], enforce_detection=False)
    
    # 4. 결과 반환 (DeepFace가 여러 얼굴을 감지한 경우 리스트를 반환함)
    if isinstance(results, list):
        return results[0]['age']
    else:
        return results['age']
