import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# MediaPipe 모델 다운로드
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print("Downloading MediaPipe Face Landmarker model...")
    urllib.request.urlretrieve(url, model_path)

# FaceLandmarker 초기화
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

def analyze_face_from_image(image_bytes: bytes) -> dict:
    """
    이미지 바이트를 입력받아 MediaPipe Tasks API로 특징점(Landmarks)을 추출하고,
    기하학적 비율 분석을 통해 얼굴형을 판별합니다.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다. 올바른 이미지 파일인지 확인하세요.")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        raise ValueError("얼굴을 찾을 수 없습니다. 정면 얼굴이 잘 보이는 사진을 올려주세요.")
        
    landmarks = detection_result.face_landmarks[0]
    h, w, _ = img.shape
    
    # 헬퍼 함수
    def get_pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
    # 얼굴 길이 (이마 최상단 10 ~ 턱 끝 152)
    face_length = np.linalg.norm(get_pt(152) - get_pt(10))
    # 이마 너비 (양쪽 이마 측정점 68 ~ 298)
    forehead_width = np.linalg.norm(get_pt(68) - get_pt(298))
    # 광대 너비 (가장 바깥쪽 외곽선 234 ~ 454)
    cheek_width = np.linalg.norm(get_pt(234) - get_pt(454))
    # 턱선 너비 (하악각 부위 132 ~ 361)
    jaw_width = np.linalg.norm(get_pt(132) - get_pt(361))
    
    shape = "알 수 없음"
    width_ratio = face_length / cheek_width
    jaw_cheek_ratio = jaw_width / cheek_width
    forehead_cheek_ratio = forehead_width / cheek_width
    
    if jaw_cheek_ratio > 0.85 and forehead_cheek_ratio > 0.75:
        shape = "각진형"
    elif forehead_width > cheek_width and cheek_width > jaw_width:
        shape = "역삼각형"
    elif cheek_width > forehead_width and cheek_width > jaw_width and forehead_cheek_ratio < 0.9 and jaw_cheek_ratio < 0.8:
        shape = "마름모형"
    elif width_ratio < 1.25:
        shape = "둥근형"
    else:
        shape = "계란형"
        
    # [3] 추가된 성별 추론 로직 (Heuristics)
    # 눈과 눈썹 사이 거리 측정 (MediaPipe 랜드마크 65: 눈썹하단, 159: 눈상단)
    brow_dist = np.linalg.norm(get_pt(65) - get_pt(159))
    brow_ratio = brow_dist / face_length

    # 남성적 특징: 턱선비율(jaw_cheek_ratio)이 넓거나 눈-눈썹 간격(brow_ratio)이 짧은 편
    if jaw_cheek_ratio > 0.81 or brow_ratio < 0.042:
        gender = "남성"
    else:
        gender = "여성"
        
    return {
        "face_shape": shape,
        "predicted_gender": gender
    }
