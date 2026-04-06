# OS_Homework_2
오픈소스SW개발방법및도구 Homework2 과제용 레포지토리 입니다.

Step1 : Build the MLOps pipeline with Antigravity
prompt : "너는 시니어DevOps 엔지니어야. 나는지금내컴퓨터를서버로사용해서MLOps파이프라인을만들려고해. Python 기반의간단한얼굴인식(가벼운모델을사용해서이미지를업로드하면나이예측) API 서버코드를FastAPI로짜줘.그리고이프로젝트를실행하기위한requirements.txt와프로젝트구조를알려줘."

### 🚀 프로젝트 구조 (Project Structure)
API 서버, 모델 추론 로직, 그리고 배포 파일을 용도에 맞게 분리했습니다.
```text
mlops_api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI 메인 실행 파일 (API 라우팅)
│   └── model_service.py   # 모델 추론 로직 (DeepFace 사용)
├── requirements.txt       # 필요한 Python 패키지 목록
└── Dockerfile             # 컨테이너화/배포를 위한 파일
```

### 📦 주요 기술 스택 (Requirements)
- **`fastapi`, `uvicorn[standard]`**: 비동기 처리에 강하고 응답 속도가 빠른 API 서버 프레임워크입니다.
- **`python-multipart`**: 이미지(파일) 데이터 업로드를 쉽게 처리하기 위한 패키지입니다.
- **`deepface`**: 가볍고 빠르게 얼굴 인식, 나이/성별 예측 등을 진행해 주는 강력하지만 코드 베이스가 매우 짧은 경량화 래퍼(Wrapper) 딥러닝 라이브러리입니다.
- **`opencv-python-headless`**: 이미지를 메모리에서 로드/디코딩하기 위해 사용됩니다. (컨테이너 환경을 위해 UI가 제외된 headless 버전 활용)

### ⚙️ 로컬 서버 실행 및 테스트 방법 (How to Run)

**1. 터미널 이동 및 패키지 설치**
```bash
cd .\mlops_api
pip install -r requirements.txt
```

**2. API 서버 실행**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**3. API 테스트 방법 (Swagger UI)**
- **주소:** http://localhost:8000/docs 에 접속합니다.
- `POST /predict` 탭을 열고 `Try it out` 버튼을 누른 후, 얼굴이 포함된 이미지 파일을 업로드합니다.
- `Execute`를 누르면 나이(Age) 예측 결과값이 반환됩니다.
*(참고: 처음 추론 시에만 DeepFace의 기본 가중치(weights)를 한 번 다운로드 받습니다.)*


Step2 : Create Docker Image with Docker Hub
prompt : "방금만든얼굴인식앱을 Docker 이미지로만들고싶어.최적화된Dockerfile을작성해줘. 그리고GitHub에코드를push하면자동으로Docker Hub에이미지를빌드해서push하는.github/workflows/ci.yml파일도만들어줘. Docker Hub ID와PW는GitHub Secrets를사용한다고가정해줘."

### 🐳 Docker 최적화 및 데브옵스 CI/CD (Step 2)

#### 1. 최적화된 Dockerfile (`mlops_api/Dockerfile`) 적용
멀티스테이지(Multi-stage) 빌드 패턴을 도입하여 도커 이미지 용량을 최적화하고 애플리케이션 보안을 강화했습니다.
- **Builder Stage**: `gcc` 등의 빌드 도구를 두고 패키지들을 휠(`wheel`)로 먼저 빌드하여 용량과 빌드 시간을 개선했습니다.
- **Final Stage**: 필요한 런타임(운영) 환경만 남겼습니다. Root 권한 해킹 리스크를 줄이기 위해 `appuser`라는 접속 권한이 제한된 전용 유저를 만들어 실행하도록 구성했습니다.
- 함께 생성된 `.dockerignore`를 통해 로컬의 쓸데없는 캐시 파일이 이미지에 들어가는 것을 방지합니다.

#### 2. GitHub Actions 자동화 워크플로우 (`.github/workflows/ci.yml`) 구축
저장소의 `main` 브랜치에 코드가 푸시(Push)되거나 PR이 열릴 때 자동으로 작동하여 Docker Hub에 배포되는 파이프라인입니다.
- Docker의 `Buildx` 캐시를 적용하여 이후 빌드부터는 시간을 획기적으로 단축할 수 있도록 구성되었습니다.
- **사전 설정 방법**: 저장소를 GitHub에 올리신 후, `Settings` > `Secrets and variables` > `Actions` 메뉴에서 **New repository secret** 버튼을 눌러 다음 2개의 Secret 변수를 등록해야 작동합니다.
  - `DOCKERHUB_USERNAME`: 본인의 Docker Hub 아이디
  - `DOCKERHUB_TOKEN`: 본인의 Docker Hub 패스워드 (혹은 Access Token)
