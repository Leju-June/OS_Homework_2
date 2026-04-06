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

Step3 : "이제Docker Hub에올라간최신이미지를내로컬컴퓨터에서자동으로내려받아실행(Run)하고싶어. 내컴퓨터를GitHub Self-hosted Runner로등록했어. 이미지가푸시된후내컴퓨터에서기존컨테이너를중지하고새컨테이너를띄우는배포워크플로우를완성해줘."

### 🔄 로컬 자동 배포 파이프라인 (Step 3)

로컬 컴퓨터를 GitHub **Self-hosted runner**로 등록한 것을 활용하여, 클라우드 저장소(Docker Hub)에 이미지가 배포되면 즉시 내 컴퓨터에서 자동으로 컨테이너를 실행(CD)하도록 워크플로우(`.github/workflows/ci.yml`)를 확장했습니다.

#### 추가된 자동화 파이프라인(`deploy-to-local` Job)의 동작 과정:
1. **의존성 설정 (`needs: build-and-push`)**: 이전 단계인 빌드 및 푸시 작업이 성공해야만 배포 작업이 시작됩니다.
2. **Self-hosted 환경 작동 (`runs-on: self-hosted`)**: 등록하신 로컬 컴퓨터의 Runner 에이전트 위에서 구동됩니다. 사용 중인 Windows 환경에 맞게 `pwsh`(PowerShell) 쉘을 기반으로 명령어를 실행합니다.
3. **Docker Hub 로그인**: GitHub Secrets(`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`)를 이용해 터미널에 비밀번호가 남지 않는 보안 방식으로 안전하게 로그인합니다.
4. **최신 이미지 Pull**: 조금 전 빌드된 최신 API 이미지를 내 컴퓨터로 다운로드 받습니다.
5. **기존 컨테이너 정리 (무중단 대비)**: 이미 돌고있는 `age-predict-api` 컨테이너가 있다면 충돌하지 않도록 깔끔하게 중지(`stop`) 및 삭제(`rm`)합니다. 여기서 발생할 수 있는 에러(최초 실행 시 컨테이너가 없는 경우 등)는 `continue-on-error: true` 옵션으로 패스하도록 최적화했습니다.
6. **새로운 컨테이너 Run**: 방금 받은 깨끗한 최신 이미지로 백그라운드(`-d`)에서 8000포트를 열고 새 버전을 가동합니다.

#### 💡 사용 방법 및 결과
- 이제 GitHub에 코드를 Push하시기만 하면, **[코드 Push] → [GitHub 내부에서 빌드] → [Docker Hub 푸시] → [내 컴퓨터에서 자동으로 기존 서버 교체 후 새 서버 가동]** 까지 모든 런타임이 무인 자동화로 동작합니다! 
- Docker Desktop이나 터미널의 `docker ps` 명령어를 통해 최신 이미지로 컨테이너가 잘 돌고 있는지 직접 확인하실 수 있습니다.