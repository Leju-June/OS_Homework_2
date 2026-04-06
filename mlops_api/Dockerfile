# 1. Builder Stage: 의존성 패키지를 모아둔 빌더 환경
FROM python:3.9-slim AS builder

WORKDIR /build

# pip 최신화 및 휠(wheel) 빌드를 위한 기본 도구
RUN pip install --upgrade pip

COPY requirements.txt .

# 종속성 패키지들을 wheel 형태로 컴파일 (용량 최적화 및 빌드 속도 향상)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# 2. Final Stage: 실제 실행 환경 (최소한의 런타임 환경 구성)
FROM python:3.9-slim

WORKDIR /app


# Builder stage에서 만들어진 패키지 파일(wheels)을 복사 후 설치
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# 애플리케이션 코드 복사
COPY ./app ./app

# 🔥 보안 최적화: root 계정 대신 권한이 제한된 appuser 생성 및 사용
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
