import os
import json
import warnings
import zipfile
from pathlib import Path

import streamlit as st
import requests
import gdown

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "📚 경남연구원 규정집 챗봇"
APP_CAPTION = "규정에 대해 궁금한 점을 물어보세요"

# 기대 폴더명(zip이 이 이름으로 풀리는 경우가 많음)
EXPECTED_DB_DIRNAME = "faiss_gyeongnam_rules"
ZIP_NAME = "faiss_db.zip"

# Google Drive 파일 ID (여기에 본인 DB zip의 ID)
GDRIVE_ID = "1kePVG0mv_YL45DdgR0YPaQknpTWJetrV"

# HuggingFace 임베딩 모델 (DB 생성 때 사용한 것과 반드시 동일해야 함)
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


# -----------------------------
# Helpers
# -----------------------------
def has_faiss_files(p: Path) -> bool:
    """FAISS load_local에 필요한 기본 파일이 있는지 확인."""
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()


def find_faiss_dir(search_root: Path) -> Path:
    """index.faiss를 기준으로 실제 FAISS 폴더를 탐색해 반환."""
    candidates = [p.parent for p in search_root.rglob("index.faiss")]
    if not candidates:
        raise FileNotFoundError("압축 해제 후 index.faiss를 찾지 못했습니다. (zip 내부 구조 확인 필요)")

    # 가장 상위(얕은) 폴더를 우선 사용
    candidates.sort(key=lambda p: len(p.parts))
    real_dir = candidates[0]

    if not has_faiss_files(real_dir):
        raise FileNotFoundError(f"{real_dir}에 index.faiss는 있지만 index.pkl이 없습니다.")
    return real_dir


# -----------------------------
# Vector DB download/unzip
# -----------------------------
@st.cache_resource
def prepare_vectordb() -> str:
    """
    1) 이미 로컬에 index.faiss/index.pkl이 있으면 그 경로 반환
    2) 없으면 gdown으로 zip 다운로드 후 압축 해제
    3) 압축 해제 결과에서 실제 index.faiss 위치를 찾아 그 경로 반환
    """
    base = Path(".").resolve()
    expected = base / EXPECTED_DB_DIRNAME
    zip_path = base / ZIP_NAME

    # ✅ 폴더가 "존재"가 아니라, 필수 파일이 "존재"할 때만 준비 완료
    if expected.exists() and has_faiss_files(expected):
        return str(expected)

    # 혹시 이전 실행에서 다른 위치에 이미 풀린 경우까지 탐색
    try:
        already = find_faiss_dir(base)
        return str(already)
    except Exception:
        pass

    st.info("🔄 데이터베이스 다운로드 중... (최초 1회, 1-2분 소요)")

    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"

    try:
        # 다운로드
        gdown.download(url, str(zip_path), quiet=False)

        # 압축 해제
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(base)

        # zip 제거
        if zip_path.exists():
            zip_path.unlink()

        # 실제 FAISS 폴더 찾기
        real_dir = find_faiss_dir(base)

        st.success("✅ 준비 완료! (FAISS 인덱스 확인)")
        return str(real_dir)

    except Exception as e:
        st.error("다운로드/초기화 실패")
        st.exception(e)
        st.stop()


# -----------------------------
# Load retriever
# -----------------------------
@st.cache_resource
def load_retriever(db_path: str):
    """
    db_path를 인자로 받아야 cache_resource가 올바르게 동작합니다.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )
    return retriever


# -----------------------------
# Gemini
# -----------------------------
def get_gemini_key() -> str:
    """
    Streamlit Cloud 권장:
    - Settings > Secrets 에 GEMINI_API_KEY 저장 후 st.secrets로 읽기
    또는 환경변수 사용
    """
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY", "")


def generate_answer(question: str, context: str, gemini_api_key: str) -> str:
    prompt = f"""당신은 경남연구원의 규정 전문가입니다.

**배경:**
- 경남연구원은 창원시에 위치
- 관내출장: 창원시 내
- 관외출장: 창원시 외 (부산, 김해 등)

**규정:**
{context}

**질문:** {question}

**지침:**
1. 자연스럽게 답변
2. 구체적 금액, 조건 명시
3. 페이지 인용
4. 추정 내용은 "⚠️ 원규집 재확인 필요" 표시
5. 존댓말 사용
6. 완결된 답변

**답변:**"""

    try:
        url = (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192},
        }

        response = requests.post(url, json=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]

        # 실패 시 상세 메시지 반환
        return f"오류 발생 (Gemini 응답 실패, status={response.status_code}): {response.text[:500]}"

    except Exception as e:
        return f"오류: {e}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="경남연구원 규정집 챗봇",
    page_icon="📚",
    layout="wide",
)

st.title(APP_TITLE)
st.caption(APP_CAPTION)

# 1) 벡터 DB 준비
db_path = prepare_vectordb()

# 2) Retriever 로드
with st.spinner("초기화 중..."):
    try:
        retriever = load_retriever(db_path)
    except Exception as e:
        st.error("초기화 실패 (FAISS 로드 실패)")
        st.exception(e)
        st.stop()

# 3) Gemini API 키 로드
GEMINI_API_KEY = get_gemini_key()
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다. (Streamlit Secrets 또는 환경변수로 설정 필요)")
    st.stop()

# 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("질문 입력 (예: 부산 출장비는?)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                docs = retriever.invoke(user_input)

                context = "\n\n".join(
                    [
                        f"[페이지 {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                        for doc in docs
                    ]
                )

                response = generate_answer(user_input, context, GEMINI_API_KEY)
                st.markdown(response)

                with st.expander("📄 참고 규정"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**[{i}] 페이지 {doc.metadata.get('page', 'N/A')}**")
                        st.text((doc.page_content or "")[:400] + "...")

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error("오류 발생")
                st.exception(e)

# 사이드바
with st.sidebar:
    st.header("📋 사용 안내")
    st.markdown(
        """
### 질문 예시
- 부산 출장 시 여비는?
- 연차 규정은?
- 경조사 휴가는?

### 정보
- 원규집: 2025.12.22
- AI: Gemini 2.5 Flash
"""
    )

    if st.button("🔄 초기화"):
        st.session_state.messages = []
        st.rerun()
