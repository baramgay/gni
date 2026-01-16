import os
import warnings
import zipfile
from pathlib import Path

import streamlit as st
import requests

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")


# =========================
# 설정값
# =========================
APP_TITLE = "📚 경남연구원 규정집 챗봇"
APP_CAPTION = "규정에 대해 궁금한 점을 물어보세요"

# ✅ 사용자가 준 Google Drive ZIP 파일 ID
GDRIVE_FILE_ID = "1JaLtAm3Xyz2Ae70ucEL9UGven5EUBOBM"

# ZIP 저장/해제 경로
ZIP_NAME = "faiss_db.zip"
EXTRACT_ROOT_DIRNAME = "faiss_db_extracted"  # 충돌 방지용 폴더

# ✅ 벡터DB 생성에 사용한 임베딩 모델과 반드시 동일해야 함
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# ✅ 요청대로: Secrets 없이 하드코딩
GEMINI_API_KEY = "AIzaSyBJcMn59KUl-937xdj00pXsEIHUpkeTTnA"


# =========================
# 유틸 함수
# =========================
def has_faiss_files(p: Path) -> bool:
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()


def find_faiss_dir(search_root: Path) -> Path:
    """
    unzip 결과에서 index.faiss를 찾아 실제 FAISS 폴더를 반환.
    zip 내부 구조가 중첩 폴더여도 자동으로 잡습니다.
    """
    candidates = [p.parent for p in search_root.rglob("index.faiss")]
    if not candidates:
        raise FileNotFoundError("압축 해제 후 index.faiss를 찾지 못했습니다. (zip 내부 구조 확인 필요)")

    candidates.sort(key=lambda p: len(p.parts))
    real_dir = candidates[0]

    if not has_faiss_files(real_dir):
        raise FileNotFoundError(f"{real_dir}에 index.faiss는 있지만 index.pkl이 없습니다.")
    return real_dir


def download_from_gdrive(file_id: str, destination: Path):
    """
    Google Drive confirm token(대용량/경고 페이지)을 처리해 실제 파일을 다운로드합니다.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    resp = session.get(URL, params={"id": file_id}, stream=True, timeout=120)
    resp.raise_for_status()

    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True, timeout=120)
        resp.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def generate_answer(question: str, context: str) -> str:
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

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192},
    }

    try:
        r = requests.post(url, json=data, timeout=60)
        if r.status_code == 200:
            result = r.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        return f"오류 발생 (Gemini 응답 실패, status={r.status_code}): {r.text[:500]}"
    except Exception as e:
        return f"오류: {e}"


# =========================
# Vector DB 준비
# =========================
@st.cache_resource
def prepare_vectordb() -> str:
    """
    1) 이미 unzip되어 index.faiss/index.pkl이 있으면 그 경로 반환
    2) 없으면 Drive에서 ZIP 다운로드 -> zip 검증 -> unzip
    3) unzip 결과에서 index.faiss 위치 탐색 후 그 폴더 반환
    """
    base = Path(".").resolve()
    extract_root = base / EXTRACT_ROOT_DIRNAME
    zip_path = base / ZIP_NAME

    # 이미 풀려있으면 재다운로드/재해제 안 함
    if extract_root.exists():
        try:
            real_dir = find_faiss_dir(extract_root)
            return str(real_dir)
        except Exception:
            pass

    st.info("🔄 데이터베이스 다운로드 중... (최초 1회, 1~2분 소요)")

    # 다운로드
    download_from_gdrive(GDRIVE_FILE_ID, zip_path)

    # ZIP 무결성 검증 (HTML 저장 방지)
    if not zipfile.is_zipfile(zip_path):
        head = zip_path.read_bytes()[:300]
        raise RuntimeError(
            "다운로드된 파일이 ZIP이 아닙니다. (권한/쿼터/경고 페이지가 내려왔을 가능성)\n"
            f"파일 앞부분(바이트): {head!r}\n"
            "Drive 공유 설정이 '링크가 있는 모든 사용자(Anyone with the link)'인지 확인하세요."
        )

    # unzip
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

    # zip 제거(원하면 유지해도 됨)
    try:
        zip_path.unlink()
    except Exception:
        pass

    real_dir = find_faiss_dir(extract_root)
    st.success("✅ 준비 완료! (FAISS 인덱스 확인)")
    return str(real_dir)


@st.cache_resource
def load_retriever(db_path: str):
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

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )


# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="경남연구원 규정집 챗봇",
    page_icon="📚",
    layout="wide",
)

st.title(APP_TITLE)
st.caption(APP_CAPTION)

# 벡터 DB 준비
try:
    db_path = prepare_vectordb()
except Exception as e:
    st.error("다운로드/초기화 실패")
    st.exception(e)
    st.stop()

# Retriever 로드
with st.spinner("초기화 중..."):
    try:
        retriever = load_retriever(db_path)
    except Exception as e:
        st.error("초기화 실패 (FAISS 로드 실패)")
        st.exception(e)
        st.stop()

# Gemini Key 체크(빈 값 방지)
if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
    st.error("GEMINI_API_KEY가 비어 있습니다. 코드 상단의 GEMINI_API_KEY 값을 확인하세요.")
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

                response = generate_answer(user_input, context)
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
