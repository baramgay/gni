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

GDRIVE_FILE_ID = "1JaLtAm3Xyz2Ae70ucEL9UGven5EUBOBM"
ZIP_NAME = "faiss_db.zip"
EXTRACT_ROOT_DIRNAME = "faiss_db_extracted"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# Secrets에서 안전하게 가져오기
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")


# =========================
# 유틸 함수
# =========================
def has_faiss_files(p: Path) -> bool:
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()


def find_faiss_dir(search_root: Path) -> Path:
    candidates = [p.parent for p in search_root.rglob("index.faiss")]
    if not candidates:
        raise FileNotFoundError("압축 해제 후 index.faiss를 찾지 못했습니다.")

    candidates.sort(key=lambda p: len(p.parts))
    real_dir = candidates[0]

    if not has_faiss_files(real_dir):
        raise FileNotFoundError(f"{real_dir}에 index.faiss는 있지만 index.pkl이 없습니다.")
    return real_dir


def download_from_gdrive(file_id: str, destination: Path):
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


def generate_answer(question: str, context: str, chat_history: list = None) -> str:
    """
    chat_history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    
    # 대화 이력을 텍스트로 변환 (최근 4턴 = 8개 메시지)
    history_text = ""
    if chat_history:
        for msg in chat_history[-8:]:  # 최근 4턴(8개 메시지)만 사용
            role = "사용자" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n\n"
    
    prompt = f"""당신은 경남연구원의 규정 전문가입니다. 논리적 추론 능력을 발휘하여 정확하게 답변해주세요.

**핵심 배경 지식:**
- 경남연구원 위치: 창원시
- 관내출장: 창원시 내부
- 관외출장: 창원시 외부 (부산, 김해, 진주, 서울, 제주도 등 창원이 아닌 모든 지역)

**논리적 추론 규칙:**
1. 지역 분류 추론:
   - 서울 = 창원 아님 → 관외출장
   - 제주도 = 창원 아님 → 관외출장
   - 부산 = 창원 아님 → 관외출장
   - 모든 비창원 지역 = 관외출장

2. 금액 적용 추론:
   - 규정에 "서울 이외 지역 7만원"이 있다면
   - 제주도는 "서울 이외 지역"에 포함됨 → 7만원 적용
   - 부산도 "서울 이외 지역"에 포함됨 → 7만원 적용
   
3. 등급 적용 추론:
   - 규정에 명시되지 않은 지역은 가장 가까운 유사 등급 적용
   - "국외만 등급 표시"라면 → 국내는 별도 기준 적용

**제공된 규정:**
{context}

**이전 대화 내역 (최근 4턴):**
{history_text if history_text else "(없음)"}

**현재 질문:** {question}

**답변 작성 절차:**
Step 1: 이전 대화를 참고하여 맥락 파악 (대명사 "그곳", "그럼" 등 해석)
Step 2: 질문에서 언급된 지역이 창원인지 아닌지 먼저 판단
Step 3: 관내/관외 분류 확정
Step 4: 해당 분류에 적용되는 규정 찾기
Step 5: 논리적 추론으로 구체적 금액/조건 도출
Step 6: 명확하고 자연스럽게 답변 작성

**답변 규칙:**
1. 논리적 추론 과정을 자연스럽게 설명하되, "Step 1, Step 2" 같은 표현은 사용하지 마세요
2. 이전 대화를 참고하되, 불필요하게 반복 설명하지 마세요
3. 구체적 금액, 조건을 반드시 명시
4. 페이지 번호 인용
5. 확실하지 않은 추정은 "⚠️ 원규집 재확인 필요" 표시
6. 존댓말 사용
7. 완결된 답변

**중요: 지나치게 보수적으로 답변하지 마세요. 논리적으로 명확하게 추론 가능하면 자신있게 답변하세요.**

**답변:**"""

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 8192,
            "topP": 0.95,
            "topK": 40,
        },
    }

    # 재시도 로직
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=data, timeout=60)
            
            if r.status_code == 200:
                result = r.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            
            if r.status_code == 503 and attempt < max_retries - 1:
                import time
                time.sleep((attempt + 1) * 2)
                continue
            
            return f"⚠️ Gemini API 오류 (status={r.status_code}): 잠시 후 다시 시도해주세요."
            
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                continue
            return f"⚠️ 오류 발생: 잠시 후 다시 시도해주세요."
    
    return "⚠️ 서버가 혼잡합니다. 잠시 후 다시 시도해주세요."


# =========================
# Vector DB 준비
# =========================
@st.cache_resource
def prepare_vectordb() -> str:
    base = Path(".").resolve()
    extract_root = base / EXTRACT_ROOT_DIRNAME
    zip_path = base / ZIP_NAME

    if extract_root.exists():
        try:
            real_dir = find_faiss_dir(extract_root)
            return str(real_dir)
        except Exception:
            pass

    st.info("🔄 데이터베이스 다운로드 중... (최초 1회, 1~2분 소요)")

    download_from_gdrive(GDRIVE_FILE_ID, zip_path)

    if not zipfile.is_zipfile(zip_path):
        head = zip_path.read_bytes()[:300]
        raise RuntimeError(
            "다운로드된 파일이 ZIP이 아닙니다.\n"
            "Drive 공유 설정이 '링크가 있는 모든 사용자'인지 확인하세요."
        )

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

    try:
        zip_path.unlink()
    except Exception:
        pass

    real_dir = find_faiss_dir(extract_root)
    st.success("✅ 준비 완료!")
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
        search_kwargs={"k": 20},
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

try:
    db_path = prepare_vectordb()
except Exception as e:
    st.error("다운로드/초기화 실패")
    st.exception(e)
    st.stop()

with st.spinner("초기화 중..."):
    try:
        retriever = load_retriever(db_path)
    except Exception as e:
        st.error("초기화 실패")
        st.exception(e)
        st.stop()

if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
    st.error("GEMINI_API_KEY가 비어 있습니다.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("질문 입력 (예: 제주도 출장 숙박비는?)")

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

                # 이전 대화 이력 전달 (현재 질문 제외, 최근 4턴까지)
                chat_history = st.session_state.messages[:-1]
                response = generate_answer(user_input, context, chat_history)
                
                st.markdown(response)

                with st.expander("📄 참고 규정"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**[{i}] 페이지 {doc.metadata.get('page', 'N/A')}**")
                        st.text((doc.page_content or "")[:400] + "...")

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error("오류 발생")
                st.exception(e)

with st.sidebar:
    st.header("📋 사용 안내")
    st.markdown(
        """
### 질문 예시
- 제주도 출장 숙박비는?
- 서울 출장 시 일비는?
- 부산 관외출장 식비는?
- 연차 규정은?

### 정보
- 원규집: 2025.12.22
- AI: Gemini 2.5 Flash
- 대화 맥락: 최근 4턴
- 논리적 추론 강화
"""
    )

    if st.button("🔄 초기화"):
        st.session_state.messages = []
        st.rerun()
