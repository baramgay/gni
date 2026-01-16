import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import json
import warnings
import os
import zipfile
import gdown

warnings.filterwarnings('ignore')

# 벡터 DB 자동 다운로드
@st.cache_resource
def download_vectordb():
    db_path = "./faiss_gyeongnam_rules"
    
    if os.path.exists(db_path):
        return db_path
    
    st.info("🔄 데이터베이스 다운로드 중... (최초 1회, 1-2분 소요)")
    
    # 여기에 Google Drive ID 입력
    gdrive_id = "1kePVG0mv_YL45DdgR0YPaQknpTWJetrV"  # 예: 1ABC123DEF456GHI789
    
    try:
        # Google Drive에서 다운로드
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        output = "faiss_db.zip"
        gdown.download(url, output, quiet=False)
        
        # 압축 해제
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(output)
        st.success("✅ 준비 완료!")
        return db_path
        
    except Exception as e:
        st.error(f"다운로드 실패: {e}")
        st.stop()
        return None

# 페이지 설정
st.set_page_config(
    page_title="경남연구원 규정집 챗봇",
    page_icon="📚",
    layout="wide"
)

st.title("📚 경남연구원 규정집 챗봇")
st.caption("규정에 대해 궁금한 점을 물어보세요")

# 벡터 DB 준비
db_path = download_vectordb()

# Gemini API 키
GEMINI_API_KEY = "AIzaSyBJcMn59KUl-937xdj00pXsEIHUpkeTTnA"

# 챗봇 로드
@st.cache_resource
def load_chatbot():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        return retriever
        
    except Exception as e:
        st.error(f"로드 오류: {e}")
        return None

with st.spinner("초기화 중..."):
    retriever = load_chatbot()

if retriever is None:
    st.error("초기화 실패")
    st.stop()

# Gemini API 호출
def generate_answer(question, context):
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
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 8192,
            }
        }
        
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result:
                return result['candidates'][0]['content']['parts'][0]['text']
        
        return "오류 발생"
        
    except Exception as e:
        return f"오류: {e}"

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
            docs = retriever.invoke(user_input)
            context = "\n\n".join([
                f"[페이지 {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}" 
                for doc in docs
            ])
            
            response = generate_answer(user_input, context)
            st.markdown(response)
            
            with st.expander("📄 참고 규정"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**[{i}] 페이지 {doc.metadata.get('page')}**")
                    st.text(doc.page_content[:400] + "...")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# 사이드바
with st.sidebar:
    st.header("📋 사용 안내")
    st.markdown("""
    ### 질문 예시
    - 부산 출장 시 여비는?
    - 연차 규정은?
    - 경조사 휴가는?
    
    ### 정보
    - 원규집: 2025.12.22
    - AI: Gemini 2.5 Flash
    """)
    
    if st.button("🔄 초기화"):
        st.session_state.messages = []
        st.rerun()