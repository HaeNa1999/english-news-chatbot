import os
import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="영어 뉴스 챗봇", page_icon="📰", layout="centered")
st.title("📰 영어 뉴스 챗봇")
st.caption("CNN 뉴스 기사를 한국어로 번역·요약해드립니다. | 김해나 (20261257)")

def get_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    return key

api_key = get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Cloud의 Secrets에 추가해주세요.")
    st.stop()

@st.cache_resource(show_spinner="뉴스 데이터베이스를 구축하는 중입니다. 잠시만 기다려주세요...")
def load_vector_db(api_key):
    df = pd.read_csv("CNN_Articles.csv")
    df = df[["headline", "text", "url"]].dropna()

    documents = [
        Document(
            page_content=row["text"],
            metadata={"headline": row["headline"], "source": row["url"]}
        )
        for _, row in df.iterrows()
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vector_db = FAISS.from_documents(split_docs[:100], embeddings)
    return vector_db

vector_db = load_vector_db(api_key)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

prompt = PromptTemplate.from_template("""
당신은 영어 뉴스를 한국어로 번역하고 요약하는 AI입니다.
검색된 기사만 참고하여 답변하세요.

기사 내용:
{context}

사용자 질문:
{question}

답변 형식:
1. 한국어 번역
2. 핵심 요약
3. 기사 출처
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("뉴스 관련 질문을 입력하세요 (예: 로봇 관련 뉴스 알려줘)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("관련 기사를 검색하는 중..."):
            docs = vector_db.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = "\n".join([f"- {doc.metadata['source']}" for doc in docs])
            final_prompt = prompt.format(context=context, question=question)
            response = llm.invoke(final_prompt)
            answer = response.content + "\n\n**출처:**\n" + sources
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
