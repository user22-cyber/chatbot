"""DATA 폴더의 PDF를 대상으로 동작하는 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 프로젝트 최상단의 .env에서 API 키를 읽습니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "DATA"
load_dotenv(PROJECT_ROOT / ".env")


def remove_invalid_local_proxy() -> None:
    """연결을 거부하는 개발용 프록시 값만 제거합니다."""
    invalid_proxy = {"http://127.0.0.1:9", "https://127.0.0.1:9"}
    for variable_name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        if os.environ.get(variable_name, "").lower() in invalid_proxy:
            os.environ.pop(variable_name, None)


remove_invalid_local_proxy()


def read_pdf_documents() -> list[Document]:
    """DATA 폴더의 모든 PDF를 페이지 단위로 읽습니다."""
    documents: list[Document] = []

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )

    return documents


@st.cache_resource(show_spinner="문서를 읽고 벡터 저장소를 준비하는 중입니다...")
def build_rag_chain() -> tuple[object, object, list[Document], int]:
    """PDF를 분할하고 임베딩한 뒤 최신 LangChain 검색 체인을 만듭니다."""
    source_documents = read_pdf_documents()
    if not source_documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 PDF 텍스트를 찾지 못했습니다.")

    # 긴 페이지를 검색하기 좋은 작은 조각으로 나눕니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(source_documents)

    # OpenAI 임베딩으로 문서 조각을 벡터화합니다.
    # OpenAI 임베딩은 tiktoken 방식으로 입력을 토큰화합니다.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        tiktoken_enabled=True,
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # 답변은 검색된 문서 내용만 사용하도록 지시합니다.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 문서 기반 한국어 안내 챗봇입니다.
반드시 아래 Context에 있는 내용만 근거로 답변하세요.
Context에 답이 없거나 판단하기 어려우면 정확히 '문서에서 확인할 수 없습니다.'라고 답하세요.
일반 상식이나 학습된 기억으로 내용을 보충하거나 추측하지 마세요.
답변은 이해하기 쉬운 한국어로 작성하세요.

Context:
{context}""",
            ),
            ("human", "질문: {input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # LangChain 1.x의 Runnable 방식으로 프롬프트와 LLM을 연결합니다.
    answer_chain = prompt | llm | StrOutputParser()
    return retriever, answer_chain, chunks, len(chunks)


def normalize_search_text(text: str) -> str:
    """띄어쓰기 차이가 있는 PDF 텍스트를 비교하기 쉽게 정규화합니다."""
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def retrieve_documents(
    question: str, retriever: object, all_chunks: list[Document]
) -> list[Document]:
    """의미 검색 결과에 키워드 검색 결과를 보완해 검색 누락을 줄입니다."""
    semantic_documents = retriever.invoke(question)
    question_terms = [
        normalize_search_text(term)
        for term in re.findall(r"[가-힣A-Za-z0-9]{2,}", question)
    ]
    question_terms = list(dict.fromkeys(term for term in question_terms if term))

    keyword_matches: list[tuple[int, int, Document]] = []
    for index, document in enumerate(all_chunks):
        normalized_content = normalize_search_text(document.page_content)
        score = sum(term in normalized_content for term in question_terms)
        if score:
            keyword_matches.append((score, -index, document))

    combined: list[Document] = []
    seen_contents: set[str] = set()
    # 키워드로 정확히 일치하는 근거를 먼저 넣고, 의미 검색 결과를 추가합니다.
    candidates = [
        document for _, _, document in sorted(keyword_matches, reverse=True)[:4]
    ] + list(semantic_documents)
    for document in candidates:
        content_key = document.page_content
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            combined.append(document)
    return combined


def format_documents(documents: list[Document]) -> str:
    """검색된 문서 조각을 프롬프트에 넣을 하나의 문자열로 합칩니다."""
    return "\n\n--- 문서 조각 ---\n\n".join(
        document.page_content for document in documents
    )


def evidence_sentences(text: str, question: str, limit: int = 2) -> list[str]:
    """검색된 문서 조각에서 질문과 관련성이 높은 근거 문장을 고릅니다."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    keywords = {word for word in re.findall(r"[가-힣A-Za-z0-9]{2,}", question.lower())}
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-sum(keyword in item[1].lower() for keyword in keywords), item[0]),
    )
    return [sentence for _, sentence in ranked[:limit]]


def clear_conversation() -> None:
    """사용자가 요청했을 때만 현재 대화 기록을 삭제합니다."""
    st.session_state.pop("messages", None)


def main() -> None:
    st.set_page_config(page_title="문서 기반 RAG 챗봇", page_icon="📚")
    st.title("📚 문서 기반 RAG 챗봇")
    st.caption("DATA 폴더의 PDF만 근거로 답변합니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 대화 기록은 세션이 유지되는 동안 자동으로 보존됩니다.
    with st.sidebar:
        st.header("대화 관리")
        st.button(
            "대화 초기화",
            key="clear_conversation_button",
            use_container_width=True,
            on_click=clear_conversation,
        )
        st.caption("대화 기록을 지우고 싶을 때만 초기화 버튼을 누르세요.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error(".env 파일에 OPENAI_API_KEY를 입력한 뒤 앱을 다시 실행해 주세요.")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("출처와 근거 문장"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['file']}** (p. {source['page']})")
                        for evidence in source["evidence"]:
                            st.markdown(f"> {evidence}")

    question = st.chat_input("문서에 대해 질문해 주세요.")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 답을 찾는 중입니다..."):
            try:
                # 첫 질문이 들어왔을 때만 임베딩과 벡터 저장소를 준비합니다.
                retriever, answer_chain, all_chunks, chunk_count = build_rag_chain()
                retrieved_documents = retrieve_documents(
                    question, retriever, all_chunks
                )
                answer = answer_chain.invoke(
                    {
                        "input": question,
                        "context": format_documents(retrieved_documents),
                    }
                )
            except Exception as exc:
                retrieved_documents = []
                answer = "문서를 준비하는 중 오류가 발생했습니다."
                st.error(str(exc))
        st.markdown(answer)

        sources: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()
        for document in retrieved_documents:
            source_file = str(document.metadata.get("source", "알 수 없는 파일"))
            page = int(document.metadata.get("page", 0))
            source_key = (source_file, page)
            if source_key in seen:
                continue
            seen.add(source_key)
            sources.append(
                {
                    "file": source_file,
                    "page": page,
                    "evidence": evidence_sentences(document.page_content, question),
                }
            )

        if sources:
            with st.expander("출처와 근거 문장", expanded=True):
                for source in sources:
                    st.markdown(f"**{source['file']}** (p. {source['page']})")
                    for evidence in source["evidence"]:
                        st.markdown(f"> {evidence}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
