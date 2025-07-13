
import streamlit as st
from Chatbot import extract_video_id, get_transcript, build_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🎬 YouTube ChatBot", layout="centered")
st.title("YouTube Transcript ChatBot")
st.write("Ask questions based on any YouTube video with captions.")

url_input = st.text_input("🔗 Enter YouTube URL or Video ID")

if url_input:
    video_id = extract_video_id(url_input)

    with st.spinner("📥 Fetching transcript..."):
        transcript = get_transcript(video_id)

    if not transcript:
        st.error("❌ No transcript found or unavailable.")
    else:
        with st.spinner("🔧 Building AI chain..."):
            chain = build_chain(transcript)

        st.success("✅ Ready! Ask your questions below.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.text_input("❓ Your question")

        if question:
            with st.spinner("🤖 Thinking..."):
                answer = chain.invoke(question)

            st.session_state.chat_history.append((question, answer))

        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")

