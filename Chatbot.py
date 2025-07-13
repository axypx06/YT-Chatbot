
import re
from dotenv import load_dotenv
load_dotenv()
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser



# Utility to extract video ID
def extract_video_id(url_or_id):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    return match.group(1) if match else url_or_id  # Support raw video ID too

# Fetch transcript from YouTube
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        print("❌ No captions available for this video.")
        return None

# Build retrieval QA chain
def build_chain(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser
    return chain

# # Main interaction
# def main():
#     url_or_id = input("🔗 Enter YouTube video URL or ID: ").strip()
#     video_id = extract_video_id(url_or_id)
#     transcript = get_transcript(video_id)

#     if not transcript:
#         return

#     chain = build_chain(transcript)
    
#     print("\n✅ You can now ask questions about the video. Type 'exit' to quit.\n")
    
#     while True:
#         question = input("❓ Your question: ").strip()
#         if question.lower() in ["exit", "quit"]:
#             print("👋 Exiting. Thank you!")
#             break
        
#         answer = chain.invoke(question)
#         print(f"\n💬 Answer: {answer}\n")

# if __name__ == "__main__":
#     main()

