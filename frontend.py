import streamlit as st
import pandas as pd
import numpy as np
import time
from openai import OpenAI
from pinecone import Pinecone

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="CSV → Pinecone Uploader", layout="wide")
st.title("📊 CSV to Pinecone Uploader")

# ------------------ INPUTS ------------------
openai_key = st.text_input("🔑 OpenAI API Key", type="password")
pinecone_key = st.text_input("🔑 Pinecone API Key", type="password")

index_name = st.text_input("📦 Pinecone Index Name", value="peepalfarm-static-data")
namespace = st.text_input("🧠 Namespace", value="faqs")

uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

# ------------------ INIT CLIENTS ------------------
def init_clients():
    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    return client, index

# ------------------ EMBEDDINGS ------------------
def get_embeddings(client, text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype="float32").tolist()
    except Exception as e:
        st.error(f"❌ Embedding Error: {e}")
        return None

# ------------------ UPLOAD FUNCTION ------------------
def upload_data(df, client, index, namespace, question_col, answer_col, batch_size=100):

    # Clean + FIX index issue
    df = df.dropna(subset=[question_col, answer_col]).reset_index(drop=True)
    total_rows = len(df)

    st.write(f"🚀 Uploading {total_rows} rows...")

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    vectors = []

    for idx, row in df.iterrows():
        title = str(row[question_col])
        answer = str(row[answer_col])
        context = f"Query: {title} | Answer: {answer}"

        vector = get_embeddings(client, context)

        if vector:
            vectors.append({
                "id": f"{namespace}_{idx}_{int(time.time())}",
                "values": vector,
                "metadata": {
                    "title": title,
                    "context": context
                }
            })

        # Batch upload
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors, namespace=namespace)
            vectors = []

        # Safe progress (NEVER > 1.0)
        progress = min((idx + 1) / total_rows, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx+1}/{total_rows}")

    # Final batch
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

    progress_bar.progress(1.0)
    st.success("✅ Upload complete!")

# ------------------ MAIN ------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("👀 Preview Data")
    st.dataframe(df.head())

    st.write("📌 Columns detected:", list(df.columns))

    # Column selection (prevents KeyError)
    question_col = st.selectbox("Select Question Column", df.columns)
    answer_col = st.selectbox("Select Answer Column", df.columns)

    if st.button("🚀 Start Upload"):

        if not openai_key or not pinecone_key:
            st.warning("⚠️ Please enter both API keys")
        else:
            try:
                client, index = init_clients()

                upload_data(
                    df,
                    client,
                    index,
                    namespace,
                    question_col,
                    answer_col
                )

                st.subheader("📊 Pinecone Stats")
                stats = index.describe_index_stats()
                st.json(stats)

            except Exception as e:
                st.error(f"❌ Error: {e}")