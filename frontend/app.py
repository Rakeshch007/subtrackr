import streamlit as st
from ml.ocr_pipeline import extract_text_from_pdf

st.set_page_config(page_title="SubTrackr Prototype", layout="wide")
st.title("📊 SubTrackr - AI-Powered Subscription Manager (Week 4 Prototype)")

uploaded_file = st.file_uploader("Upload a bank statement (PDF)", type=["pdf"])
if uploaded_file:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing your statement...")
    extracted_text = extract_text_from_pdf("data/temp.pdf")

    st.subheader("📝 Extracted Transactions")
    st.text_area("Text Output", extracted_text, height=300)
