import streamlit as st
import os
import json
from mistralai import Mistral
from scripts.prompts import PROMPTS
from dotenv import load_dotenv


load_dotenv()

# Load API key
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
FINE_TUNED_MODEL_ID = os.environ.get("FINE_TUNED_MODEL_ID")

# Initialize client
client = Mistral(api_key=MISTRAL_API_KEY)

# Model selector
st.title("📜 CUAD Clause Classifier & Summarizer")
model_type = st.selectbox("Select model", ["Base (ministral-8b-2410)", "Fine-tuned"])

# Text input
user_input = st.text_area("Paste a paragraph from a contract:", height=200)

# Generate response
if st.button("🔍 Analyze Clause") and user_input:
    with st.spinner("Analyzing clause..."):

        if model_type == "Base (ministral-8b-2410)":
            model = "ministral-8b-2410"
        else:
            model = FINE_TUNED_MODEL_ID

        prompt = [
            {"role": "system", "content": PROMPTS["system_classification_summary"]},
            {"role": "user", "content": f"Paragraph: {user_input}"}
        ]

        try:
            response = client.chat.complete(
                model=model,
                messages=prompt,
                response_format={"type": "json_object"}
            )
            content = json.loads(response.choices[0].message.content)
            st.success("Clause identified and summarized!")
            st.json(content)
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Developed for the Mistral Applied AI Engineer Take-Home Assignment.")