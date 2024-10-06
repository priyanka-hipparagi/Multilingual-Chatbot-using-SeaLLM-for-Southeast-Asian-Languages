import transformers
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SeaLLMChatbot:
    def _init_(self):
        self.model_id = "aisingapore/llama3-8b-cpt-SEA-Lionv2.1-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate_response(self, prompt):
        outputs = self.pipeline(
            [{"role": "user", "content": prompt}],
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"]

def main():
    st.title("SeaLLM Chatbot for Southeast Asian Languages")

    chatbot = SeaLLMChatbot()

    task = st.selectbox(
        "Select a task",
        ["Translation", "Sentiment Analysis", "Named Entity Recognition", "Question Answering"]
    )

    if task == "Translation":
        source_lang = st.selectbox("Source Language", ["English", "Indonesian", "Vietnamese", "Thai", "Tagalog", "Malay"])
        target_lang = st.selectbox("Target Language", ["English", "Indonesian", "Vietnamese", "Thai", "Tagalog", "Malay"])
        text = st.text_area("Enter text to translate:")
        if st.button("Translate"):
            prompt = f"Translate from {source_lang} to {target_lang}: {text}\nTranslation: "
            response = chatbot.generate_response(prompt)
            st.write("Translation:", response)

    elif task == "Sentiment Analysis":
        lang = st.selectbox("Language", ["English", "Indonesian", "Vietnamese", "Thai", "Tagalog", "Malay"])
        text = st.text_area("Enter text for sentiment analysis:")
        if st.button("Analyze Sentiment"):
            prompt = f"Analyze the sentiment of the following {lang} text: {text}\nSentiment: "
            response = chatbot.generate_response(prompt)
            st.write("Sentiment Analysis:", response)

    elif task == "Named Entity Recognition":
        lang = st.selectbox("Language", ["English", "Indonesian", "Vietnamese", "Thai", "Tagalog", "Malay"])
        text = st.text_area("Enter text for named entity recognition:")
        if st.button("Extract Entities"):
            prompt = f"Extract named entities from the following {lang} text: {text}\nEntities: "
            response = chatbot.generate_response(prompt)
            st.write("Named Entities:", response)

    elif task == "Question Answering":
        lang = st.selectbox("Language", ["English", "Indonesian", "Vietnamese", "Thai", "Tagalog", "Malay"])
        context = st.text_area("Enter context:")
        question = st.text_input("Enter question:")
        if st.button("Answer"):
            prompt = f"Context in {lang}: {context}\nQuestion: {question}\nAnswer: "
            response = chatbot.generate_response(prompt)
            st.write("Answer:", response)

if _name_ == "_main_":
    main()