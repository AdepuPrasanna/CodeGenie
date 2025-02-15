import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
# 🔑 Replace with your actual Gemini API Key 
GEMINI_API_KEY = "AIzaSyA3u0LSIAADtLLBRjf6RvyMigX5j-YYZxU"
genai.configure(api_key=GEMINI_API_KEY)

# Load CodeLlama model
MODEL_NAME = "codellama/CodeLlama-7b-hf"  # Change if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
try:
    st.sidebar.write(f"✅ Model: {MODEL_NAME} ({device})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto"
    )
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {e}")
    st.stop()

# CodeGenie Class
class CodeGenie:
    def __init__(self):
        """Initialize CodeGenie AI"""
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate_code(self, prompt):
        """Generate code using CodeLlama"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_length=200)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            return f"❌ Code Generation Failed: {str(e)}"

    def explain_code(self, code):
        """Explain the given code using Gemini AI"""
        try:
            response = genai.GenerativeModel("gemini-pro").generate_content(f"Explain this code:\n{code}")
            return response.text if response else "No explanation available."
        except Exception as e:
            return f"❌ Code Explanation Failed: {str(e)}"

    def debug_code(self, code):
        """Provide debugging suggestions using Gemini AI"""
        try:
            response = genai.GenerativeModel("gemini-pro").generate_content(f"Debug this code:\n{code}")
            return response.text if response else "No debugging suggestions available."
        except Exception as e:
            return f"❌ Code Debugging Failed: {str(e)}"

# Streamlit UI
st.title("🚀 CodeGenie: AI-Powered Code Generation")

genie = CodeGenie()

# User input
prompt = st.text_area("📝 Enter your prompt:", "Write a Python function to check if a number is prime.")
if st.button("✨ Generate Code"):
    generated_code = genie.generate_code(prompt)
    st.code(generated_code, language="python")

# Explain Code
if st.button("🔍 Explain Code"):
    explanation = genie.explain_code(generated_code)
    st.write("### 📖 Code Explanation")
    st.write(explanation)

# Debug Code
if st.button("🐞 Debug Code"):
    debug_suggestions = genie.debug_code(generated_code)
    st.write("### 🛠 Debugging Suggestions")
    st.write(debug_suggestions)

# Sidebar info
st.sidebar.subheader("ℹ️ About")
st.sidebar.write("**CodeGenie** is an AI-powered tool for code generation, explanation, and debugging using CodeLlama & Gemini AI.")
st.sidebar.write("🚀 Built with **Hugging Face Transformers, Google Gemini AI, and Streamlit.**")

