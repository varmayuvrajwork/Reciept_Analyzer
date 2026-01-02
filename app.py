import io
import os
import re
import base64
from typing import Dict

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

import google.cloud.vision as vision

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------------------------------------------------
# ENV SETUP
# -------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required")

app = Flask(__name__, template_folder="templates")
analyzed_data: Dict = {}

# -------------------------------------------------
# GEMINI LLM (LANGCHAIN)
# -------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

# -------------------------------------------------
# GOOGLE VISION OCR
# -------------------------------------------------
def vision_ai_ocr(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"OCR Error: {response.error.message}")

    return response.text_annotations[0].description if response.text_annotations else ""

# -------------------------------------------------
# RECEIPT CLASSIFICATION
# -------------------------------------------------
def classify_receipt(ocr_text: str) -> str:
    if re.search(r"subtotal|total|item|price", ocr_text, re.IGNORECASE):
        return "Shopping"
    elif re.search(r"parking|duration|paid|time", ocr_text, re.IGNORECASE):
        return "Parking"
    return "Miscellaneous"

# -------------------------------------------------
# DOMAIN EXTRACTION LOGIC (PRE-LLM)
# -------------------------------------------------
def analyze_shopping_receipt(ocr_text: str) -> str:
    lines = ocr_text.split("\n")
    items = []

    for i in range(len(lines) - 1):
        if re.match(r"^\£\d+\.\d{2}$", lines[i + 1]):
            items.append(f"{lines[i].strip()}: {lines[i + 1].strip()}")

    return "\n".join(items) if items else "No structured shopping data found."

def analyze_parking_receipt(ocr_text: str) -> str:
    lines = ocr_text.split("\n")
    info = {}

    for line in lines:
        if re.search(r"\d{2}/\d{2}/\d{2,4}", line):
            info["Date"] = line.strip()
        elif re.search(r"\d{2}:\d{2}", line):
            info["Time"] = line.strip()
        elif re.search(r"[\d\.]+ hours", line, re.IGNORECASE):
            info["Duration"] = line.strip()
        elif re.search(r"\£\d+\.\d{2}", line):
            info["Amount Paid"] = line.strip()

    return "\n".join(f"{k}: {v}" for k, v in info.items())

def analyze_miscellaneous_receipt(ocr_text: str) -> str:
    lines = ocr_text.split("\n")
    details = [line.strip() for line in lines if ":" in line]
    return "\n".join(details) if details else "No structured data found."

# -------------------------------------------------
# LANGCHAIN PROMPTS
# -------------------------------------------------
receipt_analysis_prompt = PromptTemplate(
    input_variables=["receipt_type", "ocr_text", "pre_data"],
    template="""
You are an expert receipt analyst.

Receipt Type: {receipt_type}

OCR Text:
{ocr_text}

Pre-extracted Data:
{pre_data}

Clean, improve, and summarize the receipt clearly for a human.
"""
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a precise assistant.

Receipt Data:
{context}

Question:
{question}

Answer ONLY using the receipt data.
"""
)

receipt_chain = LLMChain(llm=llm, prompt=receipt_analysis_prompt)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# -------------------------------------------------
# ANALYSIS PIPELINES
# -------------------------------------------------
def run_receipt_analysis(receipt_type: str, ocr_text: str) -> str:
    if receipt_type == "Shopping":
        pre_data = analyze_shopping_receipt(ocr_text)
    elif receipt_type == "Parking":
        pre_data = analyze_parking_receipt(ocr_text)
    else:
        pre_data = analyze_miscellaneous_receipt(ocr_text)

    result = receipt_chain.invoke({
        "receipt_type": receipt_type,
        "ocr_text": ocr_text,
        "pre_data": pre_data
    })
    return result["text"]


def run_qa(question: str, context: str) -> str:
    result = qa_chain.invoke({
        "context": context,
        "question": question
    })
    return result["text"]


# -------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze-receipt", methods=["POST"])
def analyze_receipt():
    global analyzed_data

    if "receipt" not in request.files:
        return jsonify({"error": "No receipt uploaded"}), 400

    receipt = request.files["receipt"]
    receipt_path = "uploaded_receipt.jpg"
    receipt.save(receipt_path)

    with open(receipt_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    ocr_text = vision_ai_ocr(receipt_path)
    receipt_type = classify_receipt(ocr_text)

    agent_output = run_receipt_analysis(receipt_type, ocr_text)

    analyzed_data = {
        "full_text": ocr_text,
        "type": receipt_type,
        "analysis": agent_output
    }

    return (
        f"Receipt analyzed successfully.<br>"
        f"<img src='data:image/jpeg;base64,{encoded_image}' style='width:300px;height:auto;' />",
        200
    )

@app.route("/ask-question", methods=["POST"])
def ask_question():
    if not analyzed_data:
        return jsonify({"error": "Analyze a receipt first"}), 400

    question = request.json.get("question")
    if not question:
        return jsonify({"error": "Question required"}), 400

    answer = run_qa(question, analyzed_data["full_text"])
    return answer, 200

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
