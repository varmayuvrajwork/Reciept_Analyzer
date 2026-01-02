import io
import os
import re
import base64
from typing import Dict
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import google.cloud.vision as vision
from google import genai
from crewai import Agent, Task, Crew

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__, template_folder="templates")
analyzed_data: Dict = {}

class GeminiLLM:
    def __init__(self, model="gemini-3-pro"):
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text

def vision_ai_ocr(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"OCR Error: {response.error.message}")

    return response.text_annotations[0].description if response.text_annotations else ""

def classify_receipt(ocr_text: str) -> str:
    if re.search(r"subtotal|total|item|price", ocr_text, re.IGNORECASE):
        return "Shopping"
    elif re.search(r"parking|duration|paid|time", ocr_text, re.IGNORECASE):
        return "Parking"
    return "Miscellaneous"

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
    details = []

    for line in lines:
        if ":" in line:
            details.append(line.strip())

    return "\n".join(details) if details else "No structured data found."

gemini_llm = GeminiLLM(model="gemini-3-pro")

shopping_agent = Agent(
    role="Shopping Receipt Analyst",
    goal="Extract structured shopping receipt information",
    backstory="Expert in retail receipts and invoices",
    llm=gemini_llm,          
    allow_delegation=False
)

parking_agent = Agent(
    role="Parking Receipt Analyst",
    goal="Extract parking receipt details",
    backstory="Expert in parking and transport receipts",
    llm=gemini_llm,          
    allow_delegation=False
)

misc_agent = Agent(
    role="Misc Receipt Analyst",
    goal="Extract generic receipt information",
    backstory="Expert in unstructured receipts",
    llm=gemini_llm,          
    allow_delegation=False
)

qa_agent = Agent(
    role="Receipt Q&A Agent",
    goal="Answer questions strictly from receipt text",
    backstory="Accurate assistant that never hallucinates",
    llm=gemini_llm,          
    allow_delegation=False
)

def run_receipt_analysis(receipt_type: str, ocr_text: str) -> str:
    if receipt_type == "Shopping":
        base_analysis = analyze_shopping_receipt(ocr_text)
        agent = shopping_agent
    elif receipt_type == "Parking":
        base_analysis = analyze_parking_receipt(ocr_text)
        agent = parking_agent
    else:
        base_analysis = analyze_miscellaneous_receipt(ocr_text)
        agent = misc_agent

    task = Task(
        description=f"""
        Receipt Type: {receipt_type}

        OCR Text:
        {ocr_text}

        Pre-Extracted Data:
        {base_analysis}

        Improve, clean, and summarize the receipt data.
        """,
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    return str(crew.kickoff())

def run_qa(question: str, context: str) -> str:
    task = Task(
        description=f"""
        Receipt Data:
        {context}

        Question:
        {question}

        Answer ONLY from receipt data.
        """,
        agent=qa_agent
    )

    crew = Crew(
        agents=[qa_agent],
        tasks=[task],
        verbose=False
    )

    return str(crew.kickoff())

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
        "agent_analysis": agent_output
    }

    return (
        f"Receipt analyzed successfully.<br>"
        f"<img src='data:image/jpeg;base64,{encoded_image}' "
        f"style='width:300px;height:auto;' />",
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

if __name__ == "__main__":
    app.run(debug=True)
