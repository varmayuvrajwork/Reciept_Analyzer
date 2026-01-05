import google.cloud.vision as vision
import io
import pandas as pd
import numpy as np
from langchain_community.llms import openai
from langchain.agents import Tool, initialize_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Optional, Dict
import os
import
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")
analyzed_data = {}
openai_api_key = os.getenv('OPENAI_API_KEY')  
google_application_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def vision_ai_ocr(image_path):
      client = vision.ImageAnnotatorClient()
      with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
      image = vision.Image(content=content)
      response = client.text_detection(image=image)
      if response.error.message:
            raise Exception(f"Error during OCR: {response.error.message}")
      return response.text_annotations[0].description if response.text_annotations else ""

prompts: Dict[str, PromptTemplate] = {
      "Shopping": PromptTemplate.from_template("You are analyzing a shopping receipt. Extract product names, prices, and other relevant details."),
      "Parking": PromptTemplate.from_template("You are analyzing a parking receipt. Extract date, time, duration, and payment information."),
      "Miscellaneous": PromptTemplate.from_template("You are analyzing a miscellaneous receipt. Extract as much relevant information as possible."),
      }

def classify_receipt(ocr_text):
      if re.search(r"subtotal|total|item|price", ocr_text, re.IGNORECASE):
            return "Shopping"
      elif re.search(r"parking|duration|paid|time", ocr_text, re.IGNORECASE):
            return "Parking"
      else:
            return "Miscellaneous"

def analyze_shopping_receipt(ocr_text):
      lines = ocr_text.split('\n')
      items, prices = [], []
      for i in range(len(lines) - 1):
            if re.match(r"^£\d+\.\d{2}$", lines[i + 1]):
                  items.append(lines[i].strip())
                  prices.append(lines[i + 1].replace("₤", "£"))
      if items and prices:
            return "\n".join([f"{item}: {price}" for item, price in zip(items, prices)])
      else:
            return None

def analyze_parking_receipt(ocr_text):
      lines = ocr_text.split('\n')
      parking_info = {
            "Date": None,
            "Time": None,
            "Duration": None,
            "Amount Paid": None
      }
      for line in lines:
            if re.search(r"\d{2}/\d{2}/\d{2,4}", line):
                  parking_info["Date"] = line.strip()
            elif re.search(r"\d{2}:\d{2}", line):
                  parking_info["Time"] = line.strip()
            elif re.search(r"[\d\.]+ hours", line, re.IGNORECASE):
                  parking_info["Duration"] = line.strip()
            elif re.search(r"£\d+\.\d{2}", line):
                  parking_info["Amount Paid"] = line.strip()
      return "\n".join([f"{key}: {value}" for key, value in parking_info.items() if value])

def analyze_miscellaneous_receipt(ocr_text):
      lines = ocr_text.split("\n")
      details = {}
      for line in lines:
            if ":" in line:
                  key, value = map(str.strip, line.split(":", 1))
                  details[key] = value
            elif "-" in line:
                  key, value = map(str.strip, line.split("-", 1))
                  details[key] = value
            else:
                  details.setdefault("Other Details", []).append(line.strip())

      if "Other Details" in details:
            details["Other Details"] = " ".join(details["Other Details"])
      return {"type": "Miscellaneous", "details": details}

class ShoppingReceiptTool(BaseTool):
      name: str = "Shopping_Receipt"
      description: str = "Analyze shopping receipts."

      def _run(self, image_path: str) -> str:
            prompt = prompts["Shopping"].format()
            ocr_text = vision_ai_ocr(image_path)
            return analyze_shopping_receipt(ocr_text)

      async def _arun(self, image_path: str) -> str:
            raise NotImplementedError("This tool does not support async execution.")

class ParkingReceiptTool(BaseTool):
      name: str = "Parking_Receipt"
      description: str = "Analyze parking receipts."

      def _run(self, image_path: str) -> str:
            prompt = prompts["Parking"].format()
            ocr_text = vision_ai_ocr(image_path)
            return analyze_parking_receipt(ocr_text)

      async def _arun(self, image_path: str) -> str:
            raise NotImplementedError("This tool does not support async execution.")

class MiscellaneousTool(BaseTool):
      name: str = "Miscellaneous"
      description: str = "Analyze miscellaneous receipts."

      def _run(self, image_path: str) -> str:
            prompt = prompts["Miscellaneous"].format()
            ocr_text = vision_ai_ocr(image_path)
            return analyze_miscellaneous_receipt(ocr_text)

      async def _arun(self, image_path: str) -> str:
            raise NotImplementedError("This tool does not support async execution.")
      
tools = [
      ShoppingReceiptTool(),
      ParkingReceiptTool(),
      MiscellaneousTool()
      ]

agent = initialize_agent(tools, llm=llm, agent="zero-shot-react-description", verbose=True)

@app.route('/')
def index():
      return render_template('index.html')

@app.route('/analyze-receipt', methods=['POST'])
def analyze_receipt():
      global analyzed_data
      try:
            if 'receipt' not in request.files:
                  return jsonify({"error": "No receipt image uploaded."}), 400

            receipt_image = request.files['receipt']
            receipt_path = "uploaded_receipt.jpg"
            receipt_image.save(receipt_path)
            result = agent.run({"input": f"Analyze this receipt: {receipt_path}"})
            analyzed_data = {"final_answer": result.strip()}
            return jsonify({"result": analyzed_data}), 200

      except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/ask-question', methods=['POST'])
def ask_question():
      global analyzed_data
      try:
            data = request.get_json()
            question = data.get('question')

            if not question:
                  return jsonify({"error": "Invalid input. 'question' is required."}), 400

            if not analyzed_data or "final_answer" not in analyzed_data:
                  return jsonify({"error": "No analyzed data available. Please analyze a receipt first."}), 400

            prompt_template = PromptTemplate(
                  input_variables=["context", "question"],
                  template="Given the following receipt data: {context}, answer the question: {question}"
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            context = analyzed_data["final_answer"]
            answer = chain.run({"context": context, "question": question})
            return jsonify({"response": answer}), 200

      except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
      app.run(debug=True)
