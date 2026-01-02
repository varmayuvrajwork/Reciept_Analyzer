import cv2
import numpy as np
from pytesseract import pytesseract, Output
from PIL import Image
import pandas as pd
import os
import
def detect_text_regions_tesseract(image):
      """
      Detect text regions using Tesseract OCR.

      Args:
            image (np.array): Input image.

      Returns:
            list: Bounding boxes of detected text regions.
      """
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      d = pytesseract.image_to_data(gray, output_type=Output.DICT)
      n_boxes = len(d['level'])
      boxes = []

      for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if int(d['conf'][i]) > 60:  # Filter by confidence level
                  boxes.append((x, y, w, h))

      return boxes

def extract_text_from_regions_tesseract(image, boxes):
      """
      Extract text from detected regions using Tesseract OCR.

      Args:
            image (np.array): Input image.
            boxes (list): List of bounding boxes containing text.

      Returns:
            list: List of recognized text strings.
      """
      recognized_texts = []
      for box in boxes:
            x, y, w, h = box
            cropped = image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped, config='--psm 6')
            recognized_texts.append(text.strip())
      return recognized_texts


image_path = "/Users/c100-122/ocr/sign_small.jpg"
image = cv2.imread(image_path)
if image is None:
      raise FileNotFoundError(f"Image not found at path: {image_path}")
boxes = detect_text_regions_tesseract(image)

if boxes:
      extracted_texts = extract_text_from_regions_tesseract(image, boxes)
      print("Extracted Texts:", extracted_texts)
else:
      print("No text detected in the image.")
