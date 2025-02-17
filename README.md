# Weather-Based-Outfit-Prediction-System

The WBOPS for short is an application that provides recommendations of what clothing
should be worn according to the source’s machine learning algorithm driven analysis of the
weather. Out of these, this model creates an optimized outfit that corresponds with the
expected weather conditions hence providing users with a useful tool for choosing the right
clothes to wear next. Through the use of the Random Forest Classifier, the model provides
high accurate outfit type predictions for several types of outfits. In this research, details of the
project mode of operation, the result obtained as well as the lessons learned from the
models have been provided herein.

# How to Run

## Step 1: Download All Files  
Ensure you have all the required files in your working directory.

## Step 2-5: Set Up and Run the Application  

Run the following commands step by step in your terminal:

```bash
# Step 1: Create and activate a virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # For macOS/Linux
# venv\Scripts\activate  # For Windows (Uncomment this if using Windows)

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the encoder models
python app.py

# Step 4: Start the web application
python web_app.py



