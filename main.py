from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from transformers import pipeline
from twilio.rest import Client
import os
import random
import json
import secrets
app = Flask(__name__)
app.secret_key =  secrets.token_hex(32)
api_key = os.getenv("HF_API_KEY")

# Using Facebook's BART for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load questions from JSON file
def load_questions():
    with open('questions.json', 'r') as file:
        questions = json.load(file)
        return questions


def send_sms(message):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")       #SET YOUR TWILIO ACCOUNT SID IN THE ENV VARIABLE
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")         #SET YOUR TWILIO ACCOUNT AUTHENTICATION TOKEN IN ENV VARIABLE
    client = Client(account_sid, auth_token)

    client.messages.create(
        body=message,
        from_="+19895205746",
        to="+918987411450"
    )

@app.route('/')
def load():
    return render_template("index.html")

@app.route('/survey')
def load_form():
    return render_template("survey.html")
    

@app.route('/get_questions', methods=['GET'])
def get_questions():
    questions = load_questions()
    selected_questions = random.sample(questions, 5)
    for i, question in enumerate(selected_questions, start=1):
        question['local_id'] = i
    session['selected_questions'] = selected_questions
    return jsonify(selected_questions)

@app.route('/submit', methods=['POST'])
def submit_answers():
    data = request.get_json()
    selected_questions = session.get('selected_questions', [])

    # Map question ids to their text
    questions_dict = {str(q['local_id']): q['question'] for q in selected_questions}

    mapped_answers = []
    for qid, answer in data['answers'].items():
        if qid in questions_dict:
            question_text = questions_dict[qid]
            mapped_answers.append(f"Q: {question_text} A: {answer}")

   
    text_to_analyze = "\n".join(mapped_answers)

  
    labels = ["High Stress", "Moderate Stress", "Low Stress", "Good Mental Health"]

    prompt = f"""
    The following are responses from a mental health questionnaire. 
    Analyze the responses and determine the level of distress. 
    If the user appears to be in high distress, include the phrase "DISTRESS_ALERT" in the response.

    Responses:
    {text_to_analyze}

    Insights:
    """

    # Perform zero-shot classification using BART
    classification_result = classifier(text_to_analyze, candidate_labels=labels)

    # Extract the most relevant category and confidence score
    best_category = classification_result["labels"][0]
    confidence_score = classification_result["scores"][0]

    # Generate insights based on classification
    insights = f"Your responses indicate a {best_category} level with a confidence of {confidence_score:.2f}."

    if best_category in ["High Stress", "Moderate Stress"]:
        insights = f"DISTRESS_ALERT: {insights}"
        #send_sms("Alert: a user is experiencing distress.Please consult her. \n -Team CelestHer")

    print("Generated Insight:", insights)

    
    session['insights'] = insights

    
    return redirect(url_for('result'))

@app.route('/result')
def result():
    insights = session.get('insights', 'No insights available')
    print("Insights:", insights) 
    return render_template('result.html', insights=insights)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
