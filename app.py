import os
from flask import Flask, request, render_template
from src.project.components.data_processing import DataProcessing
from src.AppFunctions.resume_extraction import *

app = Flask(__name__)

@app.route('/')
def resume():
    return render_template("resume.html")


@app.route('/pred', methods=['POST'])
def pred():

    if 'resume' in request.files:
        file = request.files['resume']
        text = extract_text_from_file(file)
        
        if text is None:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")
        
        data_processor = DataProcessing()
        cleaned_text = data_processor.clean_text(text)
        
        model, tfidf_vectorizer = load_model_and_vectorizer()
        predicted_category = make_prediction(cleaned_text, model, tfidf_vectorizer)

        email = extract_email_from_resume(text)
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)

        return render_template('resume.html', predicted_category=predicted_category,
                               email=email, extracted_skills=extracted_skills,
                               extracted_education=extracted_education)
    else:
        return render_template('resume.html', message="No file uploaded. Please upload a resume file")
if __name__ == "__main__":
    app.run(debug=True)
