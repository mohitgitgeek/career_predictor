from flask import Flask, render_template, request
from model import CareerPredictor
import os

app = Flask(__name__)
predictor = CareerPredictor()

@app.route('/')
def index():
    """Render the home page with the input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process form data and render prediction results"""
    try:
        # Extract form data
        o_score = float(request.form['o_score'])
        c_score = float(request.form['c_score'])
        e_score = float(request.form['e_score'])
        a_score = float(request.form['a_score'])
        n_score = float(request.form['n_score'])
        numerical = float(request.form['numerical'])
        spatial = float(request.form['spatial'])
        perceptual = float(request.form['perceptual'])
        abstract = float(request.form['abstract'])
        verbal = float(request.form['verbal'])
        
        # Additional inputs
        yoe = int(request.form['yoe']) if request.form['yoe'] else 0
        coding_rank = request.form['coding_rank']
        favorite_domain = request.form['favorite_domain']
        target_company = request.form['target_company']
        
        # Get disciplines
        disciplines = []
        if request.form['disciplines']:
            disciplines = [d.strip() for d in request.form['disciplines'].split(',')]
        
        # Combine all features
        features = [o_score, c_score, e_score, a_score, n_score, 
                   numerical, spatial, perceptual, abstract, verbal]
        
        # Get prediction
        result = predictor.predict_career(
            features=features,
            yoe=yoe,
            competitive_coding_rank=coding_rank,
            favorite_domain=favorite_domain,
            familiar_disciplines=disciplines,
            target_company=target_company
        )
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=f"Error processing your request: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
