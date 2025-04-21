import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CareerPredictor:
    def __init__(self, data_path="career_predictor_app\Data_final.csv"):
        """Initialize the Career Predictor using Decision Tree algorithm"""
        # Load and process data
        self.data = pd.read_csv(data_path)
        self.feature_cols = [
            'O_score', 'C_score', 'E_score', 'A_score', 'N_score',
            'Numerical Aptitude', 'Spatial Aptitude', 'Perceptual Aptitude',
            'Abstract Reasoning', 'Verbal Reasoning'
        ]
        
        # Prepare data for model training
        X = self.data[self.feature_cols].values
        y = self.data['Career'].values
        
        # Scale features for better performance
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Decision Tree model
        self.model = DecisionTreeClassifier(
            max_depth=5,           # Prevent overfitting
            min_samples_split=2,   # Minimum samples required to split
            min_samples_leaf=1,    # Minimum samples required at leaf node
            random_state=42        # For reproducibility
        )
        self.model.fit(X_scaled, y)
        
        # Domain-career mappings
        self.domain_career_mapping = {
            'technology': ['Software Developer', 'Web Developer', 'Data Scientist', 
                          'IT Support Specialist', 'Database Administrator', 'Game Developer'],
            'healthcare': ['Nurse', 'Physician', 'Physical Therapist', 'Pediatric Nurse',
                          'Pediatrician', 'Chiropractor', 'Rehabilitation Counselor'],
            'finance': ['Accountant', 'Financial Analyst', 'Financial Planner', 
                       'Financial Advisor', 'Investment Banker', 'Tax Accountant'],
            'creative': ['Graphic Designer', 'Artist', 'Fashion Designer', 
                        'Interior Designer', 'Musician', 'Film Director'],
            'engineering': ['Mechanical Engineer', 'Electrical Engineer', 'Civil Engineer',
                           'Aerospace Engineer', 'Biomedical Engineer', 'Robotics Engineer']
        }
        
        # Company to domain mappings
        self.company_domain_mapping = {
            'google': 'technology',
            'microsoft': 'technology',
            'apple': 'technology',
            'amazon': 'technology',
            'meta': 'technology',
            'jpmorgan': 'finance',
            'goldman sachs': 'finance',
            'morgan stanley': 'finance',
            'pfizer': 'healthcare',
            'johnson & johnson': 'healthcare',
            'mayo clinic': 'healthcare',
            'boeing': 'engineering',
            'lockheed martin': 'engineering',
            'disney': 'creative',
            'sony': 'creative'
        }
        
        # Careers requiring competitive coding
        self.coding_intensive_careers = [
            'Software Developer', 'Data Scientist', 'Game Developer',
            'Web Developer', 'Database Administrator'
        ]
        
        # Alternative companies
        self.alternative_companies = {
            'low_coding_rank': [
                'Startups focusing on product development',
                'Non-tech companies with tech departments',
                'Government tech agencies',
                'Healthcare IT companies',
                'Educational technology companies'
            ],
            'low_yoe': [
                'Startups with training programs',
                'Companies with internship-to-job pipelines',
                'Organizations with mentorship programs',
                'Large corporations with structured entry programs',
                'Technology service providers'
            ]
        }
    
    def predict_career(self, features, yoe=0, competitive_coding_rank=None, 
                      favorite_domain=None, familiar_disciplines=None, target_company=None):
        """Predict careers using Decision Tree classifier and assess viability"""
        # Scale input features
        features_scaled = self.scaler.transform([features])
        
        # Use Decision Tree to predict probabilities for all careers
        career_probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get indices of top 5 career probabilities
        top_indices = np.argsort(career_probabilities)[-5:][::-1]
        
        # Get corresponding career names and their probabilities
        careers = self.model.classes_[top_indices]
        probabilities = career_probabilities[top_indices]
        
        # Create top careers list with probabilities
        top_careers = [{"career": career, "probability": prob} 
                      for career, prob in zip(careers, probabilities)]
        
        # Check if recommended careers are tech-focused
        tech_career_recommended = any(
            career["career"] in self.coding_intensive_careers 
            for career in top_careers[:3]
        )
        
        # Initialize result dictionary
        result = {
            'predicted_careers': [c["career"] for c in top_careers],
            'career_probabilities': [c["probability"] for c in top_careers],
            'alternative_companies': [],
            'viability_assessment': {}
        }
        
        # Check for limitations
        low_yoe = yoe < 2
        low_coding_rank = self._is_low_coding_rank(competitive_coding_rank)
        
        # Add alternative companies if needed
        if tech_career_recommended:
            if low_coding_rank:
                result['alternative_companies'].extend(
                    self.alternative_companies['low_coding_rank'])
            if low_yoe:
                result['alternative_companies'].extend(
                    self.alternative_companies['low_yoe'])
        
        # Assess domain match
        domain_match_score = 0.0
        if favorite_domain and favorite_domain.lower() in self.domain_career_mapping:
            domain_careers = set(self.domain_career_mapping[favorite_domain.lower()])
            matching_careers = [c for c in result['predicted_careers'] if c in domain_careers]
            domain_match_score = len(matching_careers) / min(len(result['predicted_careers']), len(domain_careers))
        
        # Assess company match
        company_match_score = 0.0
        if target_company and target_company.lower() in self.company_domain_mapping:
            company_domain = self.company_domain_mapping[target_company.lower()]
            if company_domain in self.domain_career_mapping:
                domain_careers = set(self.domain_career_mapping[company_domain])
                matching_careers = [c for c in result['predicted_careers'] if c in domain_careers]
                company_match_score = len(matching_careers) / min(len(result['predicted_careers']), len(domain_careers))
        
        # Calculate confidence based on Decision Tree probabilities
        # Higher probabilities indicate stronger model confidence
        confidence_score = result['career_probabilities'][0]  # Top career probability
        
        # Calculate viability score
        viability_factors = {
            'model_confidence': confidence_score,
            'domain_match': domain_match_score,
            'company_match': company_match_score,
            'experience': min(1.0, yoe / 3.0),
            'coding_skill': 0.0 if (tech_career_recommended and low_coding_rank) else 1.0
        }
        
        # Calculate weighted viability score
        weights = {
            'model_confidence': 0.3,
            'domain_match': 0.2,
            'company_match': 0.15,
            'experience': 0.2,
            'coding_skill': 0.15
        }
        
        viability_score = sum(score * weights[factor] for factor, score in viability_factors.items())
        
        # Determine viability level
        if viability_score >= 0.7:
            viability_level = "High"
            explanation = "Your profile strongly aligns with the predicted careers."
        elif viability_score >= 0.4:
            viability_level = "Moderate"
            explanation = "Your profile has moderate alignment with the predicted careers."
        else:
            viability_level = "Low"
            explanation = "Your profile has significant gaps compared to requirements."
        
        # Generate recommendations based on viability factors
        recommendations = self._generate_recommendations(
            viability_factors, favorite_domain, target_company, 
            familiar_disciplines, result['predicted_careers']
        )
        
        # Add viability assessment to result
        result['viability_assessment'] = {
            'score': viability_score,
            'level': viability_level,
            'explanation': explanation,
            'factors': viability_factors
        }
        
        result['recommendations'] = recommendations
        
        return result
    
    def _is_low_coding_rank(self, rank):
        """Determine if competitive coding rank is considered low"""
        if rank is None:
            return True
        
        rank_lower = rank.lower()
        
        # Platform-specific thresholds
        if 'codechef' in rank_lower:
            return not any(x in rank_lower for x in ['5*', '6*', '7*'])
        elif 'codeforces' in rank_lower:
            return not any(x in rank_lower for x in ['expert', 'master', 'grandmaster'])
        elif 'leetcode' in rank_lower:
            try:
                rating = int(''.join(filter(str.isdigit, rank_lower)))
                return rating < 2000
            except:
                return True
        elif 'hackerrank' in rank_lower:
            return not any(x in rank_lower for x in ['gold', 'platinum'])
        
        # Default to low rank if platform not recognized
        return True
    
    def _generate_recommendations(self, viability_factors, favorite_domain, 
                                target_company, familiar_disciplines, top_careers):
        """Generate personalized recommendations based on viability factors"""
        recommendations = []
        
        # Domain recommendations
        if favorite_domain and viability_factors['domain_match'] < 0.5:
            recommendations.append(
                f"Your profile shows limited alignment with the {favorite_domain} domain. " +
                f"Consider exploring careers that better match your personality and aptitude."
            )
        
        # Company recommendations
        if target_company and viability_factors['company_match'] < 0.5:
            company_domain = self.company_domain_mapping.get(target_company.lower(), None)
            if company_domain:
                alt_companies = [c for c, d in self.company_domain_mapping.items() 
                               if d == company_domain and c != target_company.lower()][:3]
                if alt_companies:
                    recommendations.append(
                        f"Your profile may not be ideal for {target_company}. " +
                        f"Consider similar companies: {', '.join(alt_companies)}."
                    )
        
        # Experience recommendations
        if viability_factors['experience'] < 0.5:
            recommendations.append(
                "With limited experience, focus on entry-level positions or " +
                "companies with strong training programs."
            )
        
        # Coding skill recommendations
        if viability_factors['coding_skill'] < 0.5:
            recommendations.append(
                "Your coding skills may need improvement for technical roles. " +
                "Consider focusing on companies that value other skills or invest time in competitive coding."
            )
        
        # Model confidence recommendations
        if viability_factors['model_confidence'] < 0.4:
            recommendations.append(
                "The decision tree model has lower confidence in these predictions. " +
                "Consider taking career assessment tests to further validate these recommendations."
            )
        
        return recommendations

def main():
    """Interactive command-line interface for the career predictor"""
    print("=== Decision Tree Career Predictor 2025 ===")
    
    # Collect personality scores
    print("\n=== Personality Traits (Rate 1-10) ===")
    o_score = float(input("Openness to experience: "))
    c_score = float(input("Conscientiousness: "))
    e_score = float(input("Extraversion: "))
    a_score = float(input("Agreeableness: "))
    n_score = float(input("Neuroticism: "))
    
    # Collect aptitude scores
    print("\n=== Aptitude Scores (Rate 1-10) ===")
    numerical = float(input("Numerical Aptitude: "))
    spatial = float(input("Spatial Aptitude: "))
    perceptual = float(input("Perceptual Aptitude: "))
    abstract = float(input("Abstract Reasoning: "))
    verbal = float(input("Verbal Reasoning: "))
    
    # Collect experience and coding rank
    print("\n=== Experience and Skills ===")
    yoe = int(input("Years of Experience: "))
    coding_rank = input("Competitive Coding Rank (e.g., '4* CodeChef'): ")
    if coding_rank.strip() == "":
        coding_rank = None
    
    # Collect domain, disciplines, and target company
    print("\n=== Career Preferences ===")
    favorite_domain = input("Favorite domain (technology, healthcare, finance, creative, engineering): ")
    if favorite_domain.strip() == "":
        favorite_domain = None
    
    disciplines = input("Familiar disciplines (comma-separated): ")
    familiar_disciplines = [d.strip() for d in disciplines.split(",")] if disciplines.strip() else None
    
    target_company = input("Target company: ")
    if target_company.strip() == "":
        target_company = None
    
    # Process and display results
    features = [o_score, c_score, e_score, a_score, n_score, 
               numerical, spatial, perceptual, abstract, verbal]
    
    predictor = CareerPredictor()
    result = predictor.predict_career(
        features=features, yoe=yoe, competitive_coding_rank=coding_rank,
        favorite_domain=favorite_domain, familiar_disciplines=familiar_disciplines,
        target_company=target_company
    )
    
    print("\n=== Career Predictions (Decision Tree) ===")
    for i, (career, prob) in enumerate(zip(result['predicted_careers'], 
                                          result['career_probabilities']), 1):
        print(f"{i}. {career} (Confidence: {prob:.2f})")
    
    print("\n=== Viability Assessment ===")
    print(f"Overall Viability: {result['viability_assessment']['level']} " +
          f"(Score: {result['viability_assessment']['score']:.2f})")
    print(result['viability_assessment']['explanation'])
    
    if result['recommendations']:
        print("\n=== Recommendations ===")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
    
    if result['alternative_companies']:
        print("\n=== Alternative Companies ===")
        for i, company in enumerate(result['alternative_companies'], 1):
            print(f"{i}. {company}")

if __name__ == "__main__":
    main()