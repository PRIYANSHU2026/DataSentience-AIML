"""
HealthAI Pro - Main Application
Advanced Medical Diagnosis Platform with Intelligent Symptom Analysis
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from functools import wraps

from flask import Flask, request, render_template, redirect, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import numpy as np
import pandas as pd
import pickle
import bcrypt
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'healthai-pro-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model with improved security
class User(UserMixin, db.Model):
    """Enhanced user model with security features."""
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def __init__(self, name: str, email: str, password: str):
        self.name = name
        self.email = email
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Check password with secure hashing."""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login = datetime.utcnow()
        db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    return User.query.get(int(user_id))

# HealthAI Pro Diagnosis Engine
class DiagnosisEngine:
    """Advanced medical diagnosis engine with intelligent symptom analysis."""
    
    def __init__(self):
        """Initialize the diagnosis engine."""
        self.symptoms_dict = self._load_symptoms_dict()
        self.diseases_list = self._load_diseases_list()
        self.model = self._load_model()
        self.datasets = self._load_datasets()
        
        logger.info("DiagnosisEngine initialized successfully")
    
    def _load_symptoms_dict(self) -> Dict[str, int]:
        """Load symptoms dictionary."""
        return {
            'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
            'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
            'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
            'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
            'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
            'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
            'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
            'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
            'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
            'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
            'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
            'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
            'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
            'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
            'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
            'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
            'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
            'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
            'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
            'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
            'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
            'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
            'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
            'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
            'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91,
            'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
            'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
            'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
            'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
            'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
            'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
            'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
            'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
            'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
            'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
            'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
            'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
            'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
        }
    
    def _load_diseases_list(self) -> Dict[int, str]:
        """Load diseases list."""
        return {
            15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
            14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
            17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
            30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
            28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
            40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
            22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold',
            34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack',
            39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
            25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
            0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
            38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
        }
    
    def _load_model(self):
        """Load the trained machine learning model."""
        try:
            model_path = 'models/svc.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("ML model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load medical datasets."""
        try:
            datasets = {
                'sym_des': pd.read_csv("datasets/symtoms_df.csv"),
                'precautions': pd.read_csv("datasets/precautions_df.csv"),
                'workout': pd.read_csv("datasets/workout_df.csv"),
                'description': pd.read_csv("datasets/description.csv"),
                'medications': pd.read_csv('datasets/medications.csv'),
                'diets': pd.read_csv("datasets/diets.csv")
            }
            logger.info("Medical datasets loaded successfully")
            return datasets
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
    
    def predict_disease(self, symptoms: List[str]) -> Tuple[str, float]:
        """
        Predict disease based on symptoms with confidence score.
        
        Args:
            symptoms: List of symptoms
            
        Returns:
            Tuple of (predicted_disease, confidence_score)
        """
        try:
            # Validate symptoms
            valid_symptoms = [s for s in symptoms if s in self.symptoms_dict]
            if not valid_symptoms:
                raise ValueError("No valid symptoms provided")
            
            # Create input vector
            input_vector = np.zeros(len(self.symptoms_dict))
            for symptom in valid_symptoms:
                input_vector[self.symptoms_dict[symptom]] = 1
            
            # Make prediction
            prediction = self.model.predict([input_vector])[0]
            confidence = self.model.predict_proba([input_vector])[0].max()
            
            predicted_disease = self.diseases_list[prediction]
            
            logger.info(f"Disease predicted: {predicted_disease} with confidence: {confidence:.2f}")
            return predicted_disease, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_disease_info(self, disease: str) -> Dict[str, any]:
        """
        Get comprehensive disease information.
        
        Args:
            disease: Disease name
            
        Returns:
            Dictionary with disease information
        """
        try:
            # Get description
            desc = self.datasets['description'][self.datasets['description']['Disease'] == disease]['Description']
            description = " ".join([w for w in desc]) if not desc.empty else "Description not available"
            
            # Get precautions
            prec = self.datasets['precautions'][self.datasets['precautions']['Disease'] == disease]
            precautions = []
            if not prec.empty:
                precautions = [col for col in prec[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0] if pd.notna(col)]
            
            # Get medications
            med = self.datasets['medications'][self.datasets['medications']['Disease'] == disease]['Medication']
            medications = [med for med in med.values] if not med.empty else []
            
            # Get diet recommendations
            diet = self.datasets['diets'][self.datasets['diets']['Disease'] == disease]['Diet']
            diet_recommendations = [diet for diet in diet.values] if not diet.empty else []
            
            # Get workout recommendations
            workout = self.datasets['workout'][self.datasets['workout']['Disease'] == disease]['Workout']
            workout_recommendations = [workout for workout in workout.values] if not workout.empty else []
            
            return {
                'description': description,
                'precautions': precautions,
                'medications': medications,
                'diet': diet_recommendations,
                'workout': workout_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting disease info: {e}")
            return {
                'description': "Information not available",
                'precautions': [],
                'medications': [],
                'diet': [],
                'workout': []
            }
    
    def get_available_symptoms(self) -> List[str]:
        """Get list of available symptoms."""
        return list(self.symptoms_dict.keys())
    
    def get_available_diseases(self) -> List[str]:
        """Get list of available diseases."""
        return list(self.diseases_list.values())

# Initialize diagnosis engine
diagnosis_engine = DiagnosisEngine()

# Security decorators
def login_required_custom(f):
    """Custom login required decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

# Routes
@app.route("/")
def root():
    """Root route with proper redirection."""
    if current_user.is_authenticated:
        return redirect('/dashboard')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with validation."""
    if request.method == 'POST':
        try:
            name = request.form['name'].strip()
            email = request.form['email'].strip().lower()
            password = request.form['password']
            c_password = request.form['c_password']
            
            # Validation
            if not all([name, email, password]):
                flash('All fields are required.', 'error')
                return render_template('register.html')
            
            if password != c_password:
                flash('Passwords do not match.', 'error')
                return render_template('register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'error')
                return render_template('register.html')
            
            # Check if user already exists
            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'error')
                return render_template('register.html')
            
            # Create new user
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect('/login')
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with security features."""
    if request.method == 'POST':
        try:
            email = request.form['email'].strip().lower()
            password = request.form['password']
            
            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password):
                login_user(user)
                user.update_last_login()
                flash(f'Welcome back, {user.name}!', 'success')
                return redirect('/dashboard')
            else:
                flash('Invalid email or password.', 'error')
                return render_template('login.html')
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login failed. Please try again.', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect('/login')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard."""
    return render_template('dashboard.html', name=current_user.name)

@app.route('/diagnose')
@login_required
def diagnose():
    """Diagnosis page."""
    symptoms = diagnosis_engine.get_available_symptoms()
    return render_template('diagnosis.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Disease prediction endpoint."""
    try:
        symptoms_input = request.form.get('symptoms', '').strip()
        
        if not symptoms_input or symptoms_input == "Symptoms":
            flash('Please enter valid symptoms.', 'error')
            return redirect('/diagnose')
        
        # Parse symptoms
        user_symptoms = [s.strip() for s in symptoms_input.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        
        # Validate symptoms
        valid_symptoms = [s for s in user_symptoms if s in diagnosis_engine.symptoms_dict]
        if not valid_symptoms:
            flash('No valid symptoms provided. Please check your input.', 'error')
            return redirect('/diagnose')
        
        # Make prediction
        predicted_disease, confidence = diagnosis_engine.predict_disease(valid_symptoms)
        disease_info = diagnosis_engine.get_disease_info(predicted_disease)
        
        # Log prediction for analytics
        logger.info(f"Prediction made by {current_user.email}: {predicted_disease} (confidence: {confidence:.2f})")
        
        return render_template('diagnosis.html',
                             predicted_disease=predicted_disease,
                             confidence=f"{confidence:.1%}",
                             disease_info=disease_info,
                             symptoms=diagnosis_engine.get_available_symptoms())
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        flash('An error occurred during diagnosis. Please try again.', 'error')
        return redirect('/diagnose')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for disease prediction."""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        predicted_disease, confidence = diagnosis_engine.predict_disease(symptoms)
        disease_info = diagnosis_engine.get_disease_info(predicted_disease)
        
        return jsonify({
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'disease_info': disease_info
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page."""
    return render_template('contact.html')

@app.route('/blog')
def blog():
    """Blog page."""
    return render_template('blog.html')

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)