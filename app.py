import os
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import json
import io
import base64
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from classifiers import DiabetesClassifiers

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'diabetes_classification_secret_key_2024')

DATABASE = 'diabetes_app.db'
DATASET_PATH = 'Final_Dataset.csv'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS engineers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            address TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            address TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def login_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or session.get('role') != role:
                flash('Please log in to access this page.', 'error')
                return redirect(url_for('login', role=role))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def load_dataset():
    df = pd.read_csv(DATASET_PATH, sep=';')
    return df

def get_class_names():
    df = load_dataset()
    return sorted(df['classification'].unique().tolist())

classifier_instance = None

def get_classifier():
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = DiabetesClassifiers(DATASET_PATH)
    return classifier_instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register/<role>', methods=['GET', 'POST'])
def register(role):
    if role not in ['engineer', 'user']:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        address = request.form.get('address')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html', role=role)
        
        conn = get_db()
        cursor = conn.cursor()
        table = 'engineers' if role == 'engineer' else 'users'
        
        try:
            cursor.execute(f'''
                INSERT INTO {table} (name, mobile, email, address, password)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, mobile, email, address, generate_password_hash(password)))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login', role=role))
        except sqlite3.IntegrityError:
            flash('Email already exists!', 'error')
        finally:
            conn.close()
    
    return render_template('register.html', role=role)

@app.route('/login/<role>', methods=['GET', 'POST'])
def login(role):
    if role not in ['engineer', 'user']:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = get_db()
        cursor = conn.cursor()
        table = 'engineers' if role == 'engineer' else 'users'
        
        cursor.execute(f'SELECT * FROM {table} WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['role'] = role
            
            if role == 'engineer':
                return redirect(url_for('engineer_home'))
            else:
                return redirect(url_for('user_prediction'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('login.html', role=role)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/engineer/home')
@login_required('engineer')
def engineer_home():
    df = load_dataset()
    stats = {
        'total_samples': len(df),
        'features': len(df.columns) - 1,
        'classes': df['classification'].nunique(),
        'class_distribution': df['classification'].value_counts().to_dict()
    }
    return render_template('engineer_home.html', stats=stats, user_name=session.get('user_name'))

@app.route('/engineer/eda')
@login_required('engineer')
def engineer_eda():
    df = load_dataset()
    
    plots = {}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df['classification'].value_counts().plot(kind='bar', ax=ax, color=['#E91E63', '#4CAF50', '#FF9800'])
    ax.set_title('Classification Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classification')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#FFF0F5')
    buf.seek(0)
    plots['class_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdPu', ax=ax, fmt='.2f', 
                annot_kws={'size': 8}, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#FFF0F5')
    buf.seek(0)
    plots['correlation'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:8]):
        df[col].hist(ax=axes[i], bins=30, color='#E91E63', alpha=0.7, edgecolor='black')
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#FFF0F5')
    buf.seek(0)
    plots['distributions'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:8]):
        df.boxplot(column=col, by='classification', ax=axes[i])
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
    plt.suptitle('Features by Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#FFF0F5')
    buf.seek(0)
    plots['boxplots'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    stats = df.describe().round(2).to_html(classes='table table-striped')
    
    return render_template('eda.html', plots=plots, stats=stats, user_name=session.get('user_name'))

@app.route('/engineer/classification')
@login_required('engineer')
def engineer_classification():
    classifier = get_classifier()
    available_classifiers = classifier.get_classifier_names()
    return render_template('classification.html', 
                         classifiers=available_classifiers,
                         user_name=session.get('user_name'))

@app.route('/api/train/<classifier_name>')
@login_required('engineer')
def train_classifier(classifier_name):
    try:
        classifier = get_classifier()
        results = classifier.train_and_evaluate(classifier_name)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_results/<classifier_name>')
@login_required('engineer')
def get_classifier_results(classifier_name):
    try:
        classifier = get_classifier()
        results = classifier.get_results(classifier_name)
        if results:
            return jsonify(results)
        else:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/engineer/performance')
@login_required('engineer')
def engineer_performance():
    classifier = get_classifier()
    all_results = classifier.get_all_results()
    classifier_names = classifier.get_classifier_names()
    return render_template('performance.html', 
                         results=all_results,
                         classifier_names=classifier_names,
                         user_name=session.get('user_name'))

@app.route('/api/comparison_chart')
@login_required('engineer')
def get_comparison_chart():
    try:
        classifier = get_classifier()
        chart = classifier.get_comparison_chart()
        return jsonify({'chart': chart})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/prediction')
@login_required('user')
def user_prediction():
    df = load_dataset()
    feature_cols = [col for col in df.columns if col != 'classification']
    classifier = get_classifier()
    available_classifiers = classifier.get_trained_classifiers()
    return render_template('prediction.html', 
                         features=feature_cols,
                         classifiers=available_classifiers,
                         user_name=session.get('user_name'))

@app.route('/api/predict/single', methods=['POST'])
@login_required('user')
def predict_single():
    try:
        data = request.json
        classifier_name = data.get('classifier')
        features = data.get('features')
        
        classifier = get_classifier()
        result = classifier.predict_single(classifier_name, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
@login_required('user')
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        classifier_name = request.form.get('classifier')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        df = pd.read_csv(file, sep=';')
        
        classifier = get_classifier()
        results = classifier.predict_batch(classifier_name, df)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_db()
    port = int(os.environ.get('FLASK_PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
