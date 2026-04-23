import os
import numpy as np
import pandas as pd
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, RocCurveDisplay)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class DiabetesClassifiers:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.df = pd.read_csv(dataset_path, sep=';')
        self.feature_cols = [col for col in self.df.columns if col != 'classification']
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.X = self.df[self.feature_cols].values
        self.y_original = self.df['classification'].values
        self.y_encoded = self.label_encoder.fit_transform(self.y_original)
        self.class_names = self.label_encoder.classes_.tolist()
        self.n_classes = len(self.class_names)
        
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        self.y_train_original = self.label_encoder.inverse_transform(self.y_train)
        self.y_test_original = self.label_encoder.inverse_transform(self.y_test)
        
        self.results = {}
        
        self.classifier_names = {
            'PAC': 'Passive Aggressive Classifier (PAC)',
            'DTC': 'Decision Tree Classifier (DTC)',
            'KNN': 'K-Nearest Neighbors (KNN)',
            'DNN': 'Deep Neural Network (DNN)',
            'CNN': 'Convolutional Neural Network (CNN)',
            'RNN': 'Recurrent Neural Network (RNN)',
            'GRIM': 'Hybrid Digital Twin Feature Extraction based GRIM'
        }
    
    def get_classifier_names(self):
        return self.classifier_names
    
    def get_trained_classifiers(self):
        trained = []
        for key in self.classifier_names.keys():
            model_path = os.path.join(self.models_dir, f'{key}_model.joblib')
            keras_path = os.path.join(self.models_dir, f'{key}_model.keras')
            if os.path.exists(model_path) or os.path.exists(keras_path):
                trained.append({'key': key, 'name': self.classifier_names[key]})
        return trained
    
    def _save_model(self, name, model):
        if isinstance(model, keras.Model):
            model.save(os.path.join(self.models_dir, f'{name}_model.keras'))
        else:
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_model.joblib'))
    
    def _load_model(self, name):
        keras_path = os.path.join(self.models_dir, f'{name}_model.keras')
        joblib_path = os.path.join(self.models_dir, f'{name}_model.joblib')
        
        if os.path.exists(keras_path):
            return keras.models.load_model(keras_path)
        elif os.path.exists(joblib_path):
            return joblib.load(joblib_path)
        return None
    
    def _model_exists(self, name):
        keras_path = os.path.join(self.models_dir, f'{name}_model.keras')
        joblib_path = os.path.join(self.models_dir, f'{name}_model.joblib')
        return os.path.exists(keras_path) or os.path.exists(joblib_path)
    
    def _create_pac(self):
        return PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3)
    
    def _create_dtc(self):
        return DecisionTreeClassifier(random_state=42, max_depth=2, min_samples_split=5)
    
    def _create_knn(self):
        return KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
    
    def _create_dnn(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(len(self.feature_cols),)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_cnn(self):
        model = Sequential([
            Conv1D(64, 2, activation='relu', input_shape=(len(self.feature_cols), 1), padding='same'),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(32, 2, activation='relu', padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_rnn(self):
        model = Sequential([
            SimpleRNN(64, activation='relu', input_shape=(len(self.feature_cols), 1), return_sequences=True),
            SimpleRNN(32, activation='relu'),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_grim(self):
        from grim_digital_twin import (
            create_grim_digital_twin_classifier,
            grim_digital_twin_train
        )

        model = create_grim_digital_twin_classifier()

        if model is None:
            print("No saved GRIM with Digital Twin model found. Training now...")

            grim_digital_twin_train(
                self.X_train, self.y_train,
                self.X_test, self.y_test
            )

            model = create_grim_digital_twin_classifier()

            if model is None:
                raise ValueError("Training completed but model still not found/saved.")

        return model

    
    def _generate_confusion_matrix_plot(self, y_true, y_pred, classifier_name):
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {self.classifier_names[classifier_name]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='#FFF0F5')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_base64
    
    def _generate_roc_curve_plot(self, y_true, y_proba, classifier_name):
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#E91E63', '#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
        
        for i, class_name in enumerate(self.class_names):
            if y_proba.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                color = colors[i % len(colors)]
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {self.classifier_names[classifier_name]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#FFF0F5')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_base64
    
    def train_and_evaluate(self, classifier_name):
        if self._model_exists(classifier_name):
            model = self._load_model(classifier_name)
            print(f"Loaded existing model for {classifier_name}")
        else:
            if classifier_name == 'PAC':
                model = self._create_pac()
                model.fit(self.X_train, self.y_train)
            elif classifier_name == 'DTC':
                model = self._create_dtc()
                model.fit(self.X_train, self.y_train)
            elif classifier_name == 'KNN':
                model = self._create_knn()
                model.fit(self.X_train, self.y_train)
            elif classifier_name == 'DNN':
                model = self._create_dnn()
                y_train_cat = to_categorical(self.y_train, self.n_classes)
                model.fit(self.X_train, y_train_cat, epochs=5, batch_size=32, verbose=2, validation_split=0.1)
            elif classifier_name == 'CNN':
                model = self._create_cnn()
                X_train_cnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                y_train_cat = to_categorical(self.y_train, self.n_classes)
                model.fit(X_train_cnn, y_train_cat, epochs=5, batch_size=32, verbose=2, validation_split=0.1)
            elif classifier_name == 'RNN':
                model = self._create_rnn()
                X_train_rnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                y_train_cat = to_categorical(self.y_train, self.n_classes)
                model.fit(X_train_rnn, y_train_cat, epochs=5, batch_size=32, verbose=2, validation_split=0.1)
            elif classifier_name == 'GRIM':
                model = self._create_grim()
                model.fit(self.X_train, self.y_train)
            else:
                raise ValueError(f"Unknown classifier: {classifier_name}")
            
            self._save_model(classifier_name, model)
            print(f"Trained and saved model for {classifier_name}")
        
        if classifier_name == 'DNN':
            y_proba = model.predict(self.X_test, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)

        elif classifier_name == 'GRIM':
            y_pred = model.predict(self.X_test)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
            else:
                y_proba = None
                
        elif classifier_name == 'CNN':
            X_test_cnn = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            y_proba = model.predict(X_test_cnn, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
        elif classifier_name == 'RNN':
            X_test_rnn = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            y_proba = model.predict(X_test_rnn, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = model.predict(self.X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test)
            else:
                y_proba = np.zeros((len(y_pred), self.n_classes))
                for i, pred in enumerate(y_pred):
                    y_proba[i, pred] = 1.0
        
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.y_test_original
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        cm_plot = self._generate_confusion_matrix_plot(y_test_labels, y_pred_labels, classifier_name)
        roc_plot = self._generate_roc_curve_plot(self.y_test, y_proba, classifier_name)
        
        cr = classification_report(y_test_labels, y_pred_labels, target_names=self.class_names, output_dict=True)
        cr_html = self._format_classification_report(cr)
        
        sample_predictions = []
        for i in range(min(10, len(y_pred_labels))):
            sample_predictions.append({
                'actual': str(y_test_labels[i]),
                'predicted': str(y_pred_labels[i])
            })
        
        results = {
            'classifier_name': self.classifier_names[classifier_name],
            'classifier_key': classifier_name,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm_plot,
            'roc_curve': roc_plot,
            'classification_report': cr_html,
            'sample_predictions': sample_predictions
        }
        
        self.results[classifier_name] = results
        return results
    
    def _format_classification_report(self, cr):
        html = '<table class="table table-striped classification-report">'
        html += '<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>'
        html += '<tbody>'
        
        for key, values in cr.items():
            if key in self.class_names:
                html += f'<tr><td><strong>{key}</strong></td>'
                html += f'<td>{values["precision"]:.3f}</td>'
                html += f'<td>{values["recall"]:.3f}</td>'
                html += f'<td>{values["f1-score"]:.3f}</td>'
                html += f'<td>{int(values["support"])}</td></tr>'
        
        if 'accuracy' in cr:
            html += f'<tr class="table-info"><td><strong>Accuracy</strong></td>'
            html += f'<td colspan="2"></td>'
            html += f'<td>{cr["accuracy"]:.3f}</td>'
            html += f'<td>{int(cr["weighted avg"]["support"])}</td></tr>'
        
        if 'weighted avg' in cr:
            html += f'<tr class="table-primary"><td><strong>Weighted Avg</strong></td>'
            html += f'<td>{cr["weighted avg"]["precision"]:.3f}</td>'
            html += f'<td>{cr["weighted avg"]["recall"]:.3f}</td>'
            html += f'<td>{cr["weighted avg"]["f1-score"]:.3f}</td>'
            html += f'<td>{int(cr["weighted avg"]["support"])}</td></tr>'
        
        html += '</tbody></table>'
        return html
    
    def get_results(self, classifier_name):
        if classifier_name in self.results:
            return self.results[classifier_name]
        
        if self._model_exists(classifier_name):
            return self.train_and_evaluate(classifier_name)
        
        return None
    
    def get_all_results(self):
        all_results = {}
        for key in self.classifier_names.keys():
            if self._model_exists(key):
                if key not in self.results:
                    self.train_and_evaluate(key)
                if key in self.results:
                    all_results[key] = self.results[key]
        return all_results
    
    def get_comparison_chart(self):
        all_results = self.get_all_results()
        
        if not all_results:
            return None
        
        classifiers = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        
        for key, result in all_results.items():
            classifiers.append(key)
            accuracy.append(result['accuracy'])
            precision.append(result['precision'])
            recall.append(result['recall'])
            f1.append(result['f1_score'])
        
        x = np.arange(len(classifiers))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#E91E63')
        bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#4CAF50')
        bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#FF9800')
        bars4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#2196F3')
        
        ax.set_xlabel('Classifier', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Performance Comparison of All Classifiers', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#FFF0F5')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_base64
    
    def predict_single(self, classifier_name, features):
        model = self._load_model(classifier_name)
        if model is None:
            raise ValueError(f"Model {classifier_name} not found. Please train it first.")
        
        feature_array = np.array([features])
        feature_scaled = self.scaler.transform(feature_array)
        
        if classifier_name == 'DNN':
            proba = model.predict(feature_scaled, verbose=0)
            pred = np.argmax(proba, axis=1)
            confidence = float(np.max(proba))

        elif classifier_name == 'GRIM':
            pred = model.predict(feature_scaled)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_scaled)
                confidence = float(np.max(proba))
            else:
                proba = None
                confidence = 1.0  # fallback confidence

        elif classifier_name == 'CNN':
            feature_cnn = feature_scaled.reshape(1, len(features), 1)
            proba = model.predict(feature_cnn, verbose=0)
            pred = np.argmax(proba, axis=1)
            confidence = float(np.max(proba))
        elif classifier_name == 'RNN':
            feature_rnn = feature_scaled.reshape(1, len(features), 1)
            proba = model.predict(feature_rnn, verbose=0)
            pred = np.argmax(proba, axis=1)
            confidence = float(np.max(proba))
        else:
            pred = model.predict(feature_scaled)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_scaled)
                confidence = float(np.max(proba))
            else:
                confidence = 1.0
        
        predicted_class = self.label_encoder.inverse_transform(pred)[0]
        
        return {
            'prediction': str(predicted_class),
            'confidence': round(confidence * 100, 2),
            'classifier': self.classifier_names[classifier_name]
        }
    
    def predict_batch(self, classifier_name, df):
        model = self._load_model(classifier_name)
        if model is None:
            raise ValueError(f"Model {classifier_name} not found. Please train it first.")
        
        feature_cols = [col for col in df.columns if col != 'classification']
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        if classifier_name == 'DNN':
            proba = model.predict(X_scaled, verbose=0)
            predictions = np.argmax(proba, axis=1)

        elif classifier_name == 'GRIM':
            predictions = model.predict(X_scaled)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
            else:
                proba = None


        elif classifier_name == 'CNN':
            X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            proba = model.predict(X_cnn, verbose=0)
            predictions = np.argmax(proba, axis=1)
        elif classifier_name == 'RNN':
            X_rnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            proba = model.predict(X_rnn, verbose=0)
            predictions = np.argmax(proba, axis=1)
        else:
            predictions = model.predict(X_scaled)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = []
        for i in range(len(predicted_labels)):
            row_data = {col: str(df.iloc[i][col]) for col in feature_cols[:3]}
            row_data['prediction'] = str(predicted_labels[i])
            results.append(row_data)
        
        return {
            'predictions': results,
            'total': len(results),
            'classifier': self.classifier_names[classifier_name]
        }
