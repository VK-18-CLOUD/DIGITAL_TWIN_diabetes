import os
import joblib
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
#   GRIM CLASSIFIER (Rule-based Interpretable Model)
# ============================================================
class GRIM(BaseEstimator, ClassifierMixin):
    """
    GRIM: Guided Rule-based Interpretable Model
    """

    def __init__(self, max_depth=None, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

        self._dt_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.rule_list_ = None

    def fit(self, X, y):
        self._dt_model.fit(X, y)
        self._generate_rule_list()
        return self

    def predict(self, X):
        return self._dt_model.predict(X)

    def predict_proba(self, X):
        return self._dt_model.predict_proba(X)

    def score(self, X, y):
        return self._dt_model.score(X, y)

    def _generate_rule_list(self):
        tree = self._dt_model.tree_
        feature = tree.feature
        used_features = feature[feature >= 0]
        unique_features = np.unique(used_features)

        rules = []
        rules.append(f"[GRIM] Tree depth={tree.max_depth}, Nodes={tree.node_count}")
        rules.append(f"[GRIM] Features used={len(unique_features)}")
        self.rule_list_ = rules

    def get_rule_list(self):
        if self.rule_list_ is None:
            raise ValueError("❌ GRIM must be fitted before extracting rules.")
        return self.rule_list_


# ============================================================
#   ATTENTION LAYER (Used in Digital Twin)
# ============================================================
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        score = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        attention = keras.backend.softmax(score, axis=1)
        return keras.backend.sum(x * attention, axis=1)


# ============================================================
#   WRAPPER FOR DIGITAL TWIN MODEL (SKLEARN STYLE)
# ============================================================
class WrappedDigitalTwinClassifier:
    """
    Wrapper for Digital Twin (BiLSTM+Attention) classifier
    to behave like sklearn model: predict(), predict_proba()
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        X_seq = np.expand_dims(X, 1)
        self.model.fit(X_seq, y, epochs=1, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        proba = self.model.predict(np.expand_dims(X, 1), verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        return self.model.predict(np.expand_dims(X, 1), verbose=0)


# ============================================================
#   BUILD DIGITAL TWIN MODEL (BiLSTM + Attention)
# ============================================================
def build_digital_twin_classifier(input_dim, num_classes):
    """
    Digital Twin = BiLSTM + Attention based classifier
    """

    inputs = Input(shape=(1, input_dim))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = AttentionLayer()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
#   CREATE / LOAD BEST MODEL (AUTO LOAD)
# ============================================================
def create_grim_digital_twin_classifier():
    """
    Loads the best saved classifier:
      - Digital Twin (BiLSTM+Attention) -> .h5
      - GRIM -> .pkl
    """

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    digital_twin_h5 = os.path.join(model_dir, "best_digital_twin_classifier.h5")
    grim_pkl = os.path.join(model_dir, "best_grim_classifier.pkl")

    # Load Digital Twin model if exists
    if os.path.exists(digital_twin_h5):
        print("✅ Loaded best Digital Twin Classifier (BiLSTM + Attention)")
        model = keras.models.load_model(
            digital_twin_h5,
            compile=False,
            custom_objects={"AttentionLayer": AttentionLayer}
        )
        return WrappedDigitalTwinClassifier(model)

    # Load GRIM model if exists
    if os.path.exists(grim_pkl):
        print("✅ Loaded best GRIM Classifier")
        return joblib.load(grim_pkl)

    print("❗ No saved model found.")
    print("❗ Train first using grim_digital_twin_train(X_train, y_train, X_test, y_test)")
    return None


# ============================================================
#   TRAIN GRIM + DIGITAL TWIN AND SAVE BEST
# ============================================================
def grim_digital_twin_train(X_train, y_train, X_test, y_test):
    """
    Train both models:
      1) Digital Twin (BiLSTM + Attention)
      2) GRIM (Decision Rule List Classifier)
    Select best accuracy and save.
    """

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    digital_twin_h5 = os.path.join(model_dir, "best_digital_twin_classifier.h5")
    grim_pkl = os.path.join(model_dir, "best_grim_classifier.pkl")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    # ============================
    # 1) Train Digital Twin
    # ============================
    print("\n🚀 Training Digital Twin Classifier (BiLSTM + Attention)...")

    X_train_seq = np.expand_dims(X_train, 1)
    X_test_seq = np.expand_dims(X_test, 1)

    digital_twin_model = build_digital_twin_classifier(input_dim, num_classes)

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    digital_twin_model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=20,
        batch_size=32,
        verbose=2,
        callbacks=[es]
    )

    _, digital_twin_acc = digital_twin_model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"✅ Digital Twin Accuracy: {digital_twin_acc:.4f}")

    # ============================
    # 2) Train GRIM
    # ============================
    print("\n🚀 Training GRIM Classifier...")

    grim_model = GRIM(max_depth=None)
    grim_model.fit(X_train, y_train)

    y_pred_grim = grim_model.predict(X_test)
    grim_acc = accuracy_score(y_test, y_pred_grim)
    print(f"✅ GRIM Accuracy: {grim_acc:.4f}")

    # ============================
    # Save Best Model
    # ============================
    if grim_acc > digital_twin_acc:
        print("\n🏆 Best Model = GRIM")
        joblib.dump(grim_model, grim_pkl)

        # remove Digital Twin model if exists
        if os.path.exists(digital_twin_h5):
            os.remove(digital_twin_h5)

        print("\n📌 GRIM Rule List:")
        for rule in grim_model.get_rule_list():
            print("   ", rule)

    else:
        print("\n🏆 Best Model = Digital Twin (BiLSTM + Attention)")
        digital_twin_model.save(digital_twin_h5)

        # remove GRIM model if exists
        if os.path.exists(grim_pkl):
            os.remove(grim_pkl)

    print("\n✅ Best model saved successfully!")
