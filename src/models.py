from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

from sklearn.calibration import CalibratedClassifierCV
import numpy as np


class MentalHealthClassifier:
    """
    Optimized classifier for mental health text classification
    """
    
    def __init__(self, model_type='logistic', class_weights='balanced', random_state=42):
        """
        Initialize classifier
        
        Args:
            model_type: Type of model ('logistic', 'svm', 'lgbm', 'ensemble')
            class_weights: 'balanced' or None
            random_state: Random seed
        """
        self.model_type = model_type
        self.class_weights = class_weights
        self.random_state = random_state
        self.model = self._create_model()
    
    def _create_model(self):
        """Create the model based on type"""
        if self.model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                solver='saga',
                max_iter=1000,
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=-1,
                penalty='l2'
            )
        
        elif self.model_type == 'svm':
            base_svm = LinearSVC(
                C=0.5,
                max_iter=2000,
                class_weight=self.class_weights,
                random_state=self.random_state,
                dual=False
            )
            return CalibratedClassifierCV(base_svm, cv=3)
        
        elif self.model_type == 'lgbm':
            if LIGHTGBM_AVAILABLE:
                return LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=50,
                    min_child_samples=20,
                    class_weight=self.class_weights,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1
                )
            else:
                raise ImportError("LightGBM not available")
        
        elif self.model_type == 'xgboost':
            if XGBOOST_AVAILABLE:
                return XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=6,
                    min_child_weight=3,
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )
            else:
                raise ImportError("XGBoost not available")
        
        elif self.model_type == 'ensemble':
            lr = LogisticRegression(
                C=1.0,
                solver='saga',
                max_iter=1000,
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            if LIGHTGBM_AVAILABLE:
                lgbm = LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    class_weight=self.class_weights,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1
                )
                
                return VotingClassifier(
                    estimators=[
                        ('lr', lr),
                        ('lgbm', lgbm)
                    ],
                    voting='soft',
                    n_jobs=-1
                )
            else:
                raise ImportError("LightGBM not available for ensemble")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_model(self):
        """Get the underlying model"""
        return self.model


def get_best_models_for_comparison():
    """
    Returns 10+ models optimized for mental health classification
    Focuses on models that handle class imbalance well
    Models are optimized for speed while maintaining accuracy
    """
    models = {}
    
    # MODEL 1: Logistic Regression - Fast and effective baseline
    models["1. Logistic Regression"] = LogisticRegression(
        C=1.0,
        solver='saga',
        max_iter=500,  # Reduced for speed
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        penalty='l2'
    )
    
    # MODEL 2: Ridge Classifier - Fast linear model
    models["2. Ridge Classifier"] = RidgeClassifier(
        alpha=1.0,
        class_weight='balanced',
        random_state=42
    )
    
    # MODEL 3: SGD Classifier - Very fast for large datasets
    models["3. SGD Classifier"] = SGDClassifier(
        loss='log_loss',  # Logistic regression with SGD
        penalty='l2',
        alpha=0.0001,
        max_iter=500,  # Fast convergence
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # MODEL 4: Linear SVM (Calibrated) - Good for high-dimensional text
    models["4. Calibrated Linear SVM"] = CalibratedClassifierCV(
        LinearSVC(
            C=0.5,
            max_iter=1000,  # Reduced for speed
            class_weight='balanced',
            random_state=42,
            dual=False
        ),
        cv=3
    )
    
    # MODEL 5: Bernoulli Naive Bayes - Works with dense/negative features
    models["5. Bernoulli Naive Bayes"] = BernoulliNB(alpha=0.5)
    
    # MODEL 6: K-Nearest Neighbors (5 neighbors) - Instance-based
    models["6. K-Nearest Neighbors"] = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    
    # MODEL 7: Decision Tree - Fast and interpretable
    models["7. Decision Tree"] = DecisionTreeClassifier(
        max_depth=15,  # Limited depth for speed
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    # MODEL 8: Random Forest - Reduced trees for speed
    models["8. Random Forest"] = RandomForestClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=15,  # Limited depth
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_features='sqrt'  # Faster than 'auto'
    )
    
    # MODEL 9: Extra Trees - Faster than Random Forest
    models["9. Extra Trees"] = ExtraTreesClassifier(
        n_estimators=100,  # Fast ensemble
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_features='sqrt'
    )
    
    # MODEL 10: AdaBoost - Reduced iterations significantly
    models["10. AdaBoost"] = AdaBoostClassifier(
        n_estimators=30,  # Reduced from 50
        learning_rate=1.5,  # Increased to compensate
        random_state=42,
        algorithm='SAMME'
    )
    
    # MODEL 11: Gradient Boosting - Heavily optimized for speed
    models["11. Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=30,  # Reduced from 50 for much faster training
        learning_rate=0.2,  # Increased to compensate for fewer trees
        max_depth=3,  # Very shallow trees for speed
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.7,  # Train on 70% of data
        max_features='sqrt',  # Faster feature selection
        random_state=42
    )
    
    # MODEL 12: LightGBM (if available) - Fast gradient boosting
    if LIGHTGBM_AVAILABLE:
        models["12. LightGBM"] = LGBMClassifier(
            n_estimators=100,  # Reduced from 500
            learning_rate=0.1,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    
    # MODEL 13: XGBoost (if available) - Optimized for speed
    if XGBOOST_AVAILABLE:
        models["13. XGBoost"] = XGBClassifier(
            n_estimators=50,  # Reduced from 100
            learning_rate=0.2,  # Increased for fewer trees
            max_depth=4,  # Reduced depth
            min_child_weight=10,  # Increased for speed
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            tree_method='hist',  # Fastest training method
            max_bin=128  # Reduced bins for speed
        )
    
    # MODEL 14: Voting Ensemble - Combine top 3 fast models
    models["14. Voting Ensemble (Top 3)"] = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(
                C=1.0, solver='saga', max_iter=500,
                class_weight='balanced', random_state=42, n_jobs=-1
            )),
            ('ridge', RidgeClassifier(
                alpha=1.0, class_weight='balanced', random_state=42
            )),
            ('svm', CalibratedClassifierCV(
                LinearSVC(
                    C=0.5, max_iter=1000, class_weight='balanced',
                    random_state=42, dual=False
                ),
                cv=3
            ))
        ],
        voting='soft',
        n_jobs=-1
    )
    
    return models


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, 
                             label_encoder, model_name="Model"):
    """
    Train and evaluate a single model
    
    Args:
        model: Sklearn-compatible model
        X_train, y_train: Training data
        X_test, y_test: Test data
        label_encoder: Fitted label encoder
        model_name: Name for logging
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, 
        f1_score, 
        classification_report, 
        confusion_matrix,
        balanced_accuracy_score
    )
    import time
    

    print(f"Training {model_name}...")

    
    # Train with timing
    start_time = time.time()
    try:
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"\nResults for {model_name}:")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Macro F1:          {macro_f1:.4f}")
        print(f"  Weighted F1:       {weighted_f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'training_time': training_time,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'success': True
        }
        
    except Exception as e:
        print(f"\nError training {model_name}: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }