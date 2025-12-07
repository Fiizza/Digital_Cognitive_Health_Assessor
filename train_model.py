import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import from our modules
from preprocessing import load_and_preprocess_data, augment_minority_classes
from features import MentalHealthFeatureExtractor
from models import get_best_models_for_comparison, train_and_evaluate_model


def main():
    print("TRAINING PIPELINE")
    
    print("\n[STEP 1] Loading and preprocessing data...")
    
    data_path = "../Data/data.csv"
    texts, labels, label_encoder, df = load_and_preprocess_data(
        data_path, 
        preprocess_text=True
    )
    
    print(f"\nTotal samples: {len(texts)}")
    print(f"Classes: {label_encoder.classes_}")
    
    
    print("\n[STEP 2] Splitting data into train/test sets...")
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )
    
    print(f"Train samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}")
    

    print("\n[STEP 3] Augmenting minority classes...")
    
  
    USE_AUGMENTATION = True
    
    if USE_AUGMENTATION:
        try:
            X_train_text, y_train = augment_minority_classes(
                X_train_text, 
                y_train, 
                label_encoder,
                augment_to='mean', 
                max_augment_ratio=1  
            )
            print(f"After augmentation - Train samples: {len(X_train_text)}")
        except MemoryError as e:
            print(f"Memory error during augmentation: {e}")
            print("Skipping augmentation and proceeding with original data...")
            USE_AUGMENTATION = False
    else:
        print("Augmentation skipped - using original training data")
    
    print("\n[STEP 4] Extracting features...")
   
    feature_extractor = MentalHealthFeatureExtractor(
        use_tfidf=True,
        ngram_range=(1, 3), 
        max_features=50000,
        min_df=2, 
        max_df=0.8,
        use_svd=True,  
        svd_components=400
    )
    
    X_train = feature_extractor.fit_transform(X_train_text)
    
    X_test = feature_extractor.transform(X_test_text)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    

    print("\n[STEP 5] Training and comparing models...")

    
    models = get_best_models_for_comparison()
    results = {}
    best_model = None
    best_metric_value = 0
    best_model_name = ""
    
   
    optimize_metric = 'balanced_accuracy'
    
    for model_name, model in models.items():
        try:
           
            metrics = train_and_evaluate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                model_name=model_name
            )
            
            results[model_name] = metrics
            
       
            if metrics[optimize_metric] > best_metric_value:
                best_metric_value = metrics[optimize_metric]
                best_model = model
                best_model_name = model_name
      
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = None
    
   
    print("\n[STEP 6] Training complete!")
   
    print("\nMODEL COMPARISON SUMMARY:")
    
    print(f"{'Model':<35} {'Accuracy':<12} {'Bal. Acc':<12} {'Macro F1':<12} {'Weight F1':<12} {'Time (s)':<10}")

    
    successful_models = []
    for model_name, metrics in results.items():
        if metrics and metrics.get('success', False):
            print(f"{model_name:<35} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['balanced_accuracy']:<12.4f} "
                  f"{metrics['macro_f1']:<12.4f} "
                  f"{metrics['weighted_f1']:<12.4f} "
                  f"{metrics['training_time']:<10.2f}")
            successful_models.append((model_name, metrics))
        elif metrics and not metrics.get('success', False):
            print(f"{model_name:<35} ERROR: {metrics.get('error', 'Unknown error')}")
 
    print(f"BEST MODEL: {best_model_name}")
    print(f"Best {optimize_metric}: {best_metric_value:.4f}")

    
    # Show top 5 models
    print("\n TOP 5 MODELS BY BALANCED ACCURACY:")
    sorted_models = sorted(successful_models, key=lambda x: x[1]['balanced_accuracy'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_models[:5], 1):
        print(f"  {i}. {name}: {metrics['balanced_accuracy']:.4f}")
    
    print("\nTOP 5 MODELS BY MACRO F1:")
    sorted_models_f1 = sorted(successful_models, key=lambda x: x[1]['macro_f1'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_models_f1[:5], 1):
        print(f"  {i}. {name}: {metrics['macro_f1']:.4f}")
    
    print("\nFASTEST 5 MODELS:")
    sorted_models_time = sorted(successful_models, key=lambda x: x[1]['training_time'])
    for i, (name, metrics) in enumerate(sorted_models_time[:5], 1):
        print(f"  {i}. {name}: {metrics['training_time']:.2f}s")
   
    print("\n[STEP 7] Saving best model and components...")
    
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save best model
    with open(os.path.join(save_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved best model: {best_model_name}")
    
    # Save feature extractor components
    with open(os.path.join(save_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(feature_extractor.vectorizer, f)
    print("Saved TF-IDF vectorizer")
    
    if feature_extractor.svd is not None:
        with open(os.path.join(save_dir, "svd.pkl"), "wb") as f:
            pickle.dump(feature_extractor.svd, f)
        print("Saved SVD transformer")
    
    # Save label encoder
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print(" Saved label encoder")
    
    # Save text preprocessor flag
    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        config = {
            'use_preprocessing': True,
            'best_model_name': best_model_name,
            'classes': label_encoder.classes_.tolist(),
            'optimize_metric': optimize_metric,
            'best_metric_value': best_metric_value
        }
        pickle.dump(config, f)
    print(" Saved configuration")
    
    
    print("\n[STEP 8] Testing predictions on sample texts...")

    
    # Import preprocessing for test
    from preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    
    test_samples = [
        "I feel so anxious all the time, my heart races and I can't calm down",
        "I'm feeling great today, everything is going well",
        "I can't get out of bed, nothing brings me joy anymore",
        "My mood swings are extreme, one moment I'm on top of the world",
        "I don't want to live anymore, everything is hopeless",
        "I'm so stressed with work deadlines and family issues"
    ]
    
    for sample_text in test_samples:
        # Preprocess
        processed_text = preprocessor.preprocess(sample_text)
        
        # Extract features
        features = feature_extractor.transform([processed_text])
        
        # Predict
        prediction = best_model.predict(features)[0]
        probabilities = best_model.predict_proba(features)[0]
        predicted_class = label_encoder.classes_[prediction]
        
        print(f"\nText: \"{sample_text[:60]}...\"")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {probabilities[prediction]:.3f}")
        print(f"All probabilities:")
        for idx, (class_name, prob) in enumerate(zip(label_encoder.classes_, probabilities)):
            print(f"  {class_name}: {prob:.3f}")
    
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
 
    
    return results, best_model, feature_extractor, label_encoder


if __name__ == "__main__":
    results, best_model, feature_extractor, label_encoder = main()