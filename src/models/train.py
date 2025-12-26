"""
train.py - ML Model Training for Property Price Prediction
Golden Mile Properties AI Assignment - Stage 1

Updated with new dataset and pipeline structure.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class PropertyModelTrainer:
    """
    Trainer for property price prediction using new dataset.
    Uses sklearn Pipeline with ColumnTransformer.
    """
    
    def __init__(self, data_path: str = "/project/workspace/GoldenMile/data/processed/BangaloreDataMod.csv"):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the new property data CSV
        """
        self.data_path = data_path
        self.models = {}
        self.results = []
        self.artifacts_dir = "/project/workspace/GoldenMile/src/models/artifacts"
        
        # Feature definitions
        self.categorical_features = ["location", "property_type"]
        self.numerical_features = ["total_sqft", "bath", "bhk", "amenities_score"]
        self.target_column = "price"
        
        # Create artifacts directory
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the dataset.
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = self.categorical_features + self.numerical_features + [self.target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Display basic statistics
        print("\nDataset statistics:")
        print(f"  Number of properties: {len(df)}")
        print(f"  Price range: ₹{df[self.target_column].min():,.0f} - ₹{df[self.target_column].max():,.0f}")
        print(f"  Unique locations: {df['location'].nunique()}")
        print(f"  Property types: {df['property_type'].unique()}")
        
        return df
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", "passthrough", self.numerical_features)
            ]
        )
        
        return preprocessor
    
    def define_models(self) -> Dict[str, Any]:
        """
        Define the models to train.
        """
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=250,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        }
        
        return models
    
    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """
        Train and evaluate all models.
        """
        print("\nTraining and evaluating models...")
        
        preprocessor = self.create_preprocessor()
        models = self.define_models()
        
        results = []
        trained_pipelines = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ]
            )
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results.append({
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })
            
            # Store pipeline
            trained_pipelines[name] = pipeline
            
            print(f"    R²: {r2:.4f}, RMSE: {rmse:,.0f}, MAE: {mae:,.0f}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("R2", ascending=False)
        
        self.models = trained_pipelines
        self.results = results_df
        
        return results_df
    
    def select_best_model(self) -> Pipeline:
        """
        Select the best performing model.
        """
        best_model_name = self.results.iloc[0]["Model"]
        best_pipeline = self.models[best_model_name]
        
        print(f"\nBest model selected: {best_model_name}")
        print(f"  R² Score: {self.results.iloc[0]['R2']:.4f}")
        print(f"  RMSE: {self.results.iloc[0]['RMSE']:,.0f}")
        
        return best_pipeline
    
    def save_artifacts(self, best_pipeline: Pipeline):
        """
        Save all training artifacts.
        """
        print("\nSaving artifacts...")
        
        # Get best model name
        best_model_name = self.results.iloc[0]["Model"]
        
        # Save the complete pipeline (preprocessor + model)
        pipeline_path = os.path.join(self.artifacts_dir, "price_predictor_pipeline.pkl")
        joblib.dump(best_pipeline, pipeline_path)
        print(f"  Pipeline saved: {pipeline_path}")
        
        # Save results
        results_path = os.path.join(self.artifacts_dir, "model_evaluation.csv")
        self.results.to_csv(results_path, index=False)
        print(f"  Results saved: {results_path}")
        
        # Save feature information
        feature_info = {
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "target_column": self.target_column,
            "best_model": best_model_name,
            "all_models": list(self.models.keys())
        }
        
        features_path = os.path.join(self.artifacts_dir, "feature_info.json")
        with open(features_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"  Feature info saved: {features_path}")
        
        # Save metadata
        metadata = {
            "training_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "dataset": os.path.basename(self.data_path),
            "best_model_score": float(self.results.iloc[0]['R2']),
            "total_samples": len(self.results)
        }
        
        metadata_path = os.path.join(self.artifacts_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved: {metadata_path}")
        
        # Save sample predictions for documentation
        self.save_sample_predictions(best_pipeline)
    
    def save_sample_predictions(self, pipeline: Pipeline):
        """
        Save sample predictions for testing and documentation.
        """
        # Load data for samples
        df = pd.read_csv(self.data_path)
        X = df[self.categorical_features + self.numerical_features]
        y = df[self.target_column]
        
        # Get a few samples
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        samples = []
        
        for idx in sample_indices:
            sample_data = X.iloc[idx].to_dict()
            actual_price = y.iloc[idx]
            predicted_price = pipeline.predict(pd.DataFrame([sample_data]))[0]
            
            samples.append({
                "input": sample_data,
                "actual_price": float(actual_price),
                "predicted_price": float(predicted_price),
                "error_pct": float(((predicted_price - actual_price) / actual_price) * 100)
            })
        
        samples_path = os.path.join(self.artifacts_dir, "sample_predictions.json")
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"  Sample predictions saved: {samples_path}")
    
    def run_training(self):
        """
        Execute the complete training pipeline.
        """
        print("=" * 60)
        print("PROPERTY PRICE MODEL TRAINING")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Prepare features and target
            X = df[self.categorical_features + self.numerical_features]
            y = df[self.target_column]
            
            # Step 3: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"\nData split: {X_train.shape[0]} train, {X_test.shape[0]} test")
            
            # Step 4: Train and evaluate models
            results_df = self.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # Step 5: Display results
            print("\n" + "=" * 60)
            print("MODEL EVALUATION RESULTS")
            print("=" * 60)
            print(results_df.to_string(index=False))
            
            # Step 6: Select best model
            best_pipeline = self.select_best_model()
            
            # Step 7: Save artifacts
            self.save_artifacts(best_pipeline)
            
            # Step 8: Retrain on full dataset for production
            print("\nRetraining best model on full dataset...")
            X_full = df[self.categorical_features + self.numerical_features]
            y_full = df[self.target_column]
            
            # Get the best model type and create new pipeline
            best_model_name = self.results.iloc[0]["Model"]
            best_model = self.define_models()[best_model_name]
            
            final_pipeline = Pipeline([
                ("preprocessor", self.create_preprocessor()),
                ("model", best_model)
            ])
            
            final_pipeline.fit(X_full, y_full)
            
            # Save final pipeline
            final_path = os.path.join(self.artifacts_dir, "final_price_predictor.pkl")
            joblib.dump(final_pipeline, final_path)
            print(f"Final pipeline saved: {final_path}")
            
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            
            return {
                "best_pipeline": final_pipeline,
                "results": results_df,
                "best_model_name": best_model_name
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    Main function to run training.
    """
    try:
        trainer = PropertyModelTrainer()
        results = trainer.run_training()
        
        # Print final summary
        print("\nTRAINING SUMMARY")
        print("-" * 40)
        print(f"Best Model: {results['best_model_name']}")
        print(f"R² Score: {results['results'].iloc[0]['R2']:.4f}")
        print(f"RMSE: {results['results'].iloc[0]['RMSE']:,.0f}")
        print(f"\nArtifacts saved in: {trainer.artifacts_dir}/")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())