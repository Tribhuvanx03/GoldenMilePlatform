"""
inference.py - Property Price Inference Engine
Golden Mile Properties AI Assignment

Simple inference using the trained pipeline.
"""

import joblib
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List


class PropertyPricePredictor:
    """
    Simple property price predictor using the trained pipeline.
    """
    
    def __init__(self, artifacts_dir: str = "/project/workspace/GoldenMile/src/models/artifacts"):
        """
        Initialize predictor with trained artifacts.
        
        Args:
            artifacts_dir: Directory containing model artifacts
        """
        self.artifacts_dir = artifacts_dir
        
        # Load the trained pipeline
        pipeline_path = os.path.join(artifacts_dir, "final_price_predictor.pkl")
        if os.path.exists(pipeline_path):
            self.pipeline = joblib.load(pipeline_path)
        else:
            # Fallback to regular pipeline
            pipeline_path = os.path.join(artifacts_dir, "price_predictor_pipeline.pkl")
            self.pipeline = joblib.load(pipeline_path)
        
        # Load feature information
        features_path = os.path.join(artifacts_dir, "feature_info.json")
        with open(features_path, 'r') as f:
            feature_info = json.load(f)
        
        self.categorical_features = feature_info["categorical_features"]
        self.numerical_features = feature_info["numerical_features"]
        self.best_model = feature_info.get("best_model", "Unknown")
        
        # Yield lookup table (could be loaded from file)
        self.yield_lookup = {
            "Whitefield": 3.2,
            "Electronic City Phase II": 3.3,
            "Old Airport Road": 2.5,
            "Rajaji Nagar": 2.6,
            "Marathahalli": 3.4,
            "Hebbal": 3.1,
            "Indiranagar": 2.8,
            "Koramangala": 2.9,
            "HSR Layout": 3.2,
            "Bellandur": 3.3
        }
        
        print(f"Loaded {self.best_model} model for inference")
    
    def validate_input(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Validate input data.
        
        Returns:
            List of error messages, empty if valid
        """
        errors = []
        
        # Check required fields
        required_fields = self.categorical_features + self.numerical_features
        
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        if 'total_sqft' in input_data and input_data['total_sqft'] <= 0:
            errors.append("total_sqft must be positive")
        
        if 'bhk' in input_data and input_data['bhk'] <= 0:
            errors.append("bhk must be positive")
        
        if 'bath' in input_data and input_data['bath'] <= 0:
            errors.append("bath must be positive")
        
        if 'amenities_score' in input_data:
            score = input_data['amenities_score']
            if score < 0 or score > 10:
                errors.append("amenities_score must be between 0 and 10")
        
        return errors
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict price for a property.
        
        Args:
            input_data: Dictionary containing property features
            
        Returns:
            Dictionary with prediction results
        """
        print(f"Predicting price for {input_data.get('location', 'Unknown')}...")
        
        try:
            # Validate input
            errors = self.validate_input(input_data)
            if errors:
                return {
                    "success": False,
                    "errors": errors,
                    "prediction": None
                }
            
            # Convert to DataFrame (single row)
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            predicted_price = float(self.pipeline.predict(input_df)[0])
            
            # Calculate additional metrics
            price_per_sqft = predicted_price / input_data["total_sqft"]
            
            # Get yield percentage
            yield_pct = self.yield_lookup.get(
                input_data["location"], 
                3.5  # Default yield
            )
            
            # Calculate monthly rent (yield% of total price / 12)
            monthly_rent = (predicted_price * yield_pct / 100) / 12
            
            # Confidence interval (±10%)
            confidence_lower = predicted_price * 0.9
            confidence_upper = predicted_price * 1.1
            
            # Prepare response
            response = {
                "success": True,
                "errors": [],
                "prediction": {
                    "predicted_total_price": round(predicted_price, 2),
                    "predicted_price_per_sqft": round(price_per_sqft, 2),
                    "estimated_monthly_rent": round(monthly_rent, 2),
                    "estimated_annual_yield_pct": round(yield_pct, 2),
                    "confidence_interval_90": {
                        "lower": round(confidence_lower, 2),
                        "upper": round(confidence_upper, 2)
                    },
                    "property_details": {
                        "location": input_data["location"],
                        "property_type": input_data["property_type"],
                        "total_sqft": input_data["total_sqft"],
                        "bhk": input_data["bhk"],
                        "bath": input_data["bath"],
                        "amenities_score": input_data["amenities_score"]
                    }
                },
                "model_info": {
                    "model_type": self.best_model,
                    "features_used": self.categorical_features + self.numerical_features
                }
            }
            
            print(f"Prediction complete: ₹{predicted_price:,.0f}")
            
            return response
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "success": False,
                "errors": [f"Prediction failed: {str(e)}"],
                "prediction": None
            }
    
    def batch_predict(self, properties_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict prices for multiple properties.
        
        Args:
            properties_list: List of property dictionaries
            
        Returns:
            List of prediction results
        """
        print(f"Processing batch of {len(properties_list)} properties...")
        
        predictions = []
        
        for i, prop in enumerate(properties_list, 1):
            location = prop.get('location', f'Property_{i}')
            print(f"  Processing {i}/{len(properties_list)}: {location}")
            
            try:
                result = self.predict(prop)
                predictions.append(result)
            except Exception as e:
                predictions.append({
                    "success": False,
                    "errors": [f"Processing failed: {str(e)}"],
                    "prediction": None
                })
        
        print(f"Batch processing complete")
        
        return predictions


# Example usage and testing
def test_inference():
    """
    Test the inference engine with sample data.
    """
    print("Testing Property Price Predictor...")
    
    try:
        # Initialize predictor
        predictor = PropertyPricePredictor()
        
        # Test cases
        test_properties = [
            {
                "location": "Hebbal",
                "total_sqft": 1800,
                "bhk": 3,
                "bath": 3,
                "property_type": "Villa",
                "amenities_score": 7,
            },
            {
                "location": "Whitefield",
                "total_sqft": 1200,
                "bhk": 2,
                "bath": 2,
                "property_type": "Apartment",
                "amenities_score": 8,
            },
            {
                "location": "Indiranagar",
                "total_sqft": 1200,
                "bhk": 2,
                "bath": 2,
                "property_type": "Apartment",
                "amenities_score": 9,
            }
        ]
        
        print("\nMaking predictions...")
        results = predictor.batch_predict(test_properties)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        for i, (prop, result) in enumerate(zip(test_properties, results), 1):
            print(f"\nProperty {i}: {prop['location']} - {prop['property_type']}")
            print(f"  Size: {prop['total_sqft']} sqft, {prop['bhk']} BHK")
            
            if result['success']:
                p = result['prediction']
                print(f"  Price: ₹{p['predicted_total_price']:,.0f}")
                print(f"  Price/sqft: ₹{p['predicted_price_per_sqft']:,.0f}")
                print(f"  Rent: ₹{p['estimated_monthly_rent']:,.0f}/month")
                print(f"  Yield: {p['estimated_annual_yield_pct']}%")
                print(f"  Confidence: ₹{p['confidence_interval_90']['lower']:,.0f} - ₹{p['confidence_interval_90']['upper']:,.0f}")
            else:
                print(f"  Error: {', '.join(result['errors'])}")
        
        print("\n" + "=" * 60)
        print("Test complete")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_inference()