"""
Random Forest Classification Algorithm Implementation
------------------------------------------------------
Random Forest is an ensemble learning method that operates by constructing
multiple decision trees during training and outputting the mode of classes
(classification) of individual trees.

Advantages:
- High accuracy and robust performance
- Handles large datasets with high dimensionality
- Reduces overfitting compared to single decision trees
- Provides feature importance rankings
- Works well with missing data
- Can handle both numerical and categorical features

Disadvantages:
- Can be slow on large datasets
- Less interpretable than single decision trees
- Requires more memory
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple, Any
import joblib


class RandomForestClassificationModel:
    """
    A wrapper class for Random Forest Classification algorithm.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest Classification model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of the tree
        min_samples_split : int
            Minimum number of samples required to split an internal node
        min_samples_leaf : int
            Minimum number of samples required to be at a leaf node
        max_features : str or int
            Number of features to consider when looking for the best split
        random_state : int
            Random state for reproducibility
        n_jobs : int
            Number of jobs to run in parallel (-1 uses all processors)
        """
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """
        Fit the Random Forest model.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
        feature_names : list, optional
            Names of features
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"Model trained successfully with {self.model.n_estimators} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like
            Data to predict
        
        Returns:
        --------
        predictions : array
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like
            Data to predict
        
        Returns:
        --------
        probabilities : array
            Predicted probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        X : array-like
            Test data
        y : array-like
            True labels
        
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            try:
                y_proba = self.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
            except:
                pass
        
        return metrics
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return
        
        Returns:
        --------
        importance_df : DataFrame
            DataFrame with features and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        figsize : tuple
            Figure size
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        X : array-like
            Test data
        y : array-like
            True labels
        figsize : tuple
            Figure size
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.predict(X)
        cm = confusion_matrix(y, predictions)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target labels
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        cv_results : dict
            Cross-validation results with accuracy scores
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scoring_metric': 'accuracy'
        }
    
    def hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target labels
        param_grid : dict, optional
            Parameter grid for search
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        results : dict
            Best parameters and best score
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'RandomForestClassificationModel':
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        
        Returns:
        --------
        model : RandomForestClassificationModel
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def example_classification():
    """Example usage for Random Forest Classification."""
    print("=" * 60)
    print("Random Forest Classification Example")
    print("=" * 60)
    
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    rf_model = RandomForestClassificationModel(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    print("\nTraining model...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = rf_model.evaluate(X_test, y_test)
    print("\nClassification Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance_df = rf_model.get_feature_importance(top_n=10)
    print(importance_df.to_string(index=False))
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = rf_model.cross_validate(X_train, y_train, cv=5)
    print(f"  Mean CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")


if __name__ == "__main__":
    # Run example
    example_classification()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
