import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# GPU imports (with fallback)
try:
    from xgboost import XGBClassifier
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    XGBClassifier = None
    GPU_AVAILABLE = False

try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRandomForestClassifier
    from cuml.linear_model import LogisticRegression as cumlLogisticRegression
    from cuml.svm import SVC as cumlSVC
    CUML_AVAILABLE = True
except ImportError:
    cudf = cuml = cumlRandomForestClassifier = cumlLogisticRegression = cumlSVC = None
    CUML_AVAILABLE = False


class ChurnPredictionMLflow:
    def __init__(self, experiment_name="Advanced_Churn_Prediction"):
        """Initialize MLflow churn prediction pipeline with GPU optimization"""
        print("🔧 Initializing Churn Prediction Pipeline...")
        
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # GPU availability check - DIRECTLY IN INIT
        print("🔍 GPU durumu kontrol ediliyor...")
        self.gpu_available = {
            'cuda': False,
            'cuml': False,
            'xgboost_gpu': False
        }
        
        # CUDA check
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available['cuda'] = True
                print(f"✅ CUDA available: {torch.version.cuda}")
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("⚠️ CUDA not available, using CPU models")
        except Exception as e:
            print(f"⚠️ PyTorch error: {e}")
        
        # cuML check
        if CUML_AVAILABLE:
            self.gpu_available['cuml'] = True
            print("✅ RAPIDS cuML GPU support available")
        else:
            print("⚠️ RAPIDS cuML not found, CPU fallback will be used")
        
        # XGBoost GPU check
        try:
            if XGBClassifier is not None and self.gpu_available['cuda']:
                self.gpu_available['xgboost_gpu'] = True
                print("✅ XGBoost GPU support available")
            else:
                print("⚠️ XGBoost GPU not available, using CPU version")
        except Exception as e:
            print(f"⚠️ XGBoost GPU test failed: {e}")
        
        print(f"📊 GPU Status: {self.gpu_available}")
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        print("✅ MLflow experiment initialized")
    
    def load_and_prepare_data(self, data_path="data/raw/churn.csv"):
        """Load and prepare the churn dataset"""
        print("📊 Loading and preparing data...")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Data cleaning
        data = data.drop(columns=["customerID"], errors="ignore")
        
        # Handle TotalCharges if it exists and has issues
        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
            data = data.dropna(subset=["TotalCharges"])
        
        # Encode target variable
        data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
        
        # Separate features and target
        X = data.drop(columns=["Churn"])
        y = data["Churn"]
        
        # Identify feature types
        self.numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        print(f"   - Features: {self.X_train.shape[1]} ({len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical)")
        print(f"   - Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )
        return preprocessor
    
    def define_models(self):
        """Define all models to compare - including GPU optimized versions"""
        self.model_configs = {}
        
        # CPU Models (always available)
        self.model_configs["Logistic_Regression_CPU"] = {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__solver": ["liblinear", "lbfgs"]
            }
        }
        
        self.model_configs["Random_Forest_CPU"] = {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": [2, 5]
            }
        }
        
        self.model_configs["Gradient_Boosting_CPU"] = {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.1, 0.2],
                "classifier__max_depth": [3, 5]
            }
        }
        
        # XGBoost - GPU version if available
        if XGBClassifier is not None:
            if self.gpu_available.get('xgboost_gpu', False) and self.gpu_available.get('cuda', False):
                print("🚀 XGBoost GPU version will be used")
                self.model_configs["XGBoost_GPU"] = {
                    "model": XGBClassifier(
                        random_state=42, 
                        eval_metric='logloss',
                        tree_method='gpu_hist',
                        gpu_id=0
                    ),
                    "params": {
                        "classifier__n_estimators": [100, 200],
                        "classifier__learning_rate": [0.1, 0.2],
                        "classifier__max_depth": [3, 5]
                    }
                }
            else:
                self.model_configs["XGBoost_CPU"] = {
                    "model": XGBClassifier(random_state=42, eval_metric='logloss'),
                    "params": {
                        "classifier__n_estimators": [100, 200],
                        "classifier__learning_rate": [0.1, 0.2],
                        "classifier__max_depth": [3, 5]
                    }
                }
        
        # RAPIDS cuML GPU Models
        if self.gpu_available.get('cuml', False) and CUML_AVAILABLE:
            print("🚀 RAPIDS cuML GPU models will be added")
            
            try:
                self.model_configs["Logistic_Regression_GPU"] = {
                    "model": cumlLogisticRegression(max_iter=1000),
                    "params": {
                        "classifier__C": [0.1, 1, 10]
                    }
                }
                
                self.model_configs["Random_Forest_GPU"] = {
                    "model": cumlRandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    ),
                    "params": {
                        "classifier__n_estimators": [100, 200],
                        "classifier__max_depth": [10, 20],
                        "classifier__max_features": [0.3, 0.5]
                    }
                }
            except Exception as e:
                print(f"⚠️ Error adding cuML models: {e}")
        
        # CPU fallback models
        self.model_configs["SVM_CPU"] = {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["rbf", "linear"]
            }
        }
        
        self.model_configs["Decision_Tree"] = {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": [2, 5, 10]
            }
        }
        
        self.model_configs["Naive_Bayes"] = {
            "model": GaussianNB(),
            "params": {}
        }
        
        print(f"📊 Total {len(self.model_configs)} models defined")
        
        # Show GPU model info
        gpu_models = [name for name in self.model_configs.keys() if 'GPU' in name]
        if gpu_models:
            print(f"🎮 GPU accelerated models: {len(gpu_models)}")
            for model_name in gpu_models:
                print(f"   🚀 {model_name}")
        else:
            print("⚠️ No GPU models available, using CPU only")
    
    def train_and_evaluate_model(self, model_name, model_config, use_smote=True):
        """Train and evaluate a single model - GPU optimized"""
        print(f"\n🔄 {model_name} eğitiliyor...")
        
        # GPU memory cleanup if needed
        if 'GPU' in model_name and self.gpu_available.get('cuda', False):
            try:
                import torch
                torch.cuda.empty_cache()
                print(f"🧹 GPU memory temizlendi")
            except:
                pass
        
        with mlflow.start_run(run_name=model_name):
            # Create preprocessing pipeline
            preprocessor = self.create_preprocessor()
            
            # Create pipeline
            if use_smote and not any(x in str(type(model_config["model"])) for x in ['cuml', 'cuML']):
                # SMOTE only for sklearn models
                pipeline = ImbPipeline([
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", model_config["model"])
                ])
            else:
                pipeline = ImbPipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", model_config["model"])
                ])
            
            # Hyperparameter tuning
            start_time = pd.Timestamp.now()
            
            if model_config["params"]:
                print(f"   🎯 Hyperparameter tuning başlatılıyor...")
                
                # GPU models use fewer CV folds for memory efficiency
                cv_folds = 3 if 'GPU' in model_name else 5
                
                try:
                    grid_search = GridSearchCV(
                        pipeline, 
                        model_config["params"], 
                        cv=cv_folds, 
                        scoring='roc_auc',
                        n_jobs=1 if 'GPU' in model_name else -1,
                        verbose=0
                    )
                    grid_search.fit(self.X_train, self.y_train)
                    best_pipeline = grid_search.best_estimator_
                    
                    # Log best parameters
                    for param, value in grid_search.best_params_.items():
                        mlflow.log_param(param, value)
                        
                except Exception as e:
                    print(f"   ⚠️ Grid search başarısız, default parametreler kullanılıyor: {e}")
                    best_pipeline = pipeline
                    best_pipeline.fit(self.X_train, self.y_train)
            else:
                best_pipeline = pipeline
                best_pipeline.fit(self.X_train, self.y_train)
            
            training_time = (pd.Timestamp.now() - start_time).total_seconds()
            print(f"   ⏱️ Eğitim süresi: {training_time:.2f} saniye")
            
            # Make predictions
            y_pred_train = best_pipeline.predict(self.X_train)
            y_pred_test = best_pipeline.predict(self.X_test)
            y_pred_proba_train = best_pipeline.predict_proba(self.X_train)[:, 1]
            y_pred_proba_test = best_pipeline.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "train_accuracy": accuracy_score(self.y_train, y_pred_train),
                "test_accuracy": accuracy_score(self.y_test, y_pred_test),
                "train_precision": precision_score(self.y_train, y_pred_train),
                "test_precision": precision_score(self.y_test, y_pred_test),
                "train_recall": recall_score(self.y_train, y_pred_train),
                "test_recall": recall_score(self.y_test, y_pred_test),
                "train_f1": f1_score(self.y_train, y_pred_train),
                "test_f1": f1_score(self.y_test, y_pred_test),
                "train_roc_auc": roc_auc_score(self.y_train, y_pred_proba_train),
                "test_roc_auc": roc_auc_score(self.y_test, y_pred_proba_test),
                "training_time": training_time
            }
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model info
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("use_gpu", 'GPU' in model_name)
            mlflow.log_param("train_samples", len(self.y_train))
            mlflow.log_param("test_samples", len(self.y_test))
            
            # GPU memory info
            if 'GPU' in model_name and self.gpu_available.get('cuda', False):
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                        mlflow.log_metric("gpu_memory_used_gb", gpu_memory_used)
                        print(f"   💾 GPU Memory kullanılan: {gpu_memory_used:.2f}GB")
                except:
                    pass
            
            # Create and save visualizations
            self.plot_confusion_matrix(self.y_test, y_pred_test, model_name)
            self.plot_roc_curve(self.y_test, y_pred_proba_test, model_name)
            
            # Save model
            model_path = f"models/{model_name}_model.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_pipeline, model_path)
            mlflow.log_artifact(model_path)
            
            # Log model with MLflow
            mlflow.sklearn.log_model(best_pipeline, f"{model_name}_model")
            
            # Store model results
            self.models[model_name] = {
                "pipeline": best_pipeline,
                "metrics": metrics,
                "model_path": model_path
            }
            
            # Check if this is the best model
            test_roc_auc = metrics["test_roc_auc"]
            if test_roc_auc > self.best_score:
                self.best_score = test_roc_auc
                self.best_model = model_name
                # Save best model separately
                best_model_path = "models/best_churn_model.pkl"
                joblib.dump(best_pipeline, best_model_path)
            
            print(f"   ✅ {model_name} tamamlandı!")
            print(f"      - Test ROC AUC: {test_roc_auc:.4f}")
            print(f"      - Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"      - Test F1: {metrics['test_f1']:.4f}")
            print(f"      - Eğitim Süresi: {training_time:.2f}s")
            
            # GPU memory cleanup
            if 'GPU' in model_name and self.gpu_available.get('cuda', False):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            
            return best_pipeline, metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Create and save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["No Churn", "Churn"], 
                   yticklabels=["No Churn", "Churn"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        
        # Save plot
        os.makedirs("reports", exist_ok=True)
        plot_path = f"reports/{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Create and save ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = f"reports/{model_name}_roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)
    
    def run_model_comparison(self, data_path="data/raw/churn.csv"):
        """Run complete model comparison pipeline"""
        print("🚀 MLflow Model Karşılaştırma Pipeline'ı Başlatılıyor")
        print("=" * 60)
        
        # Show GPU info
        if self.gpu_available.get('cuda', False):
            try:
                import torch
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            except:
                print("🎮 GPU: Available but details unavailable")
        else:
            print("💻 Using CPU only")
        
        # Load data
        X, y = self.load_and_prepare_data(data_path)
        
        # Define models
        self.define_models()
        
        # Train all models
        print(f"\n🤖 {len(self.model_configs)} farklı model eğitiliyor...")
        
        total_start_time = pd.Timestamp.now()
        successful_models = 0
        
        for model_name, model_config in self.model_configs.items():
            try:
                model_start_time = pd.Timestamp.now()
                self.train_and_evaluate_model(model_name, model_config)
                model_time = (pd.Timestamp.now() - model_start_time).total_seconds()
                successful_models += 1
                
                print(f"   💪 {model_name}: {model_time:.1f}s")
                
                # Show GPU memory status
                if self.gpu_available.get('cuda', False) and 'GPU' in model_name:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated(0) / 1024**3
                            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                            print(f"      💾 GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
                    except:
                        pass
                    
            except Exception as e:
                print(f"❌ {model_name} eğitim hatası: {str(e)}")
                continue
        
        total_time = (pd.Timestamp.now() - total_start_time).total_seconds()
        
        # Create comparison report
        self.create_comparison_report()
        
        print(f"\n🏆 En İyi Model: {self.best_model}")
        print(f"🎯 En İyi ROC AUC Skoru: {self.best_score:.4f}")
        print(f"💾 En iyi model şuraya kaydedildi: models/best_churn_model.pkl")
        print(f"⏱️ Toplam süre: {total_time/60:.1f} dakika")
        print(f"✅ Başarılı modeller: {successful_models}/{len(self.model_configs)}")
        
        if self.gpu_available.get('cuda', False):
            print(f"🎮 GPU kullanım istatistikleri MLflow'da görüntülenebilir")
        
        print("\n✅ Model karşılaştırması tamamlandı!")
        
        return self.best_model, self.best_score
    
    def create_comparison_report(self):
        """Create comprehensive comparison report"""
        print("\n📊 Creating model comparison report...")
        
        # Prepare comparison data
        comparison_data = []
        for model_name, model_info in self.models.items():
            metrics = model_info["metrics"]
            comparison_data.append({
                "Model": model_name,
                "Test_ROC_AUC": metrics["test_roc_auc"],
                "Test_Accuracy": metrics["test_accuracy"],
                "Test_Precision": metrics["test_precision"],
                "Test_Recall": metrics["test_recall"],
                "Test_F1": metrics["test_f1"],
                "Train_ROC_AUC": metrics["train_roc_auc"],
                "Overfitting": metrics["train_roc_auc"] - metrics["test_roc_auc"],
                "Training_Time": metrics["training_time"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Test_ROC_AUC", ascending=False)
        
        # Save comparison report
        os.makedirs("reports", exist_ok=True)
        comparison_df.to_csv("reports/model_comparison.csv", index=False)
        
        # Create comparison visualization
        self.plot_model_comparison(comparison_df)
        
        # Print comparison table
        print("\n📋 Model Comparison Results:")
        print("-" * 100)
        print(comparison_df.round(4).to_string(index=False))
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Create model comparison visualizations"""
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC AUC comparison
        axes[0, 0].barh(comparison_df["Model"], comparison_df["Test_ROC_AUC"])
        axes[0, 0].set_xlabel("ROC AUC Score")
        axes[0, 0].set_title("Test ROC AUC Comparison")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        axes[0, 1].barh(comparison_df["Model"], comparison_df["Test_Accuracy"])
        axes[0, 1].set_xlabel("Accuracy Score")
        axes[0, 1].set_title("Test Accuracy Comparison")
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score comparison
        axes[1, 0].barh(comparison_df["Model"], comparison_df["Test_F1"])
        axes[1, 0].set_xlabel("F1 Score")
        axes[1, 0].set_title("Test F1 Score Comparison")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time comparison
        axes[1, 1].barh(comparison_df["Model"], comparison_df["Training_Time"])
        axes[1, 1].set_xlabel("Training Time (seconds)")
        axes[1, 1].set_title("Training Time Comparison")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("reports/model_comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow (create a new run for comparison)
        with mlflow.start_run(run_name="Model_Comparison_Summary"):
            mlflow.log_artifact("reports/model_comparison.csv")
            mlflow.log_artifact("reports/model_comparison_plots.png")
            mlflow.log_metric("best_model_score", self.best_score)
            mlflow.log_param("best_model_name", self.best_model)
            mlflow.log_param("total_models_trained", len(self.models))


def main():
    """Main execution function"""
    # Initialize the pipeline
    churn_pipeline = ChurnPredictionMLflow("Advanced_Churn_Prediction")
    
    # Run the complete comparison
    best_model, best_score = churn_pipeline.run_model_comparison("data/raw/churn.csv")
    
    print("\n" + "="*60)
    print("🎉 PIPELINE BAŞARIYLA TAMAMLANDI!")
    print("="*60)
    print(f"🏆 En İyi Model: {best_model}")
    print(f"🎯 En İyi ROC AUC: {best_score:.4f}")
    print("\n📁 Oluşturulan Dosyalar:")
    print("   - models/best_churn_model.pkl (En iyi model)")
    print("   - models/*_model.pkl (Tüm eğitilmiş modeller)")
    print("   - reports/model_comparison.csv (Karşılaştırma sonuçları)")
    print("   - reports/model_comparison_plots.png (Görselleştirmeler)")
    print("   - reports/*_confusion_matrix.png (Confusion matrixler)")
    print("   - reports/*_roc_curve.png (ROC eğrileri)")
    print("\n🔗 MLflow UI: 'mlflow ui' komutuyla detaylı sonuçları görüntüle")


if __name__ == "__main__":
    main()