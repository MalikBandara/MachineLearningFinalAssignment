# AuraCart Retail Analytics — Production-Grade ML & MLOps System

**ITS 2140: Machine Learning | Group Project | Semester 4, 2026**
**Team Async** | IJSE

---

## Project Overview

This project implements a production-grade machine learning system for **AuraCart**, a rapidly growing e-commerce platform. The system integrates three core ML approaches — regression, classification, and clustering — and deploys a live prediction service via **Google Cloud Vertex AI**.

---

## Repository Structure

```
TeamAsyncMLProject/
│
├── notebooks/
│   ├── 1_eda_and_preprocessing.ipynb    # EDA, feature engineering, preprocessing pipeline
│   ├── 2_supervised_modeling.ipynb      # Regression & classification with MLflow tracking
│   ├── 3_unsupervised_clustering.ipynb  # K-Means clustering, elbow method, centroid analysis
│   └── 4_mlops_deployment.ipynb         # GCS upload, Vertex AI model deployment
│
├── artifacts/
│   ├── model.joblib                     # Final unified prediction pipeline (scikit-learn 1.3.2)
│   └── requirements.txt                 # Dependency versions for cloud deployment
│
├── analysis_images/                     # All EDA and model evaluation plots
│   ├── categorical_features_distribution.png
│   ├── cluster_analysis.png
│   ├── cluster_crosstab.png
│   ├── clustering_optimal_k.png
│   ├── confusion_matrix_delivery.png
│   ├── confusion_matrix_segment.png
│   ├── correlation_matrices.png
│   ├── price_by_targets.png
│   ├── price_distribution.png
│   ├── quantity_distribution.png
│   ├── regression_evaluation.png
│   └── target_variable_distributions.png
│
├── OurReport.pdf                        # Final project report
├── assignment.pdf                       # Project specification
└── README.md                            # This file
```

---

## Machine Learning Tasks

| Task | Algorithm | Target Variable |
|---|---|---|
| Price Prediction | SGD Regressor | `price` (continuous) |
| Customer Segmentation | Random Forest Classifier | `customer_segment` (New/Returning/VIP) |
| Delivery Prediction | Logistic Regression (Softmax) | `delivery_status` |
| Behavioral Clustering | K-Means (k=10) | Unlabeled customer groups |

---

## Cloud Deployment

- **Platform:** Google Cloud Vertex AI
- **Region:** `asia-southeast1`
- **Container:** Scikit-learn 1.3 (Pre-built)
- **Endpoint ID:** `2983601767784120320`
- **Project ID:** `613147120640`

### Live Prediction Result
```
✅ PREDICTION SUCCESSFUL!
===================================
Prediction Result ID: 2
Customer Segment    : VIP
===================================
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install scikit-learn==1.3.2 pandas==2.0.3 numpy==1.26.4 joblib==1.3.2 mlflow
```

### 2. Run Notebooks (in order)
```bash
# Run each notebook top-to-bottom without errors
jupyter notebook notebooks/1_eda_and_preprocessing.ipynb
jupyter notebook notebooks/2_supervised_modeling.ipynb
jupyter notebook notebooks/3_unsupervised_clustering.ipynb
jupyter notebook notebooks/4_mlops_deployment.ipynb
```

### 3. Dataset
Dataset sourced from Hugging Face: [`millat/e-commerce-orders`](https://huggingface.co/datasets/millat/e-commerce-orders)
- 10,000 transactional records
- Features: `quantity`, `price`, `category`, `payment_method`, `device_type`, `channel`, etc.

---

## Team Members

| Name | Student ID |
|---|---|
| Sandunil Malik Bandara (Leader) | 2301691016 |
| Rashmika Navod | 2301691001 |
| Lakmal Kumarasiri | 2301691034 |
| Chamath Dilshan | 2301691093 |
| Sachini Poornima | 2301691010 |
| Kalindu Akalanka | 2301691005 |

---

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Dataset: millat/e-commerce-orders](https://huggingface.co/datasets/millat/e-commerce-orders)
