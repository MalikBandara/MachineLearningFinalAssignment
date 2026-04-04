<div align="center">

# Final Group Project Technical Report
## AuraCart Retail Analytics Predictive Engine

**Module:** ITS 2140: Machine Learning: Foundations to Production Systems  
**Date:** April 2026  
**Group:** Group 13

### Group Members
| Student Name | Registration Number |
| :--- | :--- |
| Sandunil Malik Bandara | 2301691001 |
| Sachini Poornima | 2301691034 |
| Rashmika Navod | 2301691093 |
| Chamath Dilshan | 2301691010 |
| Malka Samarakoon | 2301691016 |
| Lakmal Kumarasiri | 2301691036 |

</div>

<br><br>

---

## 1. Executive Summary

AuraCart Retail Analytics, a rapidly scaling digital e-commerce platform, recently identified a critical necessity to transition away from intuition-based decision-making methodologies toward a highly automated, data-centric predictive system. Facing unprecedented transaction volumes and highly volatile supply chain logistics, the organization's executive leadership mandated the design and deployment of a multi-faceted machine learning architecture capable of dynamic revenue forecasting, customer segmentation, logistical bottleneck prediction, and targeted behavioral clustering. 

This technical report delineates the end-to-end development life-cycle of the intelligent predictive analytics engine commissioned by AuraCart. To accomplish the mandate, an external Machine Learning Engineering team meticulously analyzed the "E-commerce Customer Order Behavior Dataset", encompassing 10,000 distinct transactional records. Our team engineered a unified solution comprising three concurrent algorithmic pillars: Continuous Regression for pricing predictions, Multi-class Classification for customer segment and delivery status forecasting, and Unsupervised Clustering to unveil hidden buyer behaviors. 

Our findings yielded highly actionable business intelligence. We identified that 'Returned' orders—which act as a heavy operational burden—could be dynamically predicted and flagged using Softmax Regression integrated with calibrated class-weighting penalties to circumvent the dataset’s inherent class imbalance. Concurrently, an integrated K-Means algorithmic pipeline unveiled complex latency disparities, cleanly segmenting customers by their shipping delay experiences and purchasing volumes, thereby allowing for hyper-targeted marketing and retention interventions. 

Beyond isolated offline model metrics, the ultimate success of the project rested upon seamless productionization. To achieve this, the entire experimental workflow was rigorously cataloged using the MLflow software ecosystem, ensuring absolute traceability of hyperparameter tuning grids and evaluation artifacts. The finalized champion model—encapsulating an entire Scikit-learn feature-engineering pipeline alongside a multi-class predictive estimator—was successfully serialized into a standardized `model.joblib` format. This monolithic artifact was subsequently migrated to a Google Cloud Storage environment and formally deployed as a scalable, high-throughput RESTful inference endpoint via Google Cloud Vertex AI infrastructure. The resulting architecture empowers AuraCart’s frontend microservices to query the predictive models deterministically in real-time, resolving the strategic mandate and laying the foundation for a deeply data-driven corporate ecosystem.

---

## 2. Introduction and Business Context

Modern digital commerce operates on razor-thin margins heavily dictated by logistical efficiency and highly personalized user experiences. AuraCart Retail Analytics, currently operating as a mid-tier digital retail platform, has observed severe operational friction explicitly resulting from manual heuristics and lagging historical reporting mechanisms. As user demographics grow increasingly heterogeneous, relying on generalized logic to predict delivery failure, assess product pricing elasticity, and target promotional materials translates rapidly to compounding financial losses.

The executive mandate authorized for this project is designed to comprehensively solve these shortfalls through the conception, mathematical optimization, and cloud deployment of an intelligent forecasting system. The objectives assigned to the engineering effort operate across three algorithmic verticals:
1. **Dynamic Forecasting (Regression):** Predict the final monetary value of incoming unpriced transactions to enable dynamic forecasting and optimize inventory liquidity.
2. **Logistical and Risk Mitigation (Classification):** Categorize unstructured transactions into proactive delivery status outcomes (Delivered, Shipped, Pending, Returned) to act as an early-warning mechanism. Concurrently, an equivalent system categorizes the buyer automatically into distinct segments (VIP, Returning, New).
3. **Behavioral Target Clustering (Unsupervised Learning):** Process incoming numerical features through unsupervised clustering mechanisms to find latent similarities unidentifiable by human supervisors, generating profiles to drive precision-marketing.

This project relies exclusively upon the `millat/e-commerce-orders` dataset sourced from the Hugging Face hub, featuring 10,000 granular commercial records. This dataset captures the inherent messiness and heterogeneous class distribution reflective of real-world operational realities, necessitating robust data-engineering tactics. The final objective extends significantly beyond localized algorithm success, requiring full encapsulation within MLflow architectural tracking and successful cloud-native endpoint hosting within Google Cloud Vertex AI, ensuring AuraCart attains a scalable production asset.

---

## 3. Data Exploration and Preprocessing

A rigorous Exploratory Data Analysis (EDA) framework was executed to extract statistical characteristics, inspect distributions, and structure an impermeable pipeline to condition data prior to mathematical optimization.

### 3.1 Analysis of Continuous and Categorical Variables
The dataset consists of multiple continuous descriptors alongside qualitative categorical groupings. During initial analysis, the continuous `price` target variable exhibited a multi-modal distribution ranging from $5 to $500. `Quantity` operated optimally as a discrete continuous variable, showing uniformity across values 1 through 10. In terms of correlation, we utilized Pearson and Spearman coefficient matrices to identify redundant features, resulting in the exclusion of arbitrary identifiers (e.g., `order_id`, `customer_id`, `product_id`) which inherently possess zero predictive validity and inflate the dataset's dimensionality arbitrarily. Temporal features such as `order_date` and `shipping_date` were unpacked using feature engineering to extract mathematically operable representations including `order_month`, `order_day_of_week`, `order_hour`, and crucially, a functional `shipping_delay_days` latency measure.

The analysis unveiled a severe distribution anomaly across the categorical target definitions, directly mirroring enterprise reality:
* **Delivery Status Imbalance:** Deliveries operated successfully (Delivered) in ~70% of rows, whereas operational failures (Returned) and pending orders sat at ~5%. 
* **Customer Segment Imbalance:** The platform consists largely of 'New' customers (~50%), while high-value 'VIP' buyers occupy purely ~15% of the database.

Feeding unchecked imbalanced distributions into optimization loss functions naturally causes naive predictive classifiers to unilaterally guess the majority class to minimize loss via the path of least resistance. To counteract this, aggressive class-weight penalization mechanisms were necessary for the classification phases over synthetic minority oversampling (SMOTE) logic, guaranteeing models were forcibly audited on minority classification accuracy.

### 3.2 Constructing the Scikit-learn Preprocessing Pipeline
Maintaining inference-time consistency is the bedrock of MLOps. If raw production data is handled inconsistently against historical training data logic, the phenomenon of "training-serving skew" severely degradates predictive performance. This risk necessitates an immovable preprocessing architecture.

We utilized the `sklearn.pipeline.Pipeline` coupled seamlessly with an encapsulated `ColumnTransformer`. 
1. **Numerical Standardization:** Continuous traits (price, quantity, delay days) were funneled through a `StandardScaler`. This aligns feature variances iteratively to a standard Gaussian curve (mean zero, unit variance). K-Means distance calculations (Euclidean metrics) and Stochastic Gradient Descent optimizations are hypersensitive to non-scaled features; standardization guarantees uniform learning topologies.
2. **Categorical Matrix Transformation:** Categorical inputs including `payment_method`, `device_type`, and `category` possessed no intrinsic ordinal rank. Consequently, passing them through arbitrary digit assignments would unintentionally compel the mathematical loss functions to identify false numeric magnitude (e.g. PayPal > Credit Card). An independent `OneHotEncoder` processed these nodes exclusively, transforming them into orthogonal sparse arrays. 

By unifying the Transformer directly into the serialized predictive artifact, future AuraCart microservices can transmit entirely raw JSON parameters—including raw string names and raw pricing—directly to the server without needing independent transformation mechanics.

---

## 4. Regression and Classification Modeling

The foundation of the predictive analytics engine relies heavily on generalized supervised methodologies explicitly aligned with Modules 3 and 4 of the syllabus.

### 4.1 Continuous Price Prediction (Multiple Linear Regression)
Pricing fluidity requires the inference of continuous financial variables via Multiple Linear Regression. To bypass historical processing constraints imposed by Closed-Form Ordinary Least Squares matrices across massive matrices, we deployed the `SGDRegressor` representing optimization via Stochastic Gradient Descent. 

Gradient descent optimizes the multiple linear weights of the algorithm progressively processing isolated batches of features to find the global minimum within the residual loss surface. Hyperparameter tuning was central to identifying an optimal path without overshooting the global minimum or descending too slowly. We explicitly tracked several parameters, including dynamic learning rates (`eta0`) and epoch bounds (`max_iter`). Cross-validation operations (K-Fold, k=5) proved instrumental in mitigating data-leakage while verifying that the regressor dynamically resisted statistical overfitting, stabilizing test loss within a highly acceptable standard deviation. MLflow documented the iterations entirely, preventing experimentation bloat.

### 4.2 Multi-Class Customer Classification (Softmax Regression)
Delineating granular traits like `customer_segment` relies inherently upon bounding mechanisms; typical continuous regression assumes infinite scale bounds and is mathematically ill-equipped to provide probabilistic output certainty on isolated, discrete text classes. By leveraging `LogisticRegression` within native Scikit-Learn libraries (utilizing Softmax/Multinomial properties), we formulated a robust classifier. 

The algorithmic core relies sequentially upon the Softmax probability function, transmuting raw unbounded regression predictions (logits) into a probability array distributed mathematically across the defined K classes, strictly summing to 1. The architecture utilizes Categorical Cross-Entropy (Log-Loss) as the objective optimization metric, penalizing the algorithm significantly when confident misclassifications are executed against the true labels.

Crucially, addressing the previously identified distribution imbalance involved strategically passing `class_weight='balanced'` down into the solver parameters. Instead of using SMOTE to artificially clone data points, balancing class weights directly altered the gradient loss values iteratively based heavily on categorical rarity. When the system incorrectly predicted a 'Returned' or 'VIP' tag, the algorithmic loss multiplier was statistically magnified, mathematically forcing the gradient to respect the boundaries of minority segments instead of uniformly defaulting to the 'Delivered' or 'New' cohort. 

---

## 5. Model Evaluation and Performance Analysis

Evaluation of an algorithmic endpoint mandates a contextual analysis bridging raw statistical metrics with direct AuraCart financial priorities.

### 5.1 Regressive Loss Evaluation
In pricing forecasting, we measured structural accuracy comparing Mean Squared Error (MSE) and Mean Absolute Error (MAE). While MAE calculates a simple linear average of regression variance, MSE actively squares error magnitudes. From AuraCart’s corporate standpoint, a marginal price deviation of $2 acts purely as noise, whereas a catastrophic algorithmic deviation of $100 dramatically slashes revenue margins and creates customer vitriol. MSE naturally penalizes severe deviations aggressively, thereby ensuring our selected model is optimized against extreme failures. Because the finalized `SGDRegressor` showcased closely aligned behavior traversing both training MSE and historical test MSE within 5-fold iterations, the architecture was definitively proven to possess a highly sustainable bias-variance tradeoff free from systemic underfitting.

### 5.2 Classification Performance and Asymmetric Risk
Raw generalized accuracy is a dangerously flawed metric representing structurally imbalanced databases. The algorithm may theoretically reach standard accuracies exceeding 70% by blindly predicting 'Delivered' across everything.

Consequently, comprehensive evaluations relied exclusively upon macro-averaged components mapping precision (true positives mapped against total positive algorithmic predictions) and recall (true positives mapped recursively against complete factual positives from reality). The detailed macro F1-score computation (acting as the harmonic mean bridging precision and recall) explicitly validated the functionality of our algorithm over the `VIP` user base that natively occupies only 15% of the total table.

Simultaneously, careful mapping via a Confusion Matrix helped identify specific misclassification clusters involving delivery delays. Evaluating the `Returned` cohort allowed engineering teams to explicitly investigate Asymmetric Risk priorities. In the context of e-commerce logisitics:
* **A False Positive (Classifying a successful delivery as a Return risk):** The system flags an order unnecessarily, incurring minor review costs.
* **A False Negative (Classifying a guaranteed Return as a successful delivery):** The system fails to predict an incoming refund loop, destroying shipping budgets and deeply eroding customer satisfaction metrics.

Evaluating the Confusion Matrix definitively proved the necessity of operating via a high-recall, lower-precision threshold paradigm over minority failure cases. Over-predicting returns protects enterprise liquid assets, a strategy made viable dynamically directly through Softmax probability outputs (`predict_proba`) over rigid Boolean classification outputs.

---

## 6. Customer Behavior Clustering 

Where supervised models are bounded strictly by labeled historical assumptions, Unsupervised Clustering possesses the capability to group raw numerical feature topographies automatically devoid of pre-existing taxonomy, surfacing deeply hidden patterns within AuraCart’s consumer behavior matrix. 

### 6.1 Algorithmic Distance and Cluster Calculation
We initialized the robust `K-Means++` iteration structure over completely scaled continuous elements including pricing arrays, volume capacities, and temporal metadata. Determining the statistically optimal `K` clusters natively requires mapping the convergence metrics recursively. 
First, we observed the Within-Cluster Sum of Squares (WCSS) plotted across progressive iterations corresponding to the classic 'Elbow Method', denoting the mathematical threshold yielding drastically diminishing returns on data cohesiveness. This mechanism was independently verified across parallel execution loops logging the overarching mathematical 'Silhouette Score'. A higher silhouette parameter directly proves that aggregated data points are fundamentally closer internally to their respective geometric centroid points whilst staying tangibly separated linearly from competitive adjacent clusters. Following evaluation execution on MLflow, the mathematical silhouette optimization algorithm inherently determined ten distinctive geometric cluster classifications.

### 6.2 Advanced Business Segment Interpretations 
While ten isolated clusters provide pure mathematical perfection, we compressed the centroids into distinct macro-interpretations allowing the AuraCart growth marketing verticals to take meaningful strategic operations.

* **The Premium High-Value Cohorts (VIP):** Highlighted heavily around Clusters 2, 3, 6, and 8, these consumer cohorts demonstrate immense single-order value spanning $350 - $375, yet only maintain marginal operational volume sizes ranging around 1.6 items. Crucially, algorithms distinctly uncovered latency bifurcations within this segment. Specifically, while subsets engaged highly with rapid delivery times (delay 2.3 days), large swathes within this premium matrix suffer immense operational delays extending to 5.6 days. AuraCart customer success operators must strictly target delayed high-value subsets with rapid apology initiatives and targeted store-credit to prevent catastrophic VIP churn rates.
* **The B2B Volume Consumers:** Clusters 4 and 9 showcased entirely disconnected procurement volumes compared against standard baseline models. Buying drastically increased item volumes representing 4.1 to 4.6 factors, their total transaction liquidations dramatically eclipse regular shoppers. Identifying wholesale or business entities buying during the central periods of the week unlocks vast dedicated account management, bulk discount campaigns, and optimized freight tracking logistics.
* **The Baseline Consumer Segment:** The remainder of the geometric centroids defined classical baseline consumers transacting small ticket items spanning smaller nominal values ~ $130 - $140. Targeted promotional campaigns such as BOGO (Buy One Get One) mechanisms natively map strategically here to incrementally accelerate low cart capacities.

---

## 7. Production Deployment and MLOps Workflow

Building machine learning models that remain isolated in localized offline `.ipynb` environments fundamentally provides zero production value. Therefore, achieving rigorous programmatic interoperability executing enterprise-grade MLOps processes remains paramount.

### 7.1 MLOps: MLflow Experimentation Tracking
Before advancing a champion model into production stages, deterministic traceability of configuration metrics ensures model accountability. The MLflow ecosystem was programmatically initialized locally within the project directory across every regressive and clustered execution layer. The API (`mlflow.start_run()`) systematically captured parameters (Solver definitions, regularization logic `C`, iteration bounds), output metrics (R-Squared, F1, Log-Loss), alongside deeply localized execution constraints. 

Querying the registry programmatically revealed the pre-eminent champion architecture possessing the foremost cross-validated macro-F1 representation metrics over the highly contested Customer Segment feature columns, setting the baseline for pipeline instantiation.

### 7.2 Unified Pipeline Standardization & Serialization
In traditional offline modeling, engineers usually process raw test sets globally natively, causing immediate failure when solitary JSON components traverse the network. To achieve architectural robustness, the `best_segment_classifier` retrieved via MLflow was algorithmically fused sequentially with the preconfigured `ColumnTransformer` explicitly creating a centralized nested `sklearn.pipeline.Pipeline` monolith artifact. The finalized binary was successfully serialized over localized disk infrastructure as a `model.joblib` file alongside a strictly scoped Python package definition dependency file (`requirements.txt`).

### 7.3 Google Cloud Delivery & Vertex API Integration
Scaling AuraCart operations across global server grids natively requires serverless Kubernetes functionality provided sequentially by Google Cloud Storage (GCS) and Google Cloud Vertex AI infrastructure.

Deploying the final engine inherently observed the following pipeline processes:
1. **Cloud Migration Storage:** Deep execution metrics automatically utilized the Google Cloud SDK `storage.Client()` architectures directly initializing the transfer of our independent `model.joblib` artifact binaries targeting raw storage buckets.
2. **Containerized Import Processing:** Vertex AI Model Registry accepted the storage URIs combined symmetrically against a prebuilt `us-docker.pkg.dev/vertex-ai/prediction/sklearn` container matrix. Leveraging a managed serving engine mitigates arbitrary backend infrastructure dependencies entirely, preventing Docker maintenance bottlenecks natively.
3. **Inference Live Server Delivery:** Finally, the centralized logic utilized `model.deploy` generating an automated public RESTful endpoint accessible via frontend API interactions. 

Comprehensive manual simulations locally verified output determinism natively, accepting raw JSON feature arrays exactly simulating external system inputs, automatically applying standard scaling features, one-hot execution transforms, executing logistical predictions via probabilistic extraction equations, and yielding the specific inverse-transformed `customer_segment` response tags sequentially returning to the local user space successfully.

---

## 8. Conclusion and Future Work

AuraCart Retail Analytics successfully transitioned from historical lagging processes into a powerful enterprise AI forecasting institution. By utilizing comprehensive continuous regression, softmax categorization, and mathematically optimized demographic clustering arrays over unstructured operational realities, the operational bottleneck was distinctly analyzed and mitigated. The strategic execution seamlessly bridged robust offline testing paradigms iteratively against production-grade cloud architectural hosting logic utilizing Google Vertex AI container deployments guaranteeing scale potential efficiently alongside robust MLflow logging mechanisms. 

Moving forward, to further optimize operations sequentially, AuraCart should pursue several future enhancement verticals. Incorporating sophisticated temporal sequence analysis including explicitly deploying Recurrent Neural Networks (RNN) or state-of-the-art transformer architectures using raw streaming time-series purchase metadata will allow more acute seasonal prediction matrices over linear algorithms natively. Furthermore, executing MLOps CI/CD triggers utilizing specific model drift detection logic ensures automated retraining loops when incoming JSON distributions structurally decouple compared against current mathematical centroids, retaining flawless predictive integrity deep into structural scaling.

Overall, the deployed platform serves precisely against the operational mandates designated by the group objective, completing the foundational integration requirements entirely.

---
*End of Technical Report*
