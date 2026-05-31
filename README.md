🎓 Student Academic Performance Prediction Using Deep Learning

📌 Project Title

Student Academic Performance Prediction Using Deep Learning and Behavioral Data Analysis

📖 Brief Summary of the Project

This project develops a Deep  learning-based system to predict student academic performance using behavioral, demographic, and educational factors. The model analyzes student-related data to identify patterns that influence academic success and provides predictive insights that can assist educators in making informed decisions.

🌟 Project Overview

Educational institutions often face challenges in identifying students who may struggle academically. Early prediction of academic performance can help educators implement targeted interventions and support systems.

This project utilizes Deep learning techniques to analyze student behavior and academic records, enabling the prediction of future academic outcomes. A user-friendly web interface is developed to allow users to input student information and obtain performance predictions.

❗ Problem Statement

Many students experience academic difficulties due to various factors such as attendance, study habits, participation, and personal circumstances. Traditional evaluation methods may not identify at-risk students early enough.

The objective of this project is to:

Predict student academic performance using machine/Deep learning algorithms.
Identify key factors affecting academic success.
Provide data-driven insights to educators and institutions.
Support early intervention strategies for improving student outcomes.

📊 Dataset

The project uses the Student Academic Behavior Dataset, which contains information related to:

Academic factors


Attendance records

Study hours

previous gpa

Tutoring Sessions / Week

wellbeing fectors 

Sleep Hours / Night

Stress Level

Screen Time / Day

Exam Anxiety 

Demographic factors 

gender 

age

part-time job

study method

diet quality 

internet quality

Extracurricular

family income

📊 Dataset Feature Descriptions

🎓 Academic Factors

Factor	                    Description
Attendance Records    	    Measures the percentage of classes attended by a student. Higher attendance generally reflects                               greater engagement and is often associated with improved academic performance.
Study Hours         	      Represents the average number of hours a student spends studying per day or week. Consistent                                 study habits typically contribute to better learning outcomes.
Previous GPA	              Indicates the student's cumulative Grade Point Average from prior academic periods. It serves                                as a strong predictor of future academic achievement.
Tutoring Sessions per Week  Refers to the number of tutoring or academic support sessions attended weekly. Additional                                    academic assistance may improve understanding and performance.

🧠 Well-Being Factors

Factor	                    Description
Sleep Hours per Night 	    Represents the average number of hours a student sleeps each night. Adequate sleep supports                                  concentration, memory retention, and overall academic performance.
Stress Level        	      Measures the level of academic or personal stress experienced by a student, typically                                        categorized as low, medium, or high. Excessive stress may negatively impact learning and                                     performance.
Screen Time per Day	        Indicates the average number of hours spent using electronic devices such as smartphones,                                    computers, or tablets. Excessive screen time may affect study habits and sleep quality.
Exam Anxiety        	      Reflects the degree of nervousness or anxiety a student experiences before or during                                         examinations. High anxiety levels can affect academic outcomes and test performance.

👥 Demographic and Lifestyle Factors


Factor	                    Description
Gender	                    Represents the student's gender. This variable is included for analytical purposes to examine                                potential differences in academic outcomes across groups.
Age	                        Indicates the student's age in years. Age may influence learning styles, maturity, and                                       academic behavior.
Part-Time Job       	      Specifies whether the student is employed while studying. Balancing work and academic                                        responsibilities can impact study time and performance.
Study Method	              Describes the student's preferred learning approach, such as self-study, group study, online                                 learning, or tutoring-based learning. Different methods may affect academic success.
Diet Quality	              Measures the overall nutritional quality of the student's diet. Healthy eating habits can                                    support cognitive function, energy levels, and academic performance.
Internet Quality	          Represents the reliability and speed of internet access available to the student. Good                                       internet connectivity is particularly important for online learning and research activities.
Extracurricular Activities  Indicates participation in sports, clubs, volunteer work, or other non-academic activities.                                  Such activities may contribute to personal development, time management, and social skills.
Family Income	              Represents the socioeconomic status of the student's household. Financial resources may                                      influence access to educational materials, learning opportunities, and support systems.
🛠️ Tools and Technologies

Programming Language

Python

Libraries

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib/Pickle

Web Development

HTML

CSS

Flask

Development Environment

VS Code

Jupyter Notebook

Version Control

Git

GitHub

🔬 Methodology

1. Data Collection
Load student academic behavior dataset.
2. Data Preprocessing
Handle missing values.
Remove duplicates.
Feature selection and transformation.
Data normalization/encoding.
3. Exploratory Data Analysis (EDA)
Analyze student behavior trends.
Visualize feature relationships.
Identify significant predictors.
4. Model Training

The machine/Deep learning model is trained using:

🤖 Machine Learning Models Used

🌲 Random Forest

Description:
Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. Each tree is trained on a random subset of the data, and the final prediction is determined by majority voting (classification) or averaging (regression).

Advantages:

High accuracy
Handles large datasets effectively
Reduces overfitting compared to a single decision tree
Provides feature importance analysis

Use in Project:
Used to predict student GPA, grade, and risk level based on academic, well-being, and demographic factors.

📈 Logistic Regression

Description:
Logistic Regression is a supervised learning algorithm used for classification problems. It estimates the probability that an observation belongs to a particular class.

Advantages:

Simple and interpretable
Fast training and prediction
Works well for binary classification problems

Use in Project:
Used for predicting student risk categories such as High Risk or Low Risk.

🔥 Gradient Boosting

Description:
Gradient Boosting is an ensemble technique that builds models sequentially, where each new model corrects the errors of the previous one. It combines multiple weak learners to create a strong predictive model.

Advantages:

High predictive performance
Handles complex relationships in data
Effective for both classification and regression tasks

Use in Project:
Used to improve prediction accuracy for GPA and grade prediction.

⚡ XGBoost (Extreme Gradient Boosting)

Description:
XGBoost is an optimized implementation of Gradient Boosting that provides faster training, better performance, and built-in regularization to prevent overfitting.

Advantages:

High accuracy
Fast computation
Handles missing values efficiently
Widely used in machine learning competitions

Use in Project:
Applied to achieve accurate student performance and risk predictions.

💡 LightGBM (Light Gradient Boosting Machine)

Description:
LightGBM is a gradient boosting framework developed by Microsoft. It uses a leaf-wise tree growth strategy that improves efficiency and speed.

Advantages:

Faster training on large datasets
Lower memory consumption
High prediction accuracy

Use in Project:
Used for efficient prediction of academic performance when working with larger datasets.

🐱 CatBoost

Description:
CatBoost is a gradient boosting algorithm developed by Yandex that handles categorical variables effectively without extensive preprocessing.

Advantages:

Excellent handling of categorical data
Minimal feature engineering required
Reduces overfitting

Use in Project:
Suitable for datasets containing categorical features such as gender, study method, and internet quality.

🏗️ Stacking Ensemble

Description:
Stacking Ensemble combines multiple machine learning models and uses a meta-model to generate the final prediction. It leverages the strengths of different algorithms to improve overall performance.

Advantages:

Higher predictive accuracy
Reduces weaknesses of individual models
Produces more robust predictions

Use in Project:
Used as the final model to combine predictions from Random Forest, XGBoost, LightGBM, and CatBoost for enhanced GPA, grade, and risk prediction.

📊 TabNet

Description:
TabNet is a deep learning architecture specifically designed for tabular data. It uses attention mechanisms to select the most relevant features during training.

Advantages:

Designed specifically for structured/tabular datasets
Provides feature interpretability
Learns complex feature relationships

Use in Project:
Explored as an advanced deep learning approach for student academic performance prediction.

The deep learning model is trained using:

🧠 Deep Learning Models Evaluated

🔗 Residual Neural Network (ResNet-style)

Description:
A deep neural network that uses skip connections (residual connections) to pass information directly between layers. This helps prevent the vanishing gradient problem and allows deeper networks to learn effectively.

Use Case:
Suitable for complex tabular datasets where deeper architectures may capture hidden relationships among student factors.

🎯 Attention MLP (Self-Gate)

Description:
A Multi-Layer Perceptron (MLP) enhanced with an attention mechanism that automatically focuses on the most important features during prediction.

Use Case:
Useful for identifying which student attributes (attendance, GPA, stress level, etc.) contribute most to academic performance.

🌐 Wide & Deep Network

Description:
Combines a linear model (Wide component) with a deep neural network (Deep component). The wide part memorizes patterns while the deep part generalizes to unseen combinations.

Use Case:
Effective for capturing both simple and complex relationships in educational datasets.

⚡ Swish-SELU Deep MLP

Description:
A deep feed-forward neural network that uses Swish and SELU activation functions to improve learning efficiency and model stability.

Use Case:
Suitable for predicting GPA and grades from multiple student-related features.

🕸️ DenseNet MLP

Description:
Inspired by DenseNet architecture, where each layer receives information from all previous layers. This improves feature reuse and information flow.

Use Case:
Helps capture interactions among academic, demographic, and well-being factors.

🤖 Transformer-Based Models
🎭 FT-Transformer

Description:
A transformer architecture specifically designed for tabular data. It uses attention mechanisms to learn relationships between different features.

Advantages:

State-of-the-art performance on tabular datasets
Learns complex feature interactions
High prediction accuracy

Use Case:
Student performance prediction using structured educational data.

🤖 TabFormer (BERT-Style)

Description:
A transformer model inspired by BERT that processes tabular data similarly to how BERT processes text. It learns contextual relationships among features.

Advantages:

Captures complex dependencies
Strong performance on structured data
Learns feature importance automatically

Use Case:
Predicting GPA, grades, and student risk levels.

✨ SAINT (Self-Attention and Intersample Attention Transformer)

Description:
An advanced transformer architecture designed specifically for tabular datasets. It applies attention both within features and across samples.

Advantages:

Excellent performance on classification tasks
Robust feature learning
Effective on mixed categorical and numerical data

Use Case:
Student academic risk classification.

🚪 Gated MLP (gMLP)

Description:
A neural network architecture that replaces traditional attention mechanisms with gating operations, reducing computational complexity.

Advantages:

Faster training
Lower computational cost
Competitive performance

Use Case:
Efficient prediction of academic outcomes.

🔄 Sequence and Convolution Models
🔁 BiLSTM + GRU

Description:
Combines Bidirectional Long Short-Term Memory (BiLSTM) and Gated Recurrent Unit (GRU) networks to learn sequential patterns.

Advantages:

Captures long-term dependencies
Effective for time-series and sequential data

Use Case:
Useful if student performance data is collected over multiple semesters.

📡 1D Convolutional Neural Network (1D-CNN)

Description:
A convolutional neural network that applies filters along one-dimensional data to identify local patterns and feature combinations.

Advantages:

Fast training
Effective feature extraction

Use Case:
Learning patterns from student attributes and behavioral indicators.

🎨 Generative and Specialized Models
🔐 Autoencoder + Classifier

Description:
An autoencoder first compresses the data into meaningful representations, then a classifier uses these representations for prediction.

Advantages:

Reduces noise
Improves feature extraction
Useful for dimensionality reduction

Use Case:
Student grade and risk prediction.

🌌 Variational Autoencoder (VAE) Classifier

Description:
A probabilistic version of an autoencoder that learns latent representations and performs classification.

Advantages:

Handles uncertainty well
Learns robust feature representations

Use Case:
Academic performance classification with complex datasets.

🌳 NODE (Neural Oblivious Decision Ensembles)

Description:
A neural network architecture inspired by decision trees that is specifically optimized for tabular data.

Advantages:

Strong performance on structured datasets
Combines tree-based and deep learning benefits

Use Case:
Student GPA and risk prediction.

💊 Capsule Network (CapsuleNet)

Description:
A neural network architecture that groups neurons into capsules to preserve hierarchical relationships among features.

Advantages:

Captures complex relationships
Better feature representation

Use Case:
Advanced educational data modeling and classification.

5. Model Evaluation

Evaluation metrics include:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
6. Deployment
Flask web application for user interaction.
Prediction results displayed through a web interface.

📈 Key Insights

The analysis revealed several factors strongly associated with academic performance:

Higher attendance leads to better academic outcomes.
Increased study hours positively impact performance.
Assignment completion significantly influences final grades.
Student engagement and participation contribute to academic success.
Historical academic records are strong predictors of future performance.
📊 Dashboard / Model Output

<img width="960" height="365" alt="image" src="https://github.com/user-attachments/assets/f493ab5f-c7a7-474a-b03c-c0821d9499b6" />

<img width="958" height="415" alt="image" src="https://github.com/user-attachments/assets/6fbf98d7-8019-470c-8e93-d48d2468a329" />



The system provides:

Input

Users enter student-related information such as:

📥 Input Features

🎓 Academic Factors

Attendance Records
Study Hours
Previous GPA
Tutoring Sessions per Week

🧠 Well-Being Factors

Sleep Hours per Night
Stress Level
Screen Time per Day
Exam Anxiety

👥 Demographic & Lifestyle Factors

Gender
Age
Part-Time Job
Study Method
Diet Quality
Internet Quality
Extracurricular Activities
Family Income

Output

The model predicts:

🎯 Target Variables

The machine learning model is designed to predict a student's academic outcome and risk level based on academic, well-being, and demographic factors.

Target Variable	Description
Predicted GPA	Estimates the student's expected Grade Point Average (GPA) based on the input features. This provides a                      numerical measure of academic performance.
Predicted Grade	Predicts the student's likely academic grade category (e.g., A, B, C, D, F) based on their behavioral                        and academic characteristics.
Risk Level	Classifies students into risk categories to identify those who may require academic support or                               intervention.

Example Output:

Prediction Result:
<img width="959" height="377" alt="image" src="https://github.com/user-attachments/assets/53f2882b-1fc8-41b1-a7a2-d47ed48557ac" />


Confidence Score: 92%
🚀 How to Run This Project
Step 1: Clone Repository
git clone https://github.com/your-username/student-academic-performance-prediction.git
Step 2: Navigate to Project Directory
cd student-academic-performance-prediction
Step 3: Install Dependencies
pip install -r requirements.txt
Step 4: Train Model
python train.py
Step 5: Run Application
python app.py
Step 6: Open Browser
http://127.0.0.1:5000
📋 Results and Conclusions

The machine/Deep learning model successfully predicts student academic performance with satisfactory accuracy.

Results
High prediction accuracy achieved.
Important academic success factors identified.
User-friendly prediction interface developed.
Conclusion

The project demonstrates how machine learning can support educational institutions in identifying at-risk students and improving academic outcomes through timely intervention and data-driven decision-making.

🔮 Future Work

Potential enhancements include:

Integration with real-time educational databases.
Deployment on cloud platforms.
Mobile application development.
Deep Learning implementation.
Student recommendation system.
Real-time performance monitoring dashboard.
Explainable AI (XAI) for prediction interpretation.
👨‍💻 Author

Muhammad Asif Riaz
Final Year Project
Department of Data Science

📜 License

This project is developed for academic and educational purposes.

MIT License
