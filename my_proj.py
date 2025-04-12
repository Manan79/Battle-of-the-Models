import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder
# from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve
# from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , r2_score, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import graph_objects as go
import time


# project starts 
st.set_page_config(page_title="ML Model Comparator", layout="wide")



# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# loading of data
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        # make timer to show message
        # with st.spinner("Loading data..."):
        #     time.sleep(2)
        st.sidebar.success("Data loaded successfully!")
        
        # Show basic info
        if st.sidebar.checkbox("Show dataset info"):
            st.subheader("Dataset Information")
            # st.write("")
            st.write(f"Shape of the dataset is  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            st.write("First 5 rows:")
            st.dataframe(df.head())

            st.write("Statistical summary(Description of the dataset):")
            st.write(df.describe())
            
            st.write("Column types:")
            st.write(df.dtypes)

            st.write("Missing values:")
            st.write(df.isna().sum())

        if st.sidebar.checkbox("Show basic Visualisation of the dataset"):

            # visualization of missing values
            for i in df.columns:
                if df[i].isnull().sum() > 0:
                    plt.figure(figsize=(10, 5))
                    sns.heatmap(df[i].isnull().values, cmap='viridis', cbar=False)
                    plt.title(f"Missing values in {i}")
                    st.pyplot(plt)
                    plt.close()
            # visualization of df
            st.subheader("Data Visualization (Top 100 rows)")
            st.write("First 100 rows of the dataset:")
            for i in df.columns:
                new_df = df.copy()
                new_df.head(100)
                if df[i].dtype == 'object':
                    plt.figure(figsize=(10, 5))
                    sns.countplot(x=new_df[i])
                    plt.title(f"Count plot of {i}")
                    plt.xticks(rotation=70)
                    st.pyplot(plt)
                    plt.close()
                else:
                    plt.figure(figsize=(10, 5))
                    sns.histplot(new_df[i], kde=True)
                    plt.title(f"Distribution of {i}")
                    st.pyplot(plt)
                    plt.close()
        
        # select target variable which is not object
        target_col = st.sidebar.selectbox("Select target variable", df.select_dtypes(include=[np.number]).columns.tolist()) 
        
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.sidebar.multiselect(
            "Select features to include",
            available_features,
            default=available_features
        )
        
        # modeliing starts model parameter hai
        if df.shape[0] < 30000:
            df = df
        else:
            df = df.head(30000)
        st.sidebar.subheader("Preprocessing Options")
        test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)
        random_state = st.sidebar.number_input("Random state", 0, 100, 42)
        scale_numeric = st.sidebar.checkbox("Scale numeric features", True)
        encode_categorical = st.sidebar.selectbox("Select type of Encoding on data", ["Label Encoding"] , index=0)
        
        if encode_categorical == "Label Encoding":
            encode_categorical = LabelEncoder()
            for i in df.columns:
                if df[i].dtype == 'object':
                    df[i] = encode_categorical.fit_transform(df[i])
            
         # Model selection
        st.sidebar.subheader("Model Selection")

        models_to_train = st.sidebar.selectbox(
            "Select models to train",
            ["Logistic Regression", "Decision Tree", "Random Forest",
             "SVM", "Gradient Boosting"], index=2
            
        )
        if models_to_train != "All of the above":
           models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                    "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                    "Random Forest": RandomForestClassifier(random_state=random_state),
                    "SVM": SVC(probability=True, random_state=random_state),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
                }
           model = models[models_to_train]
        else:
            models_to_train = ["Logistic Regression", "Decision Tree", "Random Forest",
             "SVM", "Gradient Boosting"]
            
        
        # train and test split shuru
        X = df[selected_features]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
        # train and test split khatam

        # scaling of data
        if scale_numeric:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = None
        
        train = st.sidebar.button("Train Models")
        if train:
            model.fit(X_train, y_train)
            st.sidebar.success(f"{models_to_train} trained successfully!")

            # Evaluation Metrics
            st.title(f"Model Evaluation for {models_to_train}")
            

            def evaluate_model(model, X_test, y_test, average='binary'):
                y_pred = model.predict(X_test)
                
                # Handle probability prediction safely
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_prob)
                except:
                    roc_auc = None  # Model doesn't support predict_proba
            
                # Handle multiclass case
                if len(set(y_test)) > 2:
                    average = 'macro'
            
                return {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average=average, zero_division=0),
                    "ROC AUC": roc_auc,
                    "ypred" : y_pred,
                    "y_proba" : y_prob

                }
            metrics = evaluate_model(model, X_test, y_test)
            new_metrics = {
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "ROC AUC": metrics["ROC AUC"]
            }
            st.subheader("Model Evaluation Metrics")
            st.table(pd.DataFrame(new_metrics, index=[0]).T)  
            st.subheader("Classification Report")
            st.table(classification_report(y_test, metrics['ypred'], output_dict=True))

            st.subheader("Actual vs Predicted")
            act_vs_pred = pd.DataFrame({"Actual": y_test, "Predicted": metrics['ypred']})
            st.table(act_vs_pred.head(12))

            try:
                cm = confusion_matrix(y_test, metrics['ypred'])
                cm_fig = px.imshow(cm,
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=["Pred 0", "Pred 1"],
                                   y=["Actual 0", "Actual 1"],
                                   text_auto=True,
                                   title="Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
                roc_auc = auc(fpr, tpr)

                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
                roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(roc_fig, use_container_width=True)

                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, metrics["y_proba"])
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines"))
                pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(pr_fig, use_container_width=True)
                
                
                st.title("Model's Information")

                col1 , col2  = st.columns(2)
                with col1:
                    st.subheader("Model Parameters")
                    st.write(model.get_params())
                with col2:
                    st.subheader("Model Hyperparameters")
                    st.write(model.get_params())
                
                col3 , col4 , col5 = st.columns(3)

                with col3:
                    st.write(f"Test size: {test_size}%")
                with col4:
                    st.write(f"Random state: {random_state}")
                with col5 :
                    st.write(f"Scaling: {scale_numeric}")
                
                    
                col6 , col7= st.columns(2)
                with col6:
                    st.write(f"Encoding type: {encode_categorical}")
                with col7:
                    st.write(f"Target variable: {target_col}")
                
                st.subheader("Feature Importance")
                if hasattr(model, "feature_importances_"):
                    feature_importances = model.feature_importances_
                else:
                    feature_importances = model.coef_[0] if hasattr(model, "coef_") else None
                st.write(f"Selected features: {selected_features}")
                
                st.table(pd.DataFrame(feature_importances, index=selected_features, columns=["Importance"]).sort_values(by="Importance", ascending=False))
                fig = plt.figure(figsize=(10, 5))
                sns.barplot(x=selected_features, y=feature_importances)
                plt.title("Feature Importance")
                plt.xticks(rotation=90)
                st.pyplot(fig)
                plt.close(fig)

     


            except Exception as e:
                st.error(f"Error plotting confusion matrix or ROC curve: {e}")
        else:
            
        # Title and description
            st.title("Machine Learning Model Comparison Dashboard")
            st.markdown("""
            This app allows you to:
            - Upload your dataset
            - Preprocess the data
            - Train multiple classification models
            - Comparest.subheader("Model Accuracy")
                    st.write(f"{metrics['Accuracy']:.2f}")
                with col4: their performance metrics
            - Tune hyperparameters
            - Visualize
             results
            """)

    compare = st.sidebar.checkbox("Comapre all Models", value=False)
    if compare:
        # make a loader
        with st.spinner("Comparing models..."):
            time.sleep(4)
            st.success("Models compared successfully!")
            
        import streamlit as st
        import pandas as pd
        import altair as alt
        import os
                
        # Title
        st.subheader("🤖 Model Evaluation & Comparison Dashboard")
        
        # Load data
        data = pd.read_csv("model_metrics.csv")
        
        # --- 1. METRICS TABLE ---
        st.subheader("📊 Overall Performance Metrics")
        st.dataframe(data)
        
        # --- 2. BAR CHARTS FOR EACH METRIC ---
        st.subheader("📈 Metric Comparisons")
        
        metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
        for metric in metric_columns:
            st.markdown(f"### 🔹 {metric}")
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X("Model", sort="-y"),
                y=alt.Y(metric, scale=alt.Scale(domain=[0, 1])),
                color="Model"
            ).properties(width=600, height=300)
            st.altair_chart(chart, use_container_width=True)
        
        # --- 3. CONFUSION MATRICES & CLASSIFICATION REPORTS ---
        st.subheader("🧩 Detailed Per-Model Insights")
        
        model_selection = st.selectbox("Select a model to view details:", data["Model"])
        
        # Show Confusion Matrix
        st.markdown("#### 📌 Confusion Matrix")
        cm_path = f"confusion_images/{model_selection}.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"{model_selection} - Confusion Matrix" )
        else:
            st.warning("Confusion matrix image not found.")




    
        

    



        

        


            




        
            




