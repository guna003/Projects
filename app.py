# app.py

import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.set_page_config(page_title="Employee Attrition ML App", page_icon="🧠", layout="wide")
st.title("🔍 Employee Attrition Prediction App (Pro Version)")

# Sidebar
st.sidebar.header("Upload and Configure")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])



# Main function
def main():
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        

        # Example DataFrame (replace with your own data)
        # df = pd.read_csv("your_data.csv")

        st.title("⚙️ Data Visualization")

        # Create the plot
        st.subheader("Attrition vs Monthly Income")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', ax=ax)
        #ax.set_title("Monthly Income vs Attrition")
        st.pyplot(fig)

        # Heatmap Correlation
        st.subheader("Coorelation Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        #Distance from home vs Attrition
        st.subheader("Distance from Home vs Attrition")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x='DistanceFromHome', hue='Attrition', ax=ax)
        st.pyplot(fig)

        #Department vs Attrition
        st.subheader("Department vs Attrition")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x='Department', hue='Attrition', ax=ax)
        st.pyplot(fig)
       

        # Display the plot in Streamlit
        #st.pyplot(fig)


        if "Attrition" not in df.columns:
            st.error("❌ 'Attrition' column not found. Please upload correct dataset.")
            return
        else:
            # Preprocessing
            st.title("⚙️ Data Preprocessing")

            # Attrition Encoding
            if df['Attrition'].dtype == 'object':
                df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

            st.write("#### Attrition Distribution (Before Balancing)")
            st.bar_chart(df['Attrition'].value_counts())

            X = df.drop('Attrition', axis=1)
            y = df['Attrition']

            X = pd.get_dummies(X, drop_first=True)

            # Train Test Split
            test_size = st.sidebar.slider('Test Set Size (%)', 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y
            )

            # Standardization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # SMOTE
            balance_data = st.sidebar.checkbox("Apply SMOTE to Balance Data?", value=True)
            if balance_data:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.success("✅ Applied SMOTE Balancing!")
            

            # Model Selection
            st.sidebar.subheader("Choose Model")
            model_option = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"])

            if st.button("🚀 Train Model"):
                model = None

                if model_option == "Logistic Regression":
                    model = LogisticRegression()
                elif model_option == "Random Forest":
                    model = RandomForestClassifier()
                elif model_option == "Support Vector Machine (SVM)":
                    model = SVC(probability=True)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)

                st.metric(label="Accuracy", value=f"{acc*100:.2f}%")
                st.metric(label="ROC-AUC Score", value=f"{roc_auc:.2f}")

                st.subheader("📈 Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                st.pyplot(fig)

                st.subheader("📈 ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax2.plot([0,1],[0,1],'k--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('ROC Curve')
                ax2.legend()
                st.pyplot(fig2)

                if model_option == "Random Forest":
                    st.subheader("🌟 Feature Importance (Random Forest)")
                    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
                    feat_top = feat_importances.sort_values(ascending=False).head(10)
                    fig3, ax3 = plt.subplots()
                    feat_top.plot(kind='barh', color='teal', ax=ax3)
                    st.pyplot(fig3)

                # Download predictions
                download_preds = st.sidebar.checkbox("Enable Predictions Download")
                if download_preds:
                    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )

if __name__ == "__main__":
    main()

# Footer
st.caption("Built with ❤️ using Streamlit and Scikit-learn.")
