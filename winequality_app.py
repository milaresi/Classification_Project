import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine_df = pd.read_csv("winequality-red.csv")

st.title("Wine Quality Classification App")
st.subheader("Dataset Preview")
st.write(wine_df.head())

st.subheader("Summary Statistics")
st.write(wine_df.describe())
dup_count = wine_df.duplicated().sum()
st.write(f"**Duplicates found:** {dup_count}")

# remove duplicates
wine_df = wine_df.drop_duplicates()
st.write(f"**Duplicates removed:** {dup_count}")
# misisng values
st.subheader("Missing Value Report")
missing_vals = wine_df.isnull().sum()
st.write(missing_vals)
total_missing = missing_vals.sum()
st.write(f"Total missing values in dataset:{total_missing}")
# clean missing values
wine_df = wine_df.dropna()
st.write("Missing values removed.")
# EDA
st.header("EDA")
# correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(wine_df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# boxplot
st.subheader("Boxplots of pH and Alcohol")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=wine_df["pH"], ax=axs[0])
axs[0].set_title("pH")
sns.boxplot(y=wine_df["alcohol"], ax=axs[1])
axs[1].set_title("Alcohol")

st.pyplot(fig)
# scatterplot
st.subheader("Scatterplots")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(wine_df["alcohol"], wine_df["quality"], alpha=0.5, color="blue")
axs[0].set_xlabel("Alcohol")
axs[0].set_ylabel("Quality")
axs[0].set_title("Alcohol vs Quality")

axs[1].scatter(wine_df["pH"], wine_df["quality"], alpha=0.5, color="pink")
axs[1].set_xlabel("pH")
axs[1].set_ylabel("Quality")
axs[1].set_title("pH vs Quality")
st.pyplot(fig)
# Model training
st.header("Model Training")
X = wine_df.drop("quality", axis=1)
y = wine_df["quality"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
models = {
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVC": SVC(kernel="rbf", C=2),
    "Logistic Regression": LogisticRegression(max_iter=200),
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, pred)
# Accuracy chart
st.subheader("Model Accuracy Bar Chart")
fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values())
ax.set_ylabel("Accuracy")
st.pyplot(fig)

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
st.title(f"Best Model: {best_model_name}")

# prediction
# st.header("Predict Wine Quality")
# st.sidebar.header("Input Wine Features")
# user_data = {}
# for col in X.columns:
#     user_data[col] = st.sidebar.number_input(
#         col,
#         float(wine_df[col].min()),
#         float(wine_df[col].max()),
#         float(wine_df[col].mean()),
#     )

# user_df = pd.DataFrame([user_data])
# if st.button("Predict"):
#     scaled_input = scaler.transform(user_df)
#     pred = best_model.predict(scaled_input)
#     quality_score = int(pred[0])
#     if quality_score >= 6:
#         quality_label = "Good Quality"
#     else:
#         quality_label = "Bad Quality"

#     st.success(f"Predicted Wine Quality Score: {quality_label}")
import joblib

joblib.dump(best_model, "wine_classifier_app.joblib")
model = joblib.load("wine_Classifier_app.joblib")
# prediction
st.header("Predict Wine Quality")
st.sidebar.header("Input Wine Features")
user_data = {}
for col in X.columns:
    user_data[col] = st.sidebar.number_input(
        col,
        float(wine_df[col].min()),
        float(wine_df[col].max()),
        float(wine_df[col].mean()),
    )

user_df = pd.DataFrame([user_data])
if st.button("Predict"):
    scaled_input = scaler.transform(user_df)
    pred = best_model.predict(scaled_input)
    quality_score = int(pred[0])
    if quality_score >= 6:
        quality_label = "Good Quality"
    else:
        quality_label = "Bad Quality"

    st.success(f"Predicted Wine Quality Score: {quality_label}")
