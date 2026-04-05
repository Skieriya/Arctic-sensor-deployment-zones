import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(columns={df.columns[1]: "Lat", df.columns[2]: "Lon"})
    df = df.drop(index=0).reset_index(drop=True)

    methane_df = df[df["Response variable"].str.contains("CH4", case=False, na=False)].copy()
    if methane_df.empty:
        methane_df = df.copy()

    for col in ["Lat", "Lon", "Xwarming"]:
        methane_df[col] = pd.to_numeric(methane_df[col], errors="coerce")

    clean_df = methane_df.dropna(subset=["Lat", "Lon", "Xwarming"]).copy()
    return clean_df


def train_and_save_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["Lat", "Lon", "Xwarming"]])

    model = OPTICS(min_samples=2, xi=0.05, metric="euclidean")
    df["Cluster_ID"] = model.fit_predict(X_scaled)

    with open("permafrost_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("permafrost_optics_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model and scaler saved")
    return df, X_scaled


def evaluate_model(df, X_scaled):
    labels = df["Cluster_ID"]
    mask = labels != -1

    if len(set(labels[mask])) < 2:
        print("Not enough clusters detected for evaluation")
        return None, None

    sil_score = silhouette_score(X_scaled[mask], labels[mask])
    db_score = davies_bouldin_score(X_scaled[mask], labels[mask])

    print("Silhouette Score:", sil_score)
    print("Davies-Bouldin Score:", db_score)

    return sil_score, db_score


def plot_silhouette_and_db(sil, db):
    plt.figure(figsize=(6, 4))
    names = ["Silhouette", "Davies-Bouldin"]
    values = [sil, db]
    sns.barplot(x=names, y=values)
    plt.title("Silhouette and Davies-Bouldin Scores")
    plt.ylabel("Score")
    plt.show()


def plot_methane_map(df):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    scatter = ax.scatter(
        df["Lon"],
        df["Lat"],
        c=df["Xwarming"],
        cmap="plasma",
        s=70
    )

    plt.title("Methane Emission Zones (Xwarming)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Xwarming")

    plt.show()


def run_pipeline(file_path):
    print("Loading data")
    df = load_and_clean_data(file_path)

    print("Training model")
    df, X_scaled = train_and_save_model(df)

    print("Evaluating model")
    sil, db = evaluate_model(df, X_scaled)

    print("Plotting Silhouette and Davies-Bouldin")
    plot_silhouette_and_db(sil, db)

    print("Plotting methane map")
    plot_methane_map(df)


if __name__ == "__main__":
    run_pipeline("Data_Permafrost_Carbon.xlsx")