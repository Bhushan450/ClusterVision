import io
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# PURE PYTHON/NUMPY K-MEDOIDS (PAM) IMPLEMENTATION
# (Guarantees Python 3.14 compatibility without scikit-learn-extra)
# -------------------------------------------------------------------
class PureKMedoids:
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        dist_matrix = pairwise_distances(X, metric='euclidean')
        
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for _ in range(self.max_iter):
            labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
            new_medoid_indices = np.copy(medoid_indices)
            
            for k in range(self.n_clusters):
                members = np.where(labels == k)[0]
                if len(members) > 0:
                    sub_dist = dist_matrix[np.ix_(members, members)]
                    best_member = members[np.argmin(np.sum(sub_dist, axis=1))]
                    new_medoid_indices[k] = best_member
            
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
            medoid_indices = new_medoid_indices
            
        return np.argmin(dist_matrix[:, medoid_indices], axis=1)


# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Wholesale Customer Analytics", page_icon="🛒", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"

st.title("🛒 Wholesale Customer Segmentation & Purchasing Analytics")
st.caption("Unsupervised Machine Learning with K-Means, K-Medoids (PAM), and DBSCAN Clustering")

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_URL)

def normalize_columns(df):
    df = df.copy()
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"channel"}:
            rename[col] = "Channel"
        elif key in {"region"}:
            rename[col] = "Region"
        elif key in {"fresh"}:
            rename[col] = "Fresh"
        elif key in {"milk"}:
            rename[col] = "Milk"
        elif key in {"grocery"}:
            rename[col] = "Grocery"
        elif key in {"frozen"}:
            rename[col] = "Frozen"
        elif key in {"detergents_paper", "detergent_paper", "detergents paper"}:
            rename[col] = "Detergents_Paper"
        elif key in {"delicassen", "delicatessen"}:
            rename[col] = "Delicassen"
    return df.rename(columns=rename)

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data & Dataset Settings")
    uploaded = st.file_uploader("Upload Wholesale CSV (optional)", type=["csv"])
    
    st.header("2. Model & Algorithm Selection")
    algo_choice = st.selectbox("Select Clustering Algorithm", ["K-Means", "K-Medoids (PAM)", "DBSCAN"])
    
    if algo_choice in ["K-Means", "K-Medoids (PAM)"]:
        k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
    else:
        eps = st.slider("Epsilon (eps)", min_value=0.2, max_value=2.0, value=0.8, step=0.1)
        min_samples = st.slider("Min Samples", min_value=2, max_value=15, value=5)

# Load Dataset
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_label = "Uploaded CSV File"
    else:
        raw = load_default_data()
        source_label = "UCI Wholesale Customers Dataset"
except Exception as e:
    st.error("Could not load the dataset. Please upload a valid CSV file.")
    st.caption(f"Error detail: {e}")
    st.stop()

df = normalize_columns(raw)
if df.empty:
    st.error("The loaded dataset contains no rows.")
    st.stop()

# Identify features for clustering
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_features = [c for c in numeric_cols if c not in {"Channel", "Region"}]
preferred_features = [c for c in ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"] if c in all_features]

if not preferred_features:
    preferred_features = all_features

if len(all_features) < 2:
    st.error("Please upload a dataset with at least two continuous annual spending attributes.")
    st.stop()

with st.sidebar:
    features = st.multiselect("Features used for clustering", options=all_features, default=preferred_features)

if len(features) < 2:
    st.warning("Select at least two numeric spending features to execute clustering.")
    st.stop()

# Preprocessing & Model Execution
model_df = df[features].apply(pd.to_numeric, errors="coerce")
valid_mask = model_df.notna().all(axis=1)
clean = df.loc[valid_mask].copy()
model_df = model_df.loc[valid_mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(model_df)

# Model Training
if algo_choice == "K-Means":
    if len(model_df) <= k:
        st.error(f"Need more valid customer rows than K={k}.")
        st.stop()
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")

elif algo_choice == "K-Medoids (PAM)":
    if len(model_df) <= k:
        st.error(f"Need more valid customer rows than K={k}.")
        st.stop()
    model = PureKMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")

else: # DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    non_noise_mask = labels != -1
    if len(set(labels)) > 1 and non_noise_mask.sum() > 0:
        sil = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
    else:
        sil = float("nan")

clean["Cluster"] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]

# Metric Display Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wholesale Clients Analyzed", f"{len(clean):,}")

if algo_choice in ["K-Means", "K-Medoids (PAM)"]:
    c2.metric("Clusters Formed", str(k))
else:
    c2.metric("Noise Points Detected", str(list(labels).count(-1)))

c3.metric("Silhouette Score", "N/A" if np.isnan(sil) else f"{sil:.3f}")
c4.metric("Features Selected", str(len(features)))

st.caption(f"Data source: {source_label}. Non-numeric rows and excluded categorical metadata are filtered during preprocessing.")

# -------------------------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cluster Spending Profiles", "Client Explorer", "Methods & Manual Mapping"])

with tab1:
    left, right = st.columns(2)
    with left:
        counts = clean["Cluster"].value_counts().reset_index()
        counts.columns = ["Cluster", "Clients"]
        fig_bar = px.bar(counts, x="Cluster", y="Clients", color="Cluster", 
                         title="Wholesale Clients Distribution per Cluster", text="Clients")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with right:
        if len(features) == 2:
            fig_scatter = px.scatter(clean, x=features[0], y=features[1], color="Cluster",
                                     hover_data=[c for c in clean.columns if c in ["Channel", "Region"] + features],
                                     title="2D Spending Scatter Analysis")
        else:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            plot_df = clean.copy()
            plot_df["PCA Component 1"] = coords[:, 0]
            plot_df["PCA Component 2"] = coords[:, 1]
            fig_scatter = px.scatter(plot_df, x="PCA Component 1", y="PCA Component 2", color="Cluster",
                                     hover_data=features, title="Client Segments (PCA 2D Projection)")
            fig_scatter.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
            
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.info("Clusters represent natural groupings discovered in client spending patterns across product categories.")

with tab2:
    st.subheader("Average Annual Spending per Cluster ($)")
    profile = clean.groupby("Cluster")[features].mean().round(2)
    profile.insert(0, "Total Clients", clean.groupby("Cluster").size())
    st.dataframe(profile, use_container_width=True)
    
    if "Fresh" in features and "Grocery" in features:
        st.plotly_chart(px.scatter(clean, x="Fresh", y="Grocery", color="Cluster",
                                   title="Fresh Goods vs. Grocery Spending Comparison"), use_container_width=True)
        
    st.subheader("Marketing & Purchasing Interpretations")
    for cluster_id, row in profile.iterrows():
        if cluster_id == "Noise":
            st.write("**Noise Points:** Client profiles with irregular spending behavior or severe outliers.")
            continue
            
        means = row[features]
        fresh_spend = means.get("Fresh", 0)
        grocery_spend = means.get("Grocery", 0)
        
        if fresh_spend > clean["Fresh"].median() and grocery_spend > clean["Grocery"].median():
            strategy = "High-Volume Distributors/Restaurants: Offer volume-based wholesale discount tiers and priority delivery routes."
        elif fresh_spend > clean["Fresh"].median():
            strategy = "Fresh Produce Specialists/HORECA: Target with seasonal fresh food promotions and supply-chain logistics offers."
        elif grocery_spend > clean["Grocery"].median():
            strategy = "Retail Supermarkets/Grocers: Recommend bulk non-perishable packages and inventory management solutions."
        else:
            strategy = "Small Businesses/Cafes: Offer flexible ordering terms, lower minimum order quantities (MOQs), and loyalty credits."
            
        st.write(f"**{cluster_id}:** {strategy}")

with tab3:
    st.subheader("Explore Clustered Clients")
    unique_clusters = sorted(clean["Cluster"].unique())
    cluster_choice = st.selectbox("Select Cluster to Filter", unique_clusters)
    
    filtered_df = clean[clean["Cluster"] == cluster_choice]
    st.write(f"Total clients in **{cluster_choice}**: **{len(filtered_df)}**")
    st.dataframe(filtered_df, use_container_width=True)
    
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Segment as CSV", csv_bytes, 
                       file_name=f"wholesale_{cluster_choice.lower().replace(' ', '_')}_clients.csv", mime="text/csv")

with tab4:
    st.markdown("""
    ### ML Workflow & Academic Alignment
    1. **Data Preprocessing & Scaling**: Normalizes feature columns and scales all features with `StandardScaler` so high-value attributes (like *Fresh* or *Grocery*) do not skew Euclidean distances[cite: 42, 58].
    2. **Dimensionality Reduction**: Uses Principal Component Analysis (PCA) to compress high-dimensional purchasing data into 2 components for clear 2D plotting[cite: 58].
    3. **Unsupervised Modeling**:
       - **K-Means**: Partitions clients into centroids by minimizing intra-cluster distance variance[cite: 42].
       - **K-Medoids (PAM)**: Selects actual dataset data points as cluster centers, providing greater resistance to extreme spending outliers[cite: 42, 43].
       - **DBSCAN**: Groups clients based on spatial density ($\epsilon$ and $min\_samples$), automatically identifying outliers as Noise (-1)[cite: 48].
    4. **Evaluation**: Calculates Silhouette Scores to validate cluster separation quality[cite: 42].
    """)