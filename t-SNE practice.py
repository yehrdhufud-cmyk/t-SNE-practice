
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load dataset
digits = load_digits()
X = digits.data
y = digits.target

print("Original shape:", X.shape)

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reduce dimensions with PCA before t-SNE
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Shape after PCA:", X_pca.shape)

# 4. Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_pca)

print("Shape after t-SNE:", X_tsne.shape)

# 5. Evaluate the embedding
score = silhouette_score(X_tsne, y)
print("Silhouette score:", round(score, 3))

# 6. Visualize the results
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="tab10",
    s=15,
    alpha=0.8
)

plt.colorbar(scatter, ticks=range(10), label="Digit")
plt.title("t-SNE Visualization of Handwritten Digits")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.show()