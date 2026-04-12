"""Экспорт графиков из snp_explorer.ipynb как PNG файлов."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

# ---- Данные (те же параметры, что в ноутбуке) ----
def generate_snp_matrix(
    n_per_pop=500, n_snps=10000, n_divergent_snps=3000,
    freq_shift=0.3, random_seed=42,
):
    rng = np.random.default_rng(random_seed)
    n_samples = 2 * n_per_pop
    base_freqs = rng.uniform(0.05, 0.5, size=n_snps)
    freqs_pop1 = base_freqs.copy()
    freqs_pop2 = base_freqs.copy()
    divergent_indices = rng.choice(n_snps, size=n_divergent_snps, replace=False)
    for idx in divergent_indices:
        shift = freq_shift * rng.standard_normal()
        freqs_pop2[idx] = np.clip(freqs_pop2[idx] + shift, 0.05, 0.95)

    def sample_genotypes(freqs, n, rng):
        p = freqs
        p_aa = (1 - p) ** 2
        p_Aa = 2 * p * (1 - p)
        cum = np.column_stack([p_aa, p_aa + p_Aa, np.ones_like(p)])
        rand_vals = rng.random((n, len(freqs)))
        genotypes = np.zeros((n, len(freqs)), dtype=np.int8)
        genotypes += (rand_vals > cum[:, 0]).astype(np.int8)
        genotypes += (rand_vals > cum[:, 1]).astype(np.int8)
        return genotypes

    geno_pop1 = sample_genotypes(freqs_pop1, n_per_pop, rng)
    geno_pop2 = sample_genotypes(freqs_pop2, n_per_pop, rng)
    return np.vstack([geno_pop1, geno_pop2]), np.array([0]*n_per_pop + [1]*n_per_pop, dtype=np.int8)


print("Генерация данных...")
X, y = generate_snp_matrix()

# ---- PCA ----
print("PCA...")
pca_full = PCA(random_state=42)
X_pca_all = pca_full.fit_transform(X)
explained_var_ratio = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var_ratio)
n_95 = np.searchsorted(cumulative_var, 0.95) + 1
X_pca_20 = X_pca_all[:, :20]

# ---- Scree plot ----
print("Scree plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
n_show = 30
axes[0].bar(range(1, n_show + 1), explained_var_ratio[:n_show], color="steelblue", edgecolor="white")
axes[0].set_xlabel("Главная компонента", fontsize=12)
axes[0].set_ylabel("Объяснённая дисперсия", fontsize=12)
axes[0].set_title("Scree Plot (первые 30 компонент)", fontsize=13)

axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, color="coral", linewidth=2)
axes[1].axhline(0.95, color="red", linestyle="--", linewidth=1.5, label="95% порог")
axes[1].axvline(n_95, color="green", linestyle="--", linewidth=1.5, label=f"{n_95} компонент")
axes[1].set_xlabel("Число компонент", fontsize=12)
axes[1].set_ylabel("Кумулятивная объяснённая дисперсия", fontsize=12)
axes[1].set_title("Кумулятивная дисперсия", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, min(200, len(cumulative_var)))
axes[1].set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("plots/scree_plot.png", bbox_inches="tight")
plt.close()
print("  -> plots/scree_plot.png")

# ---- t-SNE + UMAP ----
print("t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca_20)

print("UMAP...")
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
X_umap = umap_model.fit_transform(X_pca_20)

# ---- Dashboard ----
print("Dashboard...")
pop_labels = np.where(y == 0, "Порода 1", "Порода 2")
palette = {"Порода 1": "#2196F3", "Порода 2": "#FF5722"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for pop, color in palette.items():
    mask = pop_labels == pop
    axes[0].scatter(X_pca_20[mask, 0], X_pca_20[mask, 1], c=color, label=pop, alpha=0.6, s=15, edgecolors="none")
axes[0].set_title("PCA (PC1 vs PC2)", fontsize=14, fontweight="bold")
axes[0].set_xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}%)")
axes[0].legend(fontsize=10)

for pop, color in palette.items():
    mask = pop_labels == pop
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=pop, alpha=0.6, s=15, edgecolors="none")
axes[1].set_title("t-SNE", fontsize=14, fontweight="bold")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].legend(fontsize=10)

for pop, color in palette.items():
    mask = pop_labels == pop
    axes[2].scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, label=pop, alpha=0.6, s=15, edgecolors="none")
axes[2].set_title("UMAP", fontsize=14, fontweight="bold")
axes[2].set_xlabel("UMAP 1")
axes[2].set_ylabel("UMAP 2")
axes[2].legend(fontsize=10)

plt.suptitle("Структура популяции: PCA / t-SNE / UMAP", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/dashboard.png", bbox_inches="tight")
plt.close()
print("  -> plots/dashboard.png")

print("\nВсе графики экспортированы.")
