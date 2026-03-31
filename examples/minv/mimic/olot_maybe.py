import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import re, os

def safe_name(s: str, maxlen: int = 120) -> str:
    s = str(s)
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    # keep letters, numbers, dot, underscore, dash
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:maxlen]


def visualize_tabular_and_save(
    df: pd.DataFrame,
    label_col: str = "identity",
    top_k: int = 12,
    use_tsne: bool = False,
    out_dir: str = "viz_out",
    random_state: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    # ---------- split X / y ----------
    y = df[label_col].values
    # numeric (exclude bool/nullable-bool)
    num_cols = [
        c for c in df.columns
        if c != label_col
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and str(df[c].dtype) != "boolean"
    ]
    if len(num_cols) == 0:
        raise ValueError("No numeric (non-boolean) features found.")

    X_num = df[num_cols].values

    # ---------- 1) Class balance ----------
    counts = pd.Series(y).value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class counts"); plt.xlabel("Class"); plt.ylabel("Count")
    plt.tight_layout()
    p = os.path.join(out_dir, "class_counts.png"); plt.savefig(p, dpi=200); plt.close()
    counts.to_csv(os.path.join(out_dir, "class_counts.csv"), header=["count"])
    paths["class_counts_png"] = p
    paths["class_counts_csv"] = os.path.join(out_dir, "class_counts.csv")

    # ---------- 2) Rank features by ANOVA F ----------
    fvals, _ = f_classif(np.nan_to_num(X_num), y)
    feat_rank = pd.DataFrame({"feature": num_cols, "F": fvals}).sort_values("F", ascending=False)
    feat_rank.to_csv(os.path.join(out_dir, "feature_ranking.csv"), index=False)
    paths["feature_ranking_csv"] = os.path.join(out_dir, "feature_ranking.csv")

    top_cols = feat_rank["feature"].head(top_k).tolist()

    # ---------- 3) Per-feature distributions (grid) + save hist data ----------
    classes = np.unique(y)
    n = len(top_cols)
    rows, cols = int(np.ceil(n / 4)), 4
    plt.figure(figsize=(4.5 * cols, 3.0 * rows))
    hist_data_dir = os.path.join(out_dir, "hist_data")
    os.makedirs(hist_data_dir, exist_ok=True)

    for i, c in enumerate(top_cols, 1):
        ax = plt.subplot(rows, cols, i)
        # boolean-safe check for later (if you want to add bool features)
        is_bool = pd.api.types.is_bool_dtype(df[c]) or str(df[c].dtype) == "boolean"

        # decide bins
        if is_bool:
            bins = np.array([-0.5, 0.5, 1.5])
        else:
            vals_all = df[c].dropna().values
            # robust bins (use quantiles to avoid long tails)
            qmin, qmax = np.nanpercentile(vals_all, [1, 99]) if len(vals_all) else (0, 1)
            if qmin == qmax:
                qmin, qmax = (qmin - 0.5, qmax + 0.5)
            bins = np.linspace(qmin, qmax, 31)

        # collect per-class hist counts to save
        out_rows = []
        for cls in classes:
            vals = df.loc[df[label_col] == cls, c].dropna().values
            if is_bool:
                vals = vals.astype(int)
            counts_c, edges = np.histogram(vals, bins=bins)
            ax.step(0.5 * (edges[:-1] + edges[1:]), counts_c, where="mid", label=str(cls), alpha=0.85)

            for bl, br, ct in zip(edges[:-1], edges[1:], counts_c):
                out_rows.append({"class": int(cls), "bin_left": bl, "bin_right": br, "count": int(ct)})

        ax.set_title(c, fontsize=10)
        if i == 1:
            ax.legend(fontsize=8, ncol=2)
        ax.set_xlabel("value"); ax.set_ylabel("count")

        # save hist data for this feature
        base = safe_name(c)  # e.g., "BMI (kg/m2)" -> "BMI_kg_m2"
        hist_csv = os.path.join(hist_data_dir, f"hist_{base}.csv")
        # just in case, ensure parent dir exists (it should, but safe to keep)
        os.makedirs(os.path.dirname(hist_csv), exist_ok=True)
        pd.DataFrame(out_rows).to_csv(hist_csv, index=False)

    plt.tight_layout()
    p = os.path.join(out_dir, "hists_top_features.png"); plt.savefig(p, dpi=200); plt.close()
    paths["hists_top_features_png"] = p
    paths["hists_data_dir"] = hist_data_dir

    # ---------- prep standardized matrix for embeddings ----------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[top_cols].astype(float).fillna(df[top_cols].mean()).values)

    # ---------- 4a) PCA ----------
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(Xs)
    plt.figure(figsize=(6, 5))
    for cls in classes:
        m = (y == cls)
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=str(cls))
    plt.title("PCA (top-k features)"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    p = os.path.join(out_dir, "pca_scatter.png"); plt.savefig(p, dpi=200); plt.close()
    paths["pca_png"] = p
    pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], label_col: y}).to_csv(
        os.path.join(out_dir, "pca_embedding.csv"), index=False
    )
    paths["pca_embedding_csv"] = os.path.join(out_dir, "pca_embedding.csv")

    # ---------- 4b) t-SNE (optional; slower) ----------
    if use_tsne:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=10, random_state=random_state)
        T = tsne.fit_transform(Xs)
        plt.figure(figsize=(6, 5))
        for cls in classes:
            m = (y == cls)
            plt.scatter(T[m, 0], T[m, 1], s=12, alpha=0.7, label=str(cls))
        plt.title("t-SNE (top-k features)")
        plt.legend(markerscale=2, fontsize=8, ncol=2)
        plt.tight_layout()
        p = os.path.join(out_dir, "tsne_scatter.png"); plt.savefig(p, dpi=200); plt.close()
        paths["tsne_png"] = p
        pd.DataFrame({"TSNE1": T[:, 0], "TSNE2": T[:, 1], label_col: y}).to_csv(
            os.path.join(out_dir, "tsne_embedding.csv"), index=False
        )
        paths["tsne_embedding_csv"] = os.path.join(out_dir, "tsne_embedding.csv")

    # ---------- 4c) LDA (if feasible) ----------
    try:
        if len(classes) > 1:
            lda_n = 2 if len(classes) > 2 else 1
            lda = LDA(n_components=lda_n)
            L = lda.fit_transform(Xs, y)
            plt.figure(figsize=(6, 5))
            if lda_n == 2:
                for cls in classes:
                    m = (y == cls)
                    plt.scatter(L[m, 0], L[m, 1], s=12, alpha=0.7, label=str(cls))
                plt.xlabel("LD1"); plt.ylabel("LD2")
            else:
                for cls in classes:
                    m = (y == cls)
                    plt.scatter(L[m, 0], np.zeros_like(L[m, 0]), s=12, alpha=0.7, label=str(cls))
                plt.xlabel("LD1"); plt.yticks([])
            plt.title("LDA (top-k features)")
            plt.legend(markerscale=2, fontsize=8, ncol=2)
            plt.tight_layout()
            p = os.path.join(out_dir, "lda_scatter.png"); plt.savefig(p, dpi=200); plt.close()
            paths["lda_png"] = p
            cols = ["LD1", "LD2"][:L.shape[1]]
            pd.DataFrame(dict(zip(cols, [L[:, i] for i in range(L.shape[1])]) | {label_col: y})).to_csv(
                os.path.join(out_dir, "lda_embedding.csv"), index=False
            )
            paths["lda_embedding_csv"] = os.path.join(out_dir, "lda_embedding.csv")
    except Exception:
        pass  # LDA can fail if collinear or not enough samples per class

    # ---------- 5) Correlation heatmap (top-k) ----------
    top_df = df[top_cols].copy()
    top_df = top_df.fillna(top_df.mean(numeric_only=True))
    C = np.corrcoef(top_df.values.T)
    fig, ax = plt.subplots(figsize=(0.5*len(top_cols)+3, 0.5*len(top_cols)+3))
    im = ax.imshow(C, aspect="equal")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(top_cols))); ax.set_xticklabels(top_cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(top_cols))); ax.set_yticklabels(top_cols, fontsize=8)
    ax.set_title("Correlation (top-k features)")
    plt.tight_layout()
    p = os.path.join(out_dir, "corr_heatmap.png"); plt.savefig(p, dpi=200); plt.close()
    paths["corr_heatmap_png"] = p
    pd.DataFrame(C, index=top_cols, columns=top_cols).to_csv(os.path.join(out_dir, "corr_matrix.csv"))
    paths["corr_matrix_csv"] = os.path.join(out_dir, "corr_matrix.csv")

    # ---------- manifest ----------
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(paths, f, indent=2)

    return {"paths": paths, "top_features": top_cols}

df = pd.read_pickle("data/private_df.pkl")
out = visualize_tabular_and_save(df, label_col="identity", top_k=6, use_tsne=True, out_dir="viz_out")
print("Saved files:", json.dumps(out["paths"], indent=2))
print("Top features:", out["top_features"])