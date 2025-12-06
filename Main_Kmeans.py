import os
import re
import csv
import shutil
from glob import glob
from collections import Counter
from pathlib import Path
from Graph_PCA import plot_clusters_pca

import numpy as np
import joblib
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw, ImageFont
import pillow_heif
pillow_heif.register_heif_opener()

# ===== TensorFlow / Keras: ResNet-50 (2048D) =====
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model

# ===== scikit-learn =====
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------
# 0) Ordner & Basiseinstellungen
# ----------------------------------------------------------------------
# Bild-Ordner
IMAGEs = [
    Path(""),
    Path("")
]

IMAGEs = [Path(p) for p in IMAGEs]

# Ergebnisordner
OUT_DIR = Path("cluster_results_resnet50")
FEATURE_DIR = Path("features_resnet50")

# PCA & Clustering
PCA_DIM = 150           
L2_NORMALIZE = True     
K_MIN, K_MAX = 2, 14    
K_FINAL = None          

# Panel/Export
TOP_N = 10


# ----------------------------------------------------------------------
# 1) Bilder einsammeln 
# ----------------------------------------------------------------------
def collect_image_paths(roots):
    exts = ["*.jpeg", "*.jpg", "*.png", "*.webp", "*.heic", "*.heif"]
    paths = []
    for root in roots:
        if not Path(root).exists():
            print(f"[WARN] Root not found: {root}")
            continue
        for ext in exts:
            paths.extend(Path(root).rglob(ext))
    return sorted(paths)

img_paths = collect_image_paths(IMAGEs)
if len(img_paths) == 0:
    raise SystemExit(f"[ERROR] Keine Bilder gefunden in: {', '.join(str(r) for r in IMAGEs)}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Found {len(img_paths)} image files under {', '.join(str(r) for r in IMAGEs)}")



# ----------------------------------------------------------------------
# 2) Feature-Extraktion mit ResNet-50 (pooling='avg' → 2048D) + Cache
# ----------------------------------------------------------------------
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
feature_model = Model(inputs=base.input, outputs=base.output)
IMG_SIZE = (224, 224)

features = []
for i, p in enumerate(img_paths, 1):
    stem = p.stem
    cache_path = FEATURE_DIR / f"{stem}_resnet50_avg.pkl"
    if cache_path.exists():
        vec = joblib.load(cache_path)
    else:
        img = load_img(p, target_size=IMG_SIZE)
        arr = img_to_array(img)[None, ...]
        arr = preprocess_input(arr)
        vec = feature_model.predict(arr, verbose=0).squeeze()  # (2048,)
        joblib.dump(vec, cache_path)
    features.append(vec)
    if (i % 50 == 0) or (i == len(img_paths)):
        print(f"[EXTRACT] {i}/{len(img_paths)} processed …")

X = np.vstack(features)  # (N, 2048)
print("[INFO] Feature matrix:", X.shape)


# ----------------------------------------------------------------------
# 3) Skalierung → PCA → (optional) L2-Norm (≈ spherical K-Means)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=PCA_DIM, random_state=42)
X_red = pca.fit_transform(X_scaled)
print(f"[INFO] PCA reduced shape: {X_red.shape}  |  Var.expl.: {pca.explained_variance_ratio_.sum():.3f}")

if L2_NORMALIZE:
    X_used = normalize(X_red)
    print("[INFO] Applied L2-normalization (recommended for cosine geometry).")
else:
    X_used = X_red

# ----------------------------------------------------------------------
# 3b) Auto-Elbow (Knee Detection)
# ----------------------------------------------------------------------
import numpy as np

def auto_knee(x, y):
    """
    x: list/array der k-Werte (z.B. Ks)
    y: list/array der Metrik (z.B. inertias oder cosine_wcss), y sollte mit k sinken
    Methode: Maximum-Distance-to-Line (robuste Knee-Heuristik)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Normierung auf [0,1] für Stabilität
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    # Gerade durch erste und letzte Punkte
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    v = p2 - p1
    v /= (np.linalg.norm(v) + 1e-12)
    # senkrechter Abstand aller Punkte zur Geraden
    dists = []
    for xi, yi in zip(x_n, y_n):
        p = np.array([xi, yi])
        proj_len = np.dot(p - p1, v)
        proj = p1 + proj_len * v
        d = np.linalg.norm(p - proj)
        dists.append(d)
    idx = int(np.argmax(dists))
    return int(x[idx])



# ----------------------------------------------------------------------
# 4) Elbow & Silhouette (Silhouette mit Kosinus-Geometrie)
# ----------------------------------------------------------------------
Ks = list(range(K_MIN, K_MAX + 1))

# Elbow (Inertia) – immer euklidisch (was KMeans auch optimiert)
inertias = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    km.fit(X_used)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.title("Elbow method (Inertia)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(OUT_DIR / "elbow.png")
plt.close()

k_elbow = auto_knee(Ks, inertias)
print(f"[AUTO] Elbow (Inertia) schlägt k={k_elbow} vor.")

plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.axvline(k_elbow, linestyle="--", color="gray")
plt.title("Elbow (Inertia) mit Auto-Knee")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(OUT_DIR / "elbow_auto.png")
plt.close()


# Silhouette – für L2-normalisierte Daten ist cosine sinnvoll
sil_scores = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels_tmp = km.fit_predict(X_used)
    counts = Counter(labels_tmp)
    if min(counts.values()) < 2:
        sil_scores.append(np.nan)
    else:
        s = silhouette_score(X_used, labels_tmp, metric="cosine" if L2_NORMALIZE else "euclidean")
        sil_scores.append(s)

plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.title(f"Silhouette scores ({'cosine' if L2_NORMALIZE else 'euclidean'})")
plt.xlabel("k")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "silhouette.png")
plt.close()

def cosine_wcss_from_kmeans(X_unit, labels, centers):
    # Zentren auf Einheitsnorm projizieren, dann 1 - cos als „Kosten“
    C = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    loss = 0.0
    for c in range(C.shape[0]):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        # cos(x,c) = x·c  (X_unit ist L2-normalisiert)
        sims = X_unit[idx] @ C[c]
        loss += np.sum(1.0 - sims)
    return float(loss)

cosine_wcss = []
kmeans_cache = {}  # wir nutzen die Fits gleich nochmal
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels_tmp = km.fit_predict(X_used)
    kmeans_cache[k] = (km, labels_tmp)
    cosine_wcss.append(cosine_wcss_from_kmeans(X_used, labels_tmp, km.cluster_centers_))

k_elbow_cos = auto_knee(Ks, cosine_wcss)
print(f"[AUTO] Elbow (Cosine-WCSS) schlägt k={k_elbow_cos} vor.")

plt.figure()
plt.plot(Ks, cosine_wcss, marker="o")
plt.axvline(k_elbow_cos, linestyle="--")
plt.title("Elbow (Cosine-WCSS) mit Auto-Knee")
plt.xlabel("k"); plt.ylabel("Cosine-WCSS"); plt.tight_layout()
plt.savefig(OUT_DIR / "elbow_cosine_auto.png"); plt.close()

print(f"[INFO] Plots saved under: {OUT_DIR}")


# ----------------------------------------------------------------------
# 5) k festlegen (optional interaktiv)
# ----------------------------------------------------------------------
if K_FINAL is None:
    while True:
        try:
            k_in = int(input(f"How many clusters (k) [{K_MIN}–{K_MAX}]? ").strip())
            if k_in >= 2:
                K_FINAL = k_in
                break
            print("Please enter k >= 2.")
        except ValueError:
            print("Invalid input.")
print(f"[INFO] Using k = {K_FINAL}")


# ----------------------------------------------------------------------
# 6) Finale K-Means-Clustering (robuster init)
# ----------------------------------------------------------------------
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=50)
labels = km_final.fit_predict(X_used)
counts = dict(Counter(labels))
print("[INFO] Cluster sizes:", counts)

# Per-Cluster-Silhouette (Diagnose)
try:
    if min(counts.values()) >= 2:
        s_samples = silhouette_samples(X_used, labels, metric="cosine" if L2_NORMALIZE else "euclidean")
        per_cluster_sil = {int(c): float(np.nanmean(s_samples[labels == c])) for c in np.unique(labels)}
        print("[INFO] Per-cluster silhouette:", per_cluster_sil)
except Exception as e:
    print("[WARN] Silhouette per cluster failed:", e)


# ----------------------------------------------------------------------
# 7) Repräsentanten (Zentroid, Medoid, Typicality)
# ----------------------------------------------------------------------
def nearest_to_kmeans_center(X, labels, kmeans_model, top_n=10):
    out = {}
    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        center = kmeans_model.cluster_centers_[cl]
        d = np.linalg.norm(X[idxs] - center, axis=1)
        out[cl] = idxs[np.argsort(d)[:top_n]].tolist()
    return out

def nearest_to_medoid(X, labels, top_n=10, chunk=5000):

    out = {}
    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        m = len(idxs)
        if m == 0:
            out[cl] = []
            continue

        # Kleine Cluster
        if m <= 2000:
            P = X[idxs]
            D = np.linalg.norm(P[:, None] - P[None, :], axis=2)
            medoid_local = np.argmin(D.sum(axis=0))
            medoid = P[medoid_local]
            d = np.linalg.norm(P - medoid, axis=1)
            out[cl] = idxs[np.argsort(d)[:top_n]].tolist()
            continue

        # Große Cluster
        P = X[idxs]                              # [m, d]
        sums = np.zeros(m, dtype=np.float64)     # Distanzsummen je Kandidat
        # Wir akkumulieren Summe_j ||x_i - x_j|| über j in Blöcken
        for j0 in range(0, m, chunk):
            j1 = min(j0 + chunk, m)
            Q = P[j0:j1]                         # [q, d]
            # Distanzen von allen Kandidaten i zu Q: berechne in i-Blöcken
            for i0 in range(0, m, chunk):
                i1 = min(i0 + chunk, m)
                A = P[i0:i1]                     # [p, d]
                # Euclid-Distanzen ohne volle [m,m]-Matrix
                # (A[:,None,:]-Q[None,:,:]) -> [p,q,d]
                dists = np.linalg.norm(A[:, None, :] - Q[None, :, :], axis=2)  # [p,q]
                sums[i0:i1] += dists.sum(axis=1)

        medoid_local = int(np.argmin(sums))
        medoid = P[medoid_local]
        # Abstände zum Medoid (für die Top-N-Repräsentanten)
        # wieder in Blöcken, um RAM zu sparen
        d_to_med = np.empty(m, dtype=np.float64)
        for i0 in range(0, m, chunk):
            i1 = min(i0 + chunk, m)
            A = P[i0:i1]
            d_to_med[i0:i1] = np.linalg.norm(A - medoid, axis=1)
        out[cl] = idxs[np.argsort(d_to_med)[:top_n]].tolist()
    return out


def compute_typicality(X, labels, centers, top_n=10, n_neighbors=10, use_cosine=True):
    """
    Typicality = z(d1) - z(margin) + 0.5*z(density)
    d1: Distanz zur eigenen Mitte
    margin: Abstand zur 2.-nächsten Mitte minus zur eigenen
    density: mittlerer kNN-Abstand (kleiner = dichter)
    """
    # Distanzen zu allen Zentren
    D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    d1 = D.min(axis=1)
    D_sorted = np.sort(D, axis=1)
    margin = D_sorted[:, 1] - D_sorted[:, 0]

    # lokale Dichte
    nn_metric = "cosine" if use_cosine else "euclidean"
    n_neighbors = min(n_neighbors, max(2, X.shape[0]-1))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=nn_metric).fit(X)
    knn_dist, _ = nn.kneighbors(X)
    density = knn_dist.mean(axis=1)

    def z(a):
        a = np.asarray(a, dtype=float)
        return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)

    typicality = z(d1) - z(margin) + 0.5 * z(density)

    # Top-N pro Cluster
    out = {}
    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        order = idxs[np.argsort(typicality[idxs])]
        out[cl] = order[:top_n].tolist()
    return out, typicality

nearest_center = nearest_to_kmeans_center(X_used, labels, km_final, top_n=TOP_N)
nearest_medoid = nearest_to_medoid(X_used, labels, top_n=TOP_N)
typical_map, typicality = compute_typicality(
    X_used, labels, km_final.cluster_centers_, top_n=TOP_N,
    use_cosine=L2_NORMALIZE
)
# ----------------------------------------------------------------------
# 7a) Graph_PCA
# ----------------------------------------------------------------------

plot_clusters_pca(
    X_used,                # dieselben Features, auf denen K-Means trainiert wurde
    labels,
    kmeans_model=km_final, # finales KMeans-Modell
    title=f"K-Means (k={K_FINAL}) im PCA-Raum",
    random_state=42,
    n_components=3,
    pairs=[(0, 1)],          # 2D: PC1 vs. PC2
    plot_3d=True,            # zusätzlich 3D
    components_3d=(0, 1, 2)  # PC1, PC2, PC3
)

# ----------------------------------------------------------------------
# 8) Panels (PNG) + CSVs + Kopien
# ----------------------------------------------------------------------
def _extract_date_from_filename(path_str):
    m = re.search(r'(20\d{2})[-_\.](\d{2})[-_\.](\d{2})', os.path.basename(path_str))
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return os.path.basename(path_str)

def render_clusters_panel_10(nearest_idx_map, img_paths, labels, out_path, thumb=200, padding=16):
    cluster_ids = sorted(nearest_idx_map.keys())
    n_cols = 10
    n_rows = len(cluster_ids)
    cell_h = thumb + padding + 60
    cell_w = thumb + padding
    width = padding + n_cols * cell_w
    height = 50 + n_rows * cell_h

    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for r, cl in enumerate(cluster_ids):
        idxs = nearest_idx_map[cl]
        cluster_size = int(np.sum(labels == cl))
        draw.text((padding, 20 + r * cell_h), f"Cluster {cl:02d} (n={cluster_size})", fill=(0,0,0), font=font)
        for c, i in enumerate(idxs[:n_cols]):
            try:
                im = Image.open(img_paths[i])
                im = ImageOps.exif_transpose(im)
                im.thumbnail((thumb, thumb))
                x = padding + c * cell_w
                y = 40 + r * cell_h
                panel.paste(im, (x, y))
                txt = _extract_date_from_filename(str(img_paths[i]))
                bbox = draw.textbbox((0, 0), txt, font=font)
                txt_w = bbox[2] - bbox[0]
                txt_x = x + (thumb - txt_w) // 2
                txt_y = y + thumb + 10
                draw.text((txt_x, txt_y), txt, fill=(0,0,0), font=font)
            except Exception as e:
                print(f"[WARN] Could not load {img_paths[i]}: {e}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)
    print(f"[INFO] Panel saved: {out_path}")

def save_cluster_filenames(nearest_idx_map, img_paths, out_csv, top_n=10):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "rank", "filename"])
        for cl in sorted(nearest_idx_map.keys()):
            for rank, i in enumerate(nearest_idx_map[cl][:top_n], start=1):
                w.writerow([cl, rank, os.path.basename(str(img_paths[i]))])
    print(f"[INFO] CSV saved: {out_csv}")

def export_all_filenames_with_clusters(labels, img_paths, out_csv, extra_typicality=None):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["filename", "cluster"]
        if extra_typicality is not None:
            header.append("typicality")
        w.writerow(header)
        for i, (lbl, p) in enumerate(zip(labels, img_paths)):
            row = [os.path.basename(str(p)), int(lbl)]
            if extra_typicality is not None:
                row.append(float(extra_typicality[i]))
            w.writerow(row)
    print(f"[INFO] CSV saved: {out_csv}")

def export_typical_images(nearest_idx_map, img_paths, out_dir, method="kmeans", top_n=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cl, idxs in nearest_idx_map.items():
        cluster_dir = out_dir / f"cluster_{cl:02d}_{method}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(idxs[:top_n], start=1):
            src = Path(img_paths[i])
            dst = cluster_dir / f"{rank:02d}_{src.name}"
            shutil.copy(src, dst)
        print(f"[INFO] Copied {min(top_n, len(idxs))} images for cluster {cl} → {cluster_dir}")

# Render-Panels
preview_dir = OUT_DIR / "previews"
render_clusters_panel_10(nearest_center, img_paths, labels, out_path=preview_dir / "clusters_center_panel.png")
render_clusters_panel_10(nearest_medoid, img_paths, labels, out_path=preview_dir / "clusters_medoid_panel.png")
render_clusters_panel_10(typical_map, img_paths, labels, out_path=preview_dir / "clusters_typical_panel.png")

# CSVs
save_cluster_filenames(nearest_center, img_paths, OUT_DIR / "center_filenames.csv", top_n=TOP_N)
save_cluster_filenames(nearest_medoid, img_paths, OUT_DIR / "medoid_filenames.csv", top_n=TOP_N)
save_cluster_filenames(typical_map, img_paths, OUT_DIR / "typical_filenames.csv", top_n=TOP_N)

# Alle Bilder + Cluster (+Typicality)
export_all_filenames_with_clusters(labels, img_paths, OUT_DIR / "all_images_with_clusters.csv", extra_typicality=typicality)

# Kopien der Repräsentanten
rep_dir = OUT_DIR / "representative_images"
export_typical_images(nearest_center, img_paths, rep_dir, method="center", top_n=TOP_N)
export_typical_images(nearest_medoid, img_paths, rep_dir, method="medoid", top_n=TOP_N)
export_typical_images(typical_map, img_paths, rep_dir, method="typical", top_n=TOP_N)

# ----------------------------------------------------------------------
# 8b) Zufallsstichproben pro Cluster (20 pro Cluster) + Panel je Cluster
# ----------------------------------------------------------------------
def sample_random_indices_per_cluster(labels, n=20, seed=42):
    """
    Liefert für jeden Cluster eine Liste zufällig gezogener Indizes (ohne Zurücklegen).
    Klemmt n an die tatsächliche Clustergröße.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for cl in sorted(np.unique(labels)):
        idxs = np.where(labels == cl)[0]
        k = min(n, len(idxs))
        out[cl] = rng.choice(idxs, size=k, replace=False).tolist() if k > 0 else []
    return out

def render_random_panel_per_cluster(random_idx_map, img_paths, out_dir, grid=(4,5), thumb=220, padding=16):
    """
    Ein Panel pro Cluster (Grid, z.B. 4x5 = 20 Bilder).
    """
    rows, cols = grid
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for cl in sorted(random_idx_map.keys()):
        idxs = random_idx_map[cl][: rows*cols]
        n = len(idxs)
        if n == 0:
            continue

        cell_h = thumb + padding + 40
        cell_w = thumb + padding
        width = padding + cols * cell_w
        height = 50 + rows * cell_h

        panel = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.text((padding, 15), f"Cluster {cl:02d} – Random {n}", fill=(0,0,0), font=font)

        for j, i in enumerate(idxs):
            r = j // cols
            c = j % cols
            try:
                im = Image.open(img_paths[i])
                im = ImageOps.exif_transpose(im)
                im.thumbnail((thumb, thumb))
                x = padding + c * cell_w
                y = 40 + r * cell_h
                panel.paste(im, (x, y))
                txt = os.path.basename(str(img_paths[i]))
                bbox = draw.textbbox((0, 0), txt, font=font)
                txt_w = bbox[2] - bbox[0]
                txt_x = x + (thumb - txt_w) // 2
                txt_y = y + thumb + 8
                draw.text((txt_x, txt_y), txt, fill=(0,0,0), font=font)
            except Exception as e:
                print(f"[WARN] Could not load {img_paths[i]}: {e}")

        out_path = out_dir / f"cluster_{cl:02d}_random20_panel.png"
        panel.save(out_path)
        print(f"[INFO] Random panel (per cluster) saved: {out_path}")

# --- Ausführung: Zufall ziehen, Panels (nur per Cluster), CSV + Kopien ---
RANDOM_TOP = 20
random_map = sample_random_indices_per_cluster(labels, n=RANDOM_TOP, seed=123)

random_preview_dir = OUT_DIR / "previews" / "random_panels_per_cluster"
render_random_panel_per_cluster(
    random_map, img_paths,
    out_dir=random_preview_dir,
    grid=(4,5),      # 4×5 = 20 Bilder
    thumb=220,
    padding=16
)

# CSV der Zufallsauswahl (Cluster, Rang = Reihenfolge im Panel)
save_cluster_filenames(random_map, img_paths, OUT_DIR / "random20_filenames.csv", top_n=RANDOM_TOP)

# Kopien der zufälligen Auswahl in Ordnerstruktur
export_typical_images(random_map, img_paths, rep_dir, method="random", top_n=RANDOM_TOP)

print(f"[DONE] Results in: {OUT_DIR.resolve()}")
