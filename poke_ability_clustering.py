# -*- coding: utf-8 -*-
# poke_ability_clustering.py
#  - PokeAPIから取得 → 特徴量化 → PCA+KMeansクラスタリング（K自動探索）
#  - クラスタごとの特性分布から新キャラの「推奨特性TOPk」を返す
#  - 日本語出力対応（特性名 / ポケモン名）：out_lang="ja" or "ja-Hrkt"
# 依存:
#   pip install requests tqdm scikit-learn pandas numpy joblib

import os, time, sys, json, math, random
from typing import Dict, List, Tuple, Optional
from collections import Counter

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import joblib

BASE = "https://pokeapi.co/api/v2"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "poke-ml-unsupervised/1.0"})

TYPE_ORDER = [
    "normal","fire","water","electric","grass","ice","fighting","poison","ground",
    "flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy"
]

# ------------------------------
# 0) Utilities (retry / sleep)
# ------------------------------
def _get(url, timeout=30, max_retry=3, sleep_between=0.2):
    for i in range(max_retry):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            if i == max_retry - 1:
                raise
            time.sleep(sleep_between * (i + 1))

# ------------------------------
# 1) Fetch PokeAPI (with cache)
# ------------------------------
def fetch_all_pokemon_names(limit=2000) -> List[str]:
    r = _get(f"{BASE}/pokemon?limit={limit}&offset=0")
    return [x["name"] for x in r.json()["results"]]

def fetch_pokemon_detail(name: str) -> Dict:
    return _get(f"{BASE}/pokemon/{name}").json()

def fetch_species_detail(name_or_id: str) -> Dict:
    return _get(f"{BASE}/pokemon-species/{name_or_id}").json()

def row_from_payload(poke: Dict, species: Dict) -> Dict:
    # 主特性（is_hidden=false）複数ありうる
    primary_abilities = [a["ability"]["name"] for a in poke["abilities"] if not a.get("is_hidden", False)]

    stats_map = {s["stat"]["name"]: s["base_stat"] for s in poke["stats"]}
    hp   = stats_map.get("hp", np.nan)
    atk  = stats_map.get("attack", np.nan)
    defe = stats_map.get("defense", np.nan)
    spa  = stats_map.get("special-attack", np.nan)
    spd  = stats_map.get("special-defense", np.nan)
    spe  = stats_map.get("speed", np.nan)

    types = [t["type"]["name"] for t in poke["types"]]
    type_vec = {f"type_{t}": 1 if t in types else 0 for t in TYPE_ORDER}

    row = {
        "id": poke["id"],
        "name": poke["name"],
        "hp": hp, "attack": atk, "defense": defe, "sp_attack": spa, "sp_defense": spd, "speed": spe,
        "weight": poke.get("weight", np.nan),               # hectograms
        "height": poke.get("height", np.nan),               # decimeters
        "base_experience": poke.get("base_experience", np.nan),
        "generation": species.get("generation", {}).get("name"),
        "capture_rate": species.get("capture_rate", np.nan),
        "egg_groups": ",".join(sorted([e["name"] for e in species.get("egg_groups", [])])),
        "primary_abilities": ",".join(primary_abilities) if primary_abilities else ""
    }
    row.update(type_vec)
    return row

def build_dataset(max_pokemon=1025, sleep_sec=0.1, cache_csv="pokemon_raw_cache.csv") -> pd.DataFrame:
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)

    names = fetch_all_pokemon_names(limit=max_pokemon)
    rows = []
    for name in tqdm(names, desc="Fetching Pokémon"):
        try:
            poke = fetch_pokemon_detail(name)
            species = fetch_species_detail(poke["species"]["name"])
            rows.append(row_from_payload(poke, species))
            time.sleep(sleep_sec)  # polite delay
        except Exception as e:
            sys.stderr.write(f"[skip] {name}: {e}\n")
    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    return df

# ------------------------------
# 2) Feature engineering
# ------------------------------
def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # egg_groups -> multi-hot
    all_eggs = sorted({g for s in df["egg_groups"].fillna("") for g in s.split(",") if g})
    for g in all_eggs:
        df[f"egg_{g}"] = df["egg_groups"].apply(lambda s: 1 if g in s.split(",") else 0)

    # generation -> one-hot
    gens = sorted(df["generation"].dropna().unique().tolist())
    for g in gens:
        df[f"gen_{g}"] = (df["generation"] == g).astype(int)

    # 数値列・one-hot列（文字列の原カラムは入れない）
    num_cols = ["hp","attack","defense","sp_attack","sp_defense","speed",
                "weight","height","base_experience","capture_rate"]
    type_cols = [c for c in df.columns if c.startswith("type_")]
    egg_cols  = [c for c in df.columns if c.startswith("egg_") and c != "egg_groups"]
    gen_cols  = [c for c in df.columns if c.startswith("gen_")]

    feature_cols = num_cols + type_cols + egg_cols + gen_cols

    # 安全に数値化
    X = (df[feature_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0.0)
         .astype(float))

    meta = df[["id","name","primary_abilities","egg_groups","generation"]].copy()
    return X, meta

# ------------------------------
# 2.5) Localization helpers (Japanese output)
# ------------------------------
L10N_CACHE: Dict[Tuple[str, str, str], str] = {}

def _get_localized_name_from_payload(payload: dict, lang: str) -> Optional[str]:
    for n in payload.get("names", []):
        if n.get("language", {}).get("name") == lang:
            return n.get("name")
    # fallback
    fb = "ja-Hrkt" if lang == "ja" else "ja"
    for n in payload.get("names", []):
        if n.get("language", {}).get("name") == fb:
            return n.get("name")
    return None

def get_ability_ja(ability_en: str, lang: str = "ja") -> str:
    if not ability_en:
        return ability_en
    key = ("ability", ability_en, lang)
    if key in L10N_CACHE:
        return L10N_CACHE[key]
    try:
        r = _get(f"{BASE}/ability/{ability_en}", timeout=20)
        ja = _get_localized_name_from_payload(r.json(), lang) or ability_en
    except Exception:
        ja = ability_en
    L10N_CACHE[key] = ja
    return ja

def get_pokemon_ja(name_en: str, lang: str = "ja") -> str:
    key = ("pokemon", name_en, lang)
    if key in L10N_CACHE:
        return L10N_CACHE[key]
    try:
        r = _get(f"{BASE}/pokemon-species/{name_en}", timeout=20)
        ja = _get_localized_name_from_payload(r.json(), lang) or name_en
    except Exception:
        ja = name_en
    L10N_CACHE[key] = ja
    return ja

# ------------------------------
# 3) Clustering with K auto-select
# ------------------------------
def fit_clustering(X: pd.DataFrame, k_min=8, k_max=30, pca_dim=20, seed=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA（速度とノイズ低減）
    pca = PCA(n_components=min(pca_dim, Xs.shape[1]))
    Z = pca.fit_transform(Xs)

    # K探索（silhouette最大）
    best_k, best_score, best_model = None, -1, None
    for k in range(k_min, min(k_max, len(X))):
        try:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)  # 互換性のため n_init=10
            labels = km.fit_predict(Z)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(Z, labels)
            if score > best_score:
                best_k, best_score, best_model = k, score, km
        except Exception:
            pass

    if best_model is None:
        # フォールバック（小さめK）
        best_k = max(8, min(15, len(X)//20))
        best_model = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(Z)
        labels = best_model.labels_
        best_score = silhouette_score(Z, labels) if len(set(labels)) > 1 else -1
    else:
        labels = best_model.labels_

    return {
        "scaler": scaler,
        "pca": pca,
        "kmeans": best_model,
        "labels": labels,
        "embedding": Z,
        "silhouette": float(best_score)
    }

def summarize_clusters(model_bundle: Dict, X: pd.DataFrame, meta: pd.DataFrame) -> List[Dict]:
    labels = model_bundle["labels"]
    k = len(set(labels))
    df = meta.copy()
    df["label"] = labels

    clusters = []
    for c in range(k):
        sub = df[df["label"] == c]
        # 主特性（複数カンマ区切り）を展開
        abilities = []
        for s in sub["primary_abilities"].fillna(""):
            if s:
                abilities.extend([a.strip() for a in s.split(",") if a.strip()])
        ability_counts = Counter(abilities)
        total = sum(ability_counts.values()) if ability_counts else 0
        top_abilities = [(a, cnt/total) for a, cnt in ability_counts.most_common(10)] if total>0 else []

        ex = sub["name"].tolist()
        if len(ex) > 6:
            random.seed(0)
            ex = random.sample(ex, 6)

        clusters.append({
            "cluster_id": int(c),
            "size": int(len(sub)),
            "top_abilities": top_abilities,  # (ability, ratio)
            "examples": ex
        })
    return clusters

# ------------------------------
# 4) Save / Load
# ------------------------------
def save_artifacts(path="ability_clusters.joblib", **kwargs):
    joblib.dump(kwargs, path)
    print(f"[saved] {path}")

def load_artifacts(path="ability_clusters.joblib"):
    return joblib.load(path)

# ------------------------------
# 5) Inference for a new Pokémon
# ------------------------------
def build_row_from_features(feat: dict, feature_cols: List[str]) -> pd.DataFrame:
    # feature_cols は学習時の列順
    row = {c: 0.0 for c in feature_cols}
    # 数値
    for c in ["hp","attack","defense","sp_attack","sp_defense","speed","weight","height","base_experience","capture_rate"]:
        if c in row:
            row[c] = float(feat.get(c, 0) or 0)
    # タイプ
    for t in feat.get("types", []):
        c = f"type_{t}"
        if c in row:
            row[c] = 1.0
    # タマゴ
    for g in feat.get("egg_groups", []):
        c = f"egg_{g}"
        if c in row:
            row[c] = 1.0
    # 世代
    gen = feat.get("generation")
    if gen:
        c = f"gen_{gen}"
        if c in row:
            row[c] = 1.0
    return pd.DataFrame([row])

def predict_cluster_and_abilities(new_feat: dict, artifacts_path="ability_clusters.joblib", topk=5, out_lang: Optional[str] = None):
    bundle = load_artifacts(artifacts_path)
    feature_cols = bundle["feature_cols"]
    X = bundle["X"]              # 学習時の特徴行列（DataFrame）
    meta = bundle["meta"]        # id, name, primary_abilities, ...
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    kmeans = bundle["kmeans"]
    clusters = bundle["clusters"] # プロファイル

    X_new = build_row_from_features(new_feat, feature_cols)
    Z_new = pca.transform(scaler.transform(X_new))[0]
    label = int(kmeans.predict([Z_new])[0])

    # クラスターの特性上位
    cluster_info = next((c for c in clusters if c["cluster_id"] == label), None)
    top_abilities = cluster_info["top_abilities"][:topk] if cluster_info else []

    # 近傍個体（参考提示）
    nn = NearestNeighbors(n_neighbors=min(10, len(X)), metric="euclidean")
    Z_all = pca.transform(scaler.transform(X))
    nn.fit(Z_all)
    dists, idxs = nn.kneighbors([Z_new], return_distance=True)
    neighbors = []
    for d, i in zip(dists[0], idxs[0]):
        neighbors.append({
            "name": meta.iloc[i]["name"],
            "abilities": meta.iloc[i]["primary_abilities"],
            "distance": float(d)
        })

    result = {
        "pred_cluster": label,
        "suggested_abilities": top_abilities,  # (ability_en, ratio)
        "cluster_examples": cluster_info["examples"] if cluster_info else [],
        "nearest_neighbors": neighbors[:10]
    }

    # --- 日本語化（任意） ---
    if out_lang in ("ja", "ja-Hrkt"):
        # 特性名
        result["suggested_abilities"] = [
            (get_ability_ja(a_en, out_lang), ratio) for (a_en, ratio) in result["suggested_abilities"]
        ]
        # 例と近傍のポケモン名
        result["cluster_examples"] = [get_pokemon_ja(n, out_lang) for n in result["cluster_examples"]]
        for n in result["nearest_neighbors"]:
            n["name"] = get_pokemon_ja(n["name"], out_lang)
            if n.get("abilities"):
                en_list = [x.strip() for x in n["abilities"].split(",") if x.strip()]
                n["abilities"] = ", ".join(get_ability_ja(x, out_lang) for x in en_list) if en_list else ""

    return result

# ------------------------------
# 6) Main
# ------------------------------
def main():
    # 学習
    df = build_dataset()
    X, meta = make_features(df)
    model_bundle = fit_clustering(X, k_min=8, k_max=30, pca_dim=20, seed=42)
    clusters = summarize_clusters(model_bundle, X, meta)

    # 保存（推論に必要な最小限＋解釈用情報）
    artifacts = {
        "feature_cols": list(X.columns),
        "X": X, "meta": meta,
        "scaler": model_bundle["scaler"],
        "pca": model_bundle["pca"],
        "kmeans": model_bundle["kmeans"],
        "clusters": clusters,
        "silhouette": model_bundle["silhouette"]
    }
    save_artifacts("ability_clusters.joblib", **artifacts)

    print(f"Silhouette: {model_bundle['silhouette']:.4f}")
    print("Cluster sizes & top abilities (head):")
    for c in clusters[:5]:
        print(f"  C{c['cluster_id']:02d} size={c['size']}, top={[(a, round(r,3)) for a,r in c['top_abilities'][:3]]}")

    # ---- デモ推論（仮の新キャラ）----
    new_mon = {
        "hp": 80, "attack": 105, "defense": 75, "sp_attack": 60, "sp_defense": 75, "speed": 95,
        "weight": 320, "height": 14, "base_experience": 200, "capture_rate": 45,
        "types": ["fire","fighting"],
        "egg_groups": ["field","human-like"],
        "generation": "generation-ix"
    }
    res = predict_cluster_and_abilities(new_mon, "ability_clusters.joblib", topk=5, out_lang="ja")
    print("\n[新キャラ 推奨特性（クラスタベース）]")
    print(" クラスターID:", res["pred_cluster"])
    print(" 推奨特性TOP5:", [(a, round(r,3)) for a,r in res["suggested_abilities"]])
    print(" 例ポケモン:", res["cluster_examples"])
    print(" 近傍（名前 / 特性 / 距離）:")
    for n in res["nearest_neighbors"][:5]:
        print("  -", n["name"], "|", n["abilities"], "| d=", round(n["distance"], 3))

if __name__ == "__main__":
    main()
