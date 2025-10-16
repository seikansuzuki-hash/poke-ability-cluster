# -*- coding: utf-8 -*-
# app.py — 入力フォームから新キャラの推奨特性を表示（日本語対応 / 幅広テーブル / 安定版）
import os
import sys
import pathlib
import importlib
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd

# ---（任意）起動保険：Cloudの依存が消えた時の自己復旧 ---
try:
    import sklearn  # noqa
    import tqdm     # noqa
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    st.toast("Installing dependencies... rerunning")
    st.experimental_rerun()

st.set_page_config(page_title="ポケモン特性・クラスタ推定", layout="centered")

# ---- モジュールの安全読み込み（同ディレクトリ優先）----
ROOT = pathlib.Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 安全に import（ファイル内容の差異に強く）
try:
    pac = importlib.import_module("poke_ability_clustering")
except Exception as e:
    st.error(f"poke_ability_clustering の読み込みに失敗しました。\n\n{e}")
    st.stop()

predict_cluster_and_abilities = getattr(pac, "predict_cluster_and_abilities", None)
load_artifacts = getattr(pac, "load_artifacts", None)
TYPE_ORDER = getattr(pac, "TYPE_ORDER", None)

if not all([predict_cluster_and_abilities, load_artifacts, TYPE_ORDER]):
    st.error("poke_ability_clustering から必要な関数/定数を取得できません。ファイルの内容をご確認ください。")
    st.stop()

ART_PATH = "ability_clusters.joblib"
if not pathlib.Path(ART_PATH).exists():
    st.error(f"学習成果ファイル **{ART_PATH}** が見つかりません。\n"
             "リポジトリ直下に配置してください（無い場合は `python poke_ability_clustering.py` で生成）。")
    st.stop()

@st.cache_resource(show_spinner=True)
def _load_bundle(path: str):
    return load_artifacts(path)

bundle = _load_bundle(ART_PATH)
feature_cols: List[str] = bundle.get("feature_cols", [])
if not feature_cols:
    st.error("feature_cols が空です。ability_clusters.joblib を作り直してください。")
    st.stop()

# UI 用の選択肢（学習時に存在する列から動的取得）
gen_options = sorted({c.replace("gen_", "") for c in feature_cols if c.startswith("gen_")})
egg_options = sorted({c.replace("egg_", "") for c in feature_cols if c.startswith("egg_")})

# 既定値を安全に（候補にないものは外す）
_default_types = [t for t in ["fire", "fighting"] if t in TYPE_ORDER]
_default_eggs = [x for x in ["field", "human-like"] if x in egg_options]
_default_gen = "generation-ix" if "generation-ix" in gen_options else "未指定"

# =========================
#          UI
# =========================
st.title("ポケモンの特性（クラスタリング推定）")
st.caption("PokeAPI由来の学習成果（ability_clusters.joblib）を用いて、似たクラスターから特性分布を推定します。")

with st.form("input_form"):
    st.subheader("種族値 / 体格 / 捕獲率")
    c1, c2, c3 = st.columns(3)
    with c1:
        hp = st.number_input("HP", min_value=1, max_value=255, value=80, step=1)
        attack = st.number_input("こうげき", min_value=1, max_value=255, value=105, step=1)
        defense = st.number_input("ぼうぎょ", min_value=1, max_value=255, value=75, step=1)
    with c2:
        sp_attack = st.number_input("とくこう", min_value=1, max_value=255, value=60, step=1)
        sp_defense = st.number_input("とくぼう", min_value=1, max_value=255, value=75, step=1)
        speed = st.number_input("すばやさ", min_value=1, max_value=255, value=95, step=1)
    with c3:
        weight = st.number_input("おもさ（hectogram）", min_value=1, max_value=9999, value=320, step=1,
                                 help="PokeAPI単位: hectogram（100g）。320 → 32.0kg")
        height = st.number_input("たかさ（decimeter）", min_value=1, max_value=99, value=14, step=1,
                                 help="PokeAPI単位: decimeter（10cm）。14 → 1.4m")
        capture_rate = st.number_input("捕獲率", min_value=1, max_value=255, value=45, step=1)

    st.divider()
    st.subheader("タイプ / タマゴ / 世代")
    types = st.multiselect("タイプ（最大2）", TYPE_ORDER, default=_default_types, help="2つまで選択推奨")
    if len(types) > 2:
        st.warning("タイプは最大2つまで。先頭2つだけを使用します。")
    egg_groups = st.multiselect("タマゴグループ（任意）", egg_options, default=_default_eggs)
    generation = st.selectbox("世代（任意）", ["未指定"] + gen_options,
                              index=(["未指定"] + gen_options).index(_default_gen))

    st.divider()
    st.subheader("出力オプション")
    k = st.slider("候補の件数（Top-k）", min_value=1, max_value=10, value=5)
    lang_label = {"ja": "日本語（漢字かな）", "ja-Hrkt": "カタカナ", "en": "英語（そのまま）"}
    lang = st.selectbox("出力言語", options=["ja", "ja-Hrkt", "en"], index=0, format_func=lambda x: lang_label[x])

    submitted = st.form_submit_button("特性を推定する")

if submitted:
    new_mon: Dict[str, Optional[object]] = {
        "hp": int(hp),
        "attack": int(attack),
        "defense": int(defense),
        "sp_attack": int(sp_attack),
        "sp_defense": int(sp_defense),
        "speed": int(speed),
        "weight": int(weight),
        "height": int(height),
        "base_experience": 200,  # 不明なら仮置き（0でも可）
        "capture_rate": int(capture_rate),
        "types": types[:2],                # 念のため2つに制限
        "egg_groups": egg_groups,
        "generation": generation if generation != "未指定" else None,
    }
    out_lang = None if lang == "en" else lang

    try:
        res = predict_cluster_and_abilities(new_mon, artifacts_path=ART_PATH, topk=k, out_lang=out_lang)
    except Exception as e:
        st.error(f"推定でエラーが発生しました：\n\n{e}")
        st.stop()

    st.success(f"推定クラスターID: {res['pred_cluster']}")
    st.write("**推奨特性（上位）**")

    # 推奨特性テーブル（幅広表示）
    ab = res.get("suggested_abilities") or []
    if not ab:
        st.info("このクラスターには主特性の分布がまだ十分にありません。新作の“未知傾向”かもしれません。")
    else:
        df_abilities = pd.DataFrame(ab, columns=["特性", "クラスタ内比率"])
        df_abilities["クラスタ内比率"] = df_abilities["クラスタ内比率"].round(3)
        st.dataframe(df_abilities, use_container_width=True)

    # 2カラム：近傍 / クラスタ例（どちらも幅広）
    colA, colB = st.columns(2)
    with colA:
        st.write("**近傍個体（参考）**")
        nn = res.get("nearest_neighbors") or []
        if not nn:
            st.write("該当なし")
        else:
            nn_df = pd.DataFrame(nn)  # name / abilities / distance
            # 列名の日本語化（出力言語に合わせる）
            rename_map = {"name": "名前", "abilities": "特性", "distance": "距離"} if out_lang else {}
            nn_df = nn_df.rename(columns=rename_map)
            if "distance" in nn_df.columns:
                nn_df["distance"] = nn_df["distance"].round(5)
            st.dataframe(nn_df, use_container_width=True)

    with colB:
        st.write("**クラスタ例（学習データからランダム抽出）**")
        ex = res.get("cluster_examples") or []
        ex_df = pd.DataFrame({"例ポケモン": ex}) if ex else pd.DataFrame({"例ポケモン": []})
        st.dataframe(ex_df, use_container_width=True)

# フッターメモ
st.caption("※ Python は Streamlit Cloud の **Python 3.12** 設定で運用するのが安定です。")
