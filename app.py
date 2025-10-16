# -*- coding: utf-8 -*-
# app.py — 入力フォームから新キャラの推奨特性を表示（日本語対応・安定版）
import streamlit as st
import numpy as np
import pandas as pd
import sys, pathlib, importlib
from typing import List, Dict

st.set_page_config(page_title="ポケモン特性・クラスタ推定", layout="centered")

# ---- モジュールの安全読み込み（同ディレクトリ優先）----
ROOT = pathlib.Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    pac = importlib.import_module("poke_ability_clustering")
except Exception as e:
    st.error(f"poke_ability_clustering の読み込みに失敗しました。\n{e}")
    st.stop()

predict_cluster_and_abilities = getattr(pac, "predict_cluster_and_abilities", None)
load_artifacts = getattr(pac, "load_artifacts", None)
TYPE_ORDER = getattr(pac, "TYPE_ORDER", None)

if not all([predict_cluster_and_abilities, load_artifacts, TYPE_ORDER]):
    st.error("poke_ability_clustering から必要な関数/定数を取得できません。ファイルの内容をご確認ください。")
    st.stop()

# ---- アーティファクト読み込み ----
ART_PATH = "ability_clusters.joblib"
try:
    bundle = load_artifacts(ART_PATH)
except Exception as e:
    st.error(f"学習成果ファイル {ART_PATH} を読み込めませんでした。先に poke_ability_clustering.py を実行して作成してください。\n\n詳細: {e}")
    st.stop()

feature_cols: List[str] = bundle["feature_cols"]

# UI 用の選択肢（学習時に存在する列から動的取得）—順序と内容を固定化
gen_options = sorted({c.replace("gen_", "") for c in feature_cols if c.startswith("gen_")})
egg_options = sorted({c.replace("egg_", "") for c in feature_cols if c.startswith("egg_")})
type_options = list(TYPE_ORDER)  # 参照元をコピーしておく（将来の変更の影響を避ける）

# 既定値（候補に存在するもののみ）
default_types = [t for t in ["fire", "fighting"] if t in type_options][:2]
default_eggs = [x for x in ["field", "human-like"] if x in egg_options]
default_gen = "generation-ix" if "generation-ix" in gen_options else "未指定"

st.title("🧪 ポケモン新キャラの推奨特性（クラスタリング推定）")
st.caption("PokeAPI由来の学習成果（ability_clusters.joblib）を用いて、似たクラスターから特性分布を推定します。")

# 結果出力用のプレースホルダ（レイアウト安定化に寄与）
result_info = st.container()
result_cols = st.container()

with st.form("input_form", clear_on_submit=False):
    st.subheader("✨ 基本パラメータ")
    c1, c2, c3 = st.columns(3)
    with c1:
        hp = st.number_input("HP", min_value=1, max_value=255, value=80, step=1, key="hp")
        attack = st.number_input("こうげき", min_value=1, max_value=255, value=105, step=1, key="atk")
        defense = st.number_input("ぼうぎょ", min_value=1, max_value=255, value=75, step=1, key="def")
    with c2:
        sp_attack = st.number_input("とくこう", min_value=1, max_value=255, value=60, step=1, key="spa")
        sp_defense = st.number_input("とくぼう", min_value=1, max_value=255, value=75, step=1, key="spd")
        speed = st.number_input("すばやさ", min_value=1, max_value=255, value=95, step=1, key="spe")
    with c3:
        weight = st.number_input("おもさ（hectogram）", min_value=1, max_value=9999, value=320, step=1,
                                 help="PokeAPI単位: hectogram（10^2 g）。320 → 32.0kg", key="weight")
        height = st.number_input("たかさ（decimeter）", min_value=1, max_value=99, value=14, step=1,
                                 help="PokeAPI単位: decimeter（10 cm）。14 → 1.4m", key="height")
        capture_rate = st.number_input("捕獲率", min_value=1, max_value=255, value=45, step=1, key="caprate")

    st.divider()
    st.subheader("🛡️ タイプ / タマゴ / 世代")
    types = st.multiselect("タイプ（最大2を推奨）", type_options, default=default_types, key="types")
    if len(types) > 2:
        st.warning("タイプは最大2つまでを推奨します。最初の2つのみ使用します。")

    # egg_options が空でも落ちない
    egg_groups = st.multiselect("タマゴグループ（任意）", egg_options, default=default_eggs, key="eggs")

    gen_select_options = ["未指定"] + gen_options
    try:
        gen_index = gen_select_options.index(default_gen)
    except ValueError:
        gen_index = 0
    generation = st.selectbox("世代（任意）", gen_select_options, index=gen_index, key="gen")

    st.divider()
    st.subheader("🈶 出力オプション")
    k = st.slider("候補の件数（Top-k）", min_value=1, max_value=10, value=5, key="topk")
    lang_label = {"ja": "日本語（漢字かな）", "ja-Hrkt": "カタカナ", "en": "英語（そのまま）"}
    lang = st.selectbox("出力言語", options=["ja", "ja-Hrkt", "en"], index=0,
                        format_func=lambda x: lang_label[x], key="lang")

    submitted = st.form_submit_button("推奨特性を推定する 🚀", use_container_width=True)

if submitted:
    new_mon: Dict = {
        "hp": int(hp),
        "attack": int(attack),
        "defense": int(defense),
        "sp_attack": int(sp_attack),
        "sp_defense": int(sp_defense),
        "speed": int(speed),
        "weight": int(weight),
        "height": int(height),
        "base_experience": 200,
        "capture_rate": int(capture_rate),
        "types": types[:2],
        "egg_groups": egg_groups,
        "generation": generation if generation != "未指定" else None,
    }

    out_lang = None if lang == "en" else lang
    try:
        res = predict_cluster_and_abilities(new_mon, artifacts_path=ART_PATH, topk=k, out_lang=out_lang)
    except Exception as e:
        result_info.error(f"推定でエラーが発生しました: {e}")
    else:
        with result_info:
            st.success(f"推定クラスターID: {res['pred_cluster']}")

        with result_cols:
            left, right = st.columns([1, 2])
            with left:
                st.write("**クラスター例（代表）**")
                ex_list = res.get("cluster_examples", [])
                st.write("、".join(ex_list) if ex_list else "該当なし")

                st.write("**推奨特性（上位）**")
                sug = res.get("suggested_abilities") or []
                if not sug:
                    st.info("このクラスターには主特性の分布がまだ十分にありません。")
                else:
                    df_abilities = pd.DataFrame(sug, columns=["特性", "クラスタ内比率"])
                    df_abilities["クラスタ内比率"] = df_abilities["クラスタ内比率"].round(3)
                    st.dataframe(df_abilities, use_container_width=True)

            with right:
                st.write("**近傍個体（参考）**")
                nn = res.get("nearest_neighbors") or []
                if not nn:
                    st.write("該当なし")
                else:
                    nn_df = pd.DataFrame(nn)
                    nn_df["distance"] = nn_df["distance"].round(5)
                    st.dataframe(nn_df, use_container_width=True)
