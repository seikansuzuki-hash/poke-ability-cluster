# -*- coding: utf-8 -*-
# app.py — 日本語UI版（やっくん表記に寄せる）
import os, sys, pathlib, importlib
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd

st.set_page_config(page_title="特性クラスタ推定（日本語UI）", layout="centered")

# リポジトリ直下優先で import
ROOT = pathlib.Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# モジュール読込
try:
    pac = importlib.import_module("poke_ability_clustering")
except Exception as e:
    st.error(f"poke_ability_clustering の読み込みに失敗しました。\n\n{e}")
    st.stop()

predict_cluster_and_abilities = getattr(pac, "predict_cluster_and_abilities", None)
load_artifacts = getattr(pac, "load_artifacts", None)
TYPE_ORDER = getattr(pac, "TYPE_ORDER", None)
if not all([predict_cluster_and_abilities, load_artifacts, TYPE_ORDER]):
    st.error("必要な関数/定数が見つかりません。poke_ability_clustering.py を確認してください。")
    st.stop()

ART_PATH = "ability_clusters.joblib"
if not pathlib.Path(ART_PATH).exists():
    st.error(f"学習成果 {ART_PATH} がありません。まずローカルで生成してコミットしてください。")
    st.stop()

@st.cache_resource(show_spinner=True)
def _load_bundle(path: str):
    return load_artifacts(path)

bundle = _load_bundle(ART_PATH)
feature_cols: List[str] = bundle.get("feature_cols", [])
if not feature_cols:
    st.error("feature_cols が空です。ability_clusters.joblib を作り直してください。")
    st.stop()

# ===== 日本語ラベル =====
# タイプ（やっくん＆公式表記準拠）
TYPE_JA = {
    "normal":"ノーマル","fire":"ほのお","water":"みず","electric":"でんき","grass":"くさ","ice":"こおり",
    "fighting":"かくとう","poison":"どく","ground":"じめん","flying":"ひこう","psychic":"エスパー",
    "bug":"むし","rock":"いわ","ghost":"ゴースト","dragon":"ドラゴン","dark":"あく","steel":"はがね","fairy":"フェアリー"
}
# タマゴグループ（一般的な日本語対訳）
EGG_JA = {
    "monster":"怪獣","human-like":"人型","field":"陸上","plant":"植物","bug":"虫","mineral":"鉱物",
    "flying":"飛行","amorphous":"不定形","water1":"水中1","water2":"水中2","water3":"水中3",
    "fairy":"妖精","dragon":"ドラゴン","ditto":"メタモン","undiscovered":"タマゴ未発見"
}

# 選択肢を学習済み列から生成
gen_options = sorted({c.replace("gen_", "") for c in feature_cols if c.startswith("gen_")})
egg_options = sorted({c.replace("egg_", "") for c in feature_cols if c.startswith("egg_")})

# 既定値
_default_types_en = [t for t in ["fire","fighting"] if t in TYPE_ORDER]
_default_eggs_en = [x for x in ["field","human-like"] if x in egg_options]
_default_gen = "generation-ix" if "generation-ix" in gen_options else "未指定"

# ===== ヘッダ =====
st.title("特性クラスタ推定")
st.caption("タイプ重視（> タマゴ > 体格 > 種族値）で学習したクラスタから、特性の“傾向”を推定します。")

# ===== フォーム =====
with st.form("input_form"):
    st.subheader("基本情報")
    col1, col2 = st.columns(2)
    with col1:
        # kg / m 表示 (内部は hectogram / decimeter で渡す)
        weight_kg = st.number_input("重さ（kg）", min_value=0.1, max_value=999.9, value=32.0, step=0.1,
                                    help="従来の 320[hectogram] に相当")
        height_m  = st.number_input("高さ（m）",  min_value=0.1, max_value=9.9,   value=1.4,  step=0.1,
                                    help="従来の 14[decimeter] に相当")
        capture_rate = st.number_input("捕獲率（ゲットしやすさ）", min_value=1, max_value=255, value=45, step=1)
        base_exp = st.number_input("基礎経験値", min_value=0, max_value=1000, value=200, step=1)
    with col2:
        st.caption("※ 種族値")
        hp = st.number_input("HP", min_value=1, max_value=255, value=80, step=1)
        attack = st.number_input("こうげき", min_value=1, max_value=255, value=105, step=1)
        defense = st.number_input("ぼうぎょ", min_value=1, max_value=255, value=75, step=1)
        sp_attack = st.number_input("とくこう", min_value=1, max_value=255, value=60, step=1)
        sp_defense = st.number_input("とくぼう", min_value=1, max_value=255, value=75, step=1)
        speed = st.number_input("すばやさ", min_value=1, max_value=255, value=95, step=1)

    st.divider()
    st.subheader("タイプ / タマゴ / 世代")
    # タイプ：日本語表示、内部英語
    types_en = st.multiselect(
        "タイプ（最大2）",
        TYPE_ORDER,
        default=_default_types_en,
        help="複合タイプは2つまで。",
        format_func=lambda k: TYPE_JA.get(k, k)
    )
    if len(types_en) > 2:
        st.warning("タイプは最大2つです。先頭2つのみ使用します。")

    # タマゴグループ：日本語表示、内部英語
    eggs_en = st.multiselect(
        "タマゴグループ（任意）",
        egg_options,
        default=_default_eggs_en,
        format_func=lambda k: EGG_JA.get(k, k)
    )

    generation = st.selectbox(
        "世代（任意）",
        ["未指定"] + gen_options,
        index=(["未指定"] + gen_options).index(_default_gen),
        format_func=lambda g: g if g == "未指定" else g.replace("generation-", "第").replace("i","Ⅰ").replace("v","Ⅴ")
    )

    st.divider()
    st.subheader("出力設定")
    k = st.slider("推奨特性の件数（Top-k）", 1, 10, 5)
    lang_label = {"ja":"日本語（漢字かな）","ja-Hrkt":"カタカナ","en":"英語"}
    lang = st.selectbox("表示言語", ["ja","ja-Hrkt","en"], index=0, format_func=lambda x: lang_label[x])

    submitted = st.form_submit_button("推定する")

if submitted:
    # 単位変換：kg→hectogram, m→decimeter
    weight_hg = int(round(weight_kg * 10))   # 1kg = 10 hg
    height_dm = int(round(height_m  * 10))   # 1m  = 10 dm

    new_mon: Dict = {
        "hp": int(hp),
        "attack": int(attack),
        "defense": int(defense),
        "sp_attack": int(sp_attack),
        "sp_defense": int(sp_defense),
        "speed": int(speed),
        "weight": weight_hg,
        "height": height_dm,
        "base_experience": int(base_exp),
        "capture_rate": int(capture_rate),
        "types": types_en[:2],
        "egg_groups": eggs_en,
        "generation": generation if generation != "未指定" else None,
    }
    out_lang = None if lang == "en" else lang

    try:
        res = predict_cluster_and_abilities(new_mon, artifacts_path=ART_PATH, topk=k, out_lang=out_lang)
    except Exception as e:
        st.error(f"推定でエラーが発生しました：\n\n{e}")
        st.stop()

    # ===== 結果表示 =====
    st.success(f"推定クラスターID：{res['pred_cluster']}")

    # 推奨特性
    st.markdown("### 推奨特性（クラスタ内の上位傾向）")
    ab = res.get("suggested_abilities") or []
    if not ab:
        st.info("このクラスタには主特性の分布が十分にありません。新作の“未知傾向”かもしれません。")
    else:
        df_ab = pd.DataFrame(ab, columns=["特性", "比率"])
        df_ab["比率"] = df_ab["比率"].round(3)
        st.dataframe(df_ab, use_container_width=True)

    # 2カラム：近傍/例
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### 近傍ポケモン（参考）")
        nn = res.get("nearest_neighbors") or []
        if nn:
            nn_df = pd.DataFrame(nn).rename(columns={"name":"名前","abilities":"特性","distance":"距離"})
            if "距離" in nn_df.columns:
                nn_df["距離"] = nn_df["距離"].round(5)
            st.dataframe(nn_df, use_container_width=True, height=300)
        else:
            st.write("該当なし")

    with cb:
        st.markdown("#### クラスタ例（学習データから）")
        ex = res.get("cluster_examples") or []
        st.dataframe(pd.DataFrame({"例ポケモン": ex}), use_container_width=True, height=300)

# フッタ：参考リンク
st.caption("表記や用語はポケモン徹底攻略（やっくん）の図鑑・タマゴ表記に寄せています。")
st.caption("参考: ピカチュウの図鑑ページ（基礎経験値・捕獲率の表記など） / タマゴグループ表記 / タイプ名の日本語表記")
st.caption("© Pokémon. 本ツールは非公式のファンアプリです。")
