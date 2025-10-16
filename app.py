# app.py — デプロイ失敗の原因を特定する診断モード
import streamlit as st
import sys, os, pathlib, importlib, traceback

st.set_page_config(page_title="診断: poke-ability-cluster", layout="wide")
st.title("🔎 診断: Streamlit Cloud 起動トラブル")


with st.expander("requirements.txt の中身（先頭200文字）", expanded=True):
    try:
        txt = pathlib.Path("requirements.txt").read_text(encoding="utf-8", errors="ignore")
        st.code(txt[:200] + ("\n...（省略）" if len(txt) > 200 else ""))
    except Exception as e:
        st.error("requirements.txt を読めませんでした")
        st.exception(e)


# 1) Python/依存バージョンを表示
with st.expander("Python & ライブラリ情報", expanded=True):
    import platform
    st.write({
        "python": sys.version,
        "platform": platform.platform(),
    })
    try:
        import numpy as np, pandas as pd, sklearn, requests, joblib
        st.write({
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "requests": requests.__version__,
            "joblib": joblib.__version__,
        })
    except Exception as e:
        st.error("依存の import で失敗しました。requirements.txt を見直してください。")
        st.exception(e)

# 2) リポジトリ直下のファイル一覧を表示
with st.expander("カレントディレクトリのファイル", expanded=True):
    root = pathlib.Path(".").resolve()
    st.write("cwd:", str(root))
    files = []
    for p in sorted(root.glob("*")):
        if p.is_file():
            files.append((p.name, p.stat().st_size))
        else:
            files.append((str(p)+"/", None))
    st.table({"name":[f[0] for f in files], "size(bytes)":[f[1] for f in files]})

# 3) poke_ability_clustering の import をテスト
with st.expander("poke_ability_clustering の import 結果", expanded=True):
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        pac = importlib.import_module("poke_ability_clustering")
        st.success("import 成功")
        st.write("エクスポート一覧（抜粋）:", [x for x in dir(pac) if not x.startswith("_")][:30])
    except Exception as e:
        st.error("import に失敗しました。ファイル名や構文、依存を確認してください。")
        st.code(traceback.format_exc())
        st.stop()

# 4) ability_clusters.joblib が読めるかテスト
with st.expander("ability_clusters.joblib の読込テスト", expanded=True):
    try:
        load_artifacts = getattr(pac, "load_artifacts", None)
        if not load_artifacts:
            st.error("load_artifacts 関数が見つかりません。poke_ability_clustering.py を確認してください。")
        else:
            bundle = load_artifacts("ability_clusters.joblib")
            keys = list(bundle.keys())
            st.success("読み込み成功")
            st.write("bundle keys:", keys[:10])
            st.write("feature_cols 数:", len(bundle.get("feature_cols", [])))
    except FileNotFoundError as e:
        st.error("ability_clusters.joblib が見つかりません。リポジトリ直下に追加してください。")
        st.exception(e)
    except Exception as e:
        st.error("joblib 読み込みで失敗しました（互換性や破損の可能性）。")
        st.exception(e)

st.info("診断が通れば、元の app.py に戻し、UI を表示してください。")
