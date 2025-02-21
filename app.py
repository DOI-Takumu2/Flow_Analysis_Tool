import streamlit as st
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from scipy.optimize import minimize

# タイトル
st.title("Gravity Model & Entropy Maximization Model")

# ファイルアップロード
st.sidebar.header("ファイルをアップロード")
uploaded_file = st.sidebar.file_uploader("Excelファイルをアップロード", type=["xlsx"])

# モデル選択
model_choice = st.sidebar.radio("分析モデルを選択", ("グラビティモデル", "エントロピー最大化モデル"))

# データ読み込み
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### アップロードされたデータ")
    st.write(df)

    if model_choice == "グラビティモデル":
        st.subheader("グラビティモデルの結果")

        # データの前処理
        df = df[(df["Flow"] > 0) & (df["Distance"] > 0) & (df["Population_Origin"] > 0) & (df["Population_Destination"] > 0)].copy()
        df["log_Flow"] = np.log(df["Flow"])
        df["log_Distance"] = np.log(df["Distance"])
        df["log_PopO"] = np.log(df["Population_Origin"])
        df["log_PopD"] = np.log(df["Population_Destination"])

        # 回帰分析
        X = df[["log_Distance", "log_PopO", "log_PopD"]]
        X = sm.add_constant(X)
        y = df["log_Flow"]

        model = sm.OLS(y, X).fit()
        df["log_Flow_pred"] = model.predict(X)
        df["Flow_pred"] = np.exp(df["log_Flow_pred"])

        # 結果表示
        st.write(model.summary())
        st.write(df[["Origin", "Destination", "Flow", "Flow_pred"]])

        # Excelファイルで結果をダウンロード
        output_file = "gravity_model_results.xlsx"
        df.to_excel(output_file, index=False)
        st.download_button("結果をダウンロード", data=open(output_file, "rb"), file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif model_choice == "エントロピー最大化モデル":
        st.subheader("エントロピー最大化モデルの結果")

        # データの前処理
        df = df[(df["Flow"] > 0) & (df["Distance"] > 0)].copy()
        origins = df["Origin"].unique()
        destinations = df["Destination"].unique()
        origin_flows = df.groupby("Origin")["Flow"].sum().to_dict()

        # 最適化
        def objective(beta):
            mse_list = []
            for o in origins:
                denom = sum(math.exp(-beta * df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Distance"].values[0])
                            for d in destinations if ((df["Origin"] == o) & (df["Destination"] == d)).any())
                for d in destinations:
                    if ((df["Origin"] == o) & (df["Destination"] == d)).any():
                        pred = origin_flows[o] * math.exp(-beta * df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Distance"].values[0]) / denom
                        obs = df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Flow"].values[0]
                        mse_list.append((obs - pred)**2)
            return np.mean(mse_list)

        res = minimize(objective, x0=0.1, method="Nelder-Mead")
        beta_opt = res.x[0]

        # 予測値の計算
        df["Flow_pred"] = df.apply(lambda row: origin_flows[row["Origin"]] * math.exp(-beta_opt * row["Distance"]) /
                                   sum(math.exp(-beta_opt * df.loc[df["Origin"] == row["Origin"], "Distance"])), axis=1)

        st.write(f"推定された beta: {beta_opt:.4f}")
        st.write(df[["Origin", "Destination", "Flow", "Flow_pred"]])

        # Excelファイルで結果をダウンロード
        output_file = "entropy_model_results.xlsx"
        df.to_excel(output_file, index=False)
        st.download_button("結果をダウンロード", data=open(output_file, "rb"), file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

