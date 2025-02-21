"""
グラビティ－エントロピーモデル［gravity-entropy-model］

概要
本アプリケーションは、グラビティモデル と エントロピー最大化モデル の流動分析を実装できるアプリである。
Excelデータをアップロードすることで分析を実行する。

製作者
土居拓務（DOI,Takumu）

機能
✅ Excelデータをアップロードし、流動分析を実施
✅ グラビティモデルとエントロピー最大化モデルのいずれかを選択
✅ 統計指標 (MSE, RMSE, MAE, R²) を算出
✅ 結果をExcelファイルとしてダウンロード可能

使用方法
1. Excelファイルをアップロード
   「ファイルをアップロード」ボタンをクリックし、Excelデータを選択
   フォーマットの例:
   ------------------------------
   Origin | Destination | Distance | Population_Origin | Population_Destination | Flow
   A      | X           | 10       | 1000              | 800                    | 80
   A      | Y           | 15       | 1000              | 1200                   | 60
   B      | Z           | 20       | 1500              | 1000                   | 50
   ------------------------------

2. 分析モデルを選択
   - グラビティモデル: 対数線形回帰を用いた流動予測
   - エントロピー最大化モデル: シングリーコンストレインドモデルによる最適化

3. 結果を確認
   - 結果を画面上に表示
   - 統計指標（MSE, RMSE, MAE, R²）を確認可能

4. Excelファイルをダウンロード
   - 解析結果をExcel形式 (.xlsx) でダウンロード可能
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from scipy.optimize import minimize
import io

# タイトル
st.title("Gravity Model & Entropy Maximization Model")

# =============================================================================
# サンプルExcelダウンロード用関数
# =============================================================================
def create_sample_excel():
    """
    必要な列名だけが入った空のDataFrameをExcelファイルに変換し、バイナリデータを返す。
    """
    columns = [
        "Origin",
        "Destination",
        "Distance",
        "Population_Origin",
        "Population_Destination",
        "Flow"
    ]
    df_template = pd.DataFrame(columns=columns)
    # Excel出力
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_template.to_excel(writer, index=False)
    return output.getvalue()

# サイドバーにサンプルExcelのダウンロードボタンを設置
st.sidebar.subheader("サンプルExcelファイル")
st.sidebar.download_button(
    label="サンプルExcelをダウンロード",
    data=create_sample_excel(),
    file_name="sample_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =============================================================================
# ファイルアップロード
# =============================================================================
st.sidebar.header("ファイルをアップロード")
uploaded_file = st.sidebar.file_uploader("Excelファイルをアップロード", type=["xlsx"])

# モデル選択
model_choice = st.sidebar.radio("分析モデルを選択", ("グラビティモデル", "エントロピー最大化モデル"))

# =============================================================================
# データのチェック関数
# =============================================================================
def validate_data(df):
    required_columns = ["Origin", "Destination", "Distance", 
                        "Population_Origin", "Population_Destination", "Flow"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"次の列が欠落しています: {', '.join(missing_columns)}")
        return False

    # データ型チェック
    if not np.issubdtype(df["Distance"].dtype, np.number):
        st.error("Distance 列は数値型である必要があります。")
        return False
    if not np.issubdtype(df["Population_Origin"].dtype, np.number):
        st.error("Population_Origin 列は数値型である必要があります。")
        return False
    if not np.issubdtype(df["Population_Destination"].dtype, np.number):
        st.error("Population_Destination 列は数値型である必要があります。")
        return False
    if not np.issubdtype(df["Flow"].dtype, np.number):
        st.error("Flow 列は数値型である必要があります。")
        return False

    # 値のチェック
    if (df["Distance"] <= 0).any():
        st.error("Distance 列には0より大きい値を入力してください。")
        return False
    if (df["Flow"] < 0).any():
        st.error("Flow 列には0以上の値を入力してください。")
        return False
    if (df["Population_Origin"] < 1).any():
        st.error("Population_Origin 列には1以上の値を入力してください。")
        return False
    if (df["Population_Destination"] < 1).any():
        st.error("Population_Destination 列には1以上の値を入力してください。")
        return False

    return True

# =============================================================================
# メイン処理
# =============================================================================
if uploaded_file:
    # アップロードデータを読み込み
    df = pd.read_excel(uploaded_file)

    # バリデーション
    if not validate_data(df):
        st.stop()  # データが不正なら処理を停止

    st.write("### アップロードされたデータ")
    st.write(df)

    # ------------------------------------------------------------------------------
    # グラビティモデル
    # ------------------------------------------------------------------------------
    if model_choice == "グラビティモデル":
        st.subheader("グラビティモデルの結果")

        # データの前処理
        df = df[(df["Flow"] > 0) & (df["Distance"] > 0) 
                & (df["Population_Origin"] > 0) & (df["Population_Destination"] > 0)].copy()
        df["log_Flow"] = np.log(df["Flow"])
        df["log_Distance"] = np.log(df["Distance"])
        df["log_PopO"] = np.log(df["Population_Origin"])
        df["log_PopD"] = np.log(df["Population_Destination"])

        # 回帰分析
        X = df[["log_Distance", "log_PopO", "log_PopD"]]
        X = sm.add_constant(X)  # 切片
        y = df["log_Flow"]

        model = sm.OLS(y, X).fit()
        df["log_Flow_pred"] = model.predict(X)
        df["Flow_pred"] = np.exp(df["log_Flow_pred"])

        # 結果表示: 回帰サマリ
        st.write(model.summary())

        # 評価指標計算
        residuals = df["Flow"] - df["Flow_pred"]
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(residuals))
        ss_tot = np.sum((df["Flow"] - np.mean(df["Flow"]))**2)
        ss_res = np.sum(residuals**2)
        r2 = 1 - (ss_res / ss_tot)

        st.markdown("#### 評価指標")
        st.write(f"- MSE:  {mse:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE:  {mae:.3f}")
        st.write(f"- R²:   {r2:.3f}")

        # 結果テーブル表示
        st.write("#### 予測結果")
        st.write(df[["Origin", "Destination", "Flow", "Flow_pred"]])

        # Excelで結果をダウンロード
        out_file = "gravity_model_result.xlsx"
        df.to_excel(out_file, index=False)
        with open(out_file, "rb") as f:
            st.download_button(
                label="結果をダウンロード",
                data=f,
                file_name=out_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ------------------------------------------------------------------------------
    # エントロピー最大化モデル
    # ------------------------------------------------------------------------------
    elif model_choice == "エントロピー最大化モデル":
        st.subheader("エントロピー最大化モデルの結果")

        # データの前処理
        df = df[(df["Flow"] > 0) & (df["Distance"] > 0)].copy()

        origins = df["Origin"].unique()
        destinations = df["Destination"].unique()
        origin_flows = df.groupby("Origin")["Flow"].sum().to_dict()

        # 目的関数 (MSEを最小化)
        def objective(beta):
            mse_list = []
            for o in origins:
                # 発地点 o の分母（全目的地に対する exp(-beta*distance)）
                denom = sum(
                    math.exp(-beta * df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Distance"].values[0])
                    for d in destinations if ((df["Origin"] == o) & (df["Destination"] == d)).any()
                )
                for d in destinations:
                    if ((df["Origin"] == o) & (df["Destination"] == d)).any():
                        dist_od = df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Distance"].values[0]
                        obs = df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Flow"].values[0]

                        pred = origin_flows[o] * math.exp(-beta * dist_od) / denom
                        mse_list.append((obs - pred)**2)
            return np.mean(mse_list)

        # beta の最適化
        try:
            res = minimize(objective, x0=0.1, method="Nelder-Mead")
            beta_opt = res.x[0]
        except Exception as e:
            st.error(f"最適化中にエラーが発生しました: {e}")
            st.stop()

        # 予測値を計算
        def calculate_predicted_flow(row):
            o = row["Origin"]
            # 発地点 o における分母
            denom = sum(
                math.exp(-beta_opt * dist_val)
                for dist_val in df.loc[df["Origin"] == o, "Distance"]
            )
            return origin_flows[o] * math.exp(-beta_opt * row["Distance"]) / denom if denom != 0 else 0

        df["Flow_pred"] = df.apply(calculate_predicted_flow, axis=1)
        df["Residual"] = df["Flow"] - df["Flow_pred"]

        # 統計指標の計算
        mse = np.mean(df["Residual"]**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(df["Residual"]))
        ss_tot = np.sum((df["Flow"] - np.mean(df["Flow"]))**2)
        ss_res = np.sum(df["Residual"]**2)
        r2 = 1 - (ss_res / ss_tot)

        st.write(f"**推定された beta:** {beta_opt:.4f}")

        st.markdown("#### 評価指標")
        st.write(f"- MSE:  {mse:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE:  {mae:.3f}")
        st.write(f"- R²:   {r2:.3f}")

        st.write("#### 予測結果")
        st.write(df[["Origin", "Destination", "Flow", "Flow_pred"]])

        # Excelで結果をダウンロード
        out_file = "entropy_model_result.xlsx"
        df.to_excel(out_file, index=False)
        with open(out_file, "rb") as f:
            st.download_button(
                label="結果をダウンロード",
                data=f,
                file_name=out_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
