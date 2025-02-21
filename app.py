import streamlit as st
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t as t_dist
import io
#import openpyxl  # 必要ならインポート

# =============================================================================
# 1) レイアウト
# =============================================================================
st.markdown("""
<h1 style="text-align:center;">流動分析ツール</h1>
<h3 style="text-align:center;">[グラビティ/小売引力/エントロピー最大化モデル]</h3>

<p style="text-align:right;">
  土居拓務（<strong>DOI, Takumu</strong>)
</p>

<hr style="border:1px solid #ccc;" />

<h2>概要</h2>
<p>
本アプリケーションでは、以下の3つの流動分析モデルを実装している。
</p>
<ol>
  <li><strong>グラビティモデル</strong><br />
      出発地の人口・到着地の人口・距離を用いた対数線形回帰モデル
  </li>
  <li><strong>小売引力モデル</strong><br />
      到着地の人口（売場規模など）と距離のみで流動を推定する簡易モデル
  </li>
  <li><strong>エントロピー最大化モデル</strong><br />
      シングリーコンストレインドで、出発地ごとの総フローを固定し、観測フローに最も近い分布を求める
  </li>
</ol>
<p>
分析を実施するにあたり、以下の列を記入した Excel ファイルをアップロードしてください。<br/>
<code>Origin</code>（出発地の名称）, 
<code>Destination</code>（到着地の名称）, 
<code>Distance</code>（出発地と到着地の距離）,
<code>Population_Origin</code>（出発地の人口）,
<code>Population_Destination</code>（到着地の人口）,
<code>Flow</code>（流動量）
</p>
""", unsafe_allow_html=True)

# =============================================================================
# 2) サンプルExcelダウンロード
# =============================================================================
def create_sample_excel():
    columns = [
        "Origin",
        "Destination",
        "Distance",
        "Population_Origin",
        "Population_Destination",
        "Flow"
    ]
    df_template = pd.DataFrame(columns=columns)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_template.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    label="列名記入済みExcelをダウンロード",
    data=create_sample_excel(),
    file_name="sample_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =============================================================================
# 3) ファイルアップロード
# =============================================================================
st.subheader("ファイルをアップロード")
uploaded_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])

# =============================================================================
# 4) モデル選択
# =============================================================================
model_choice = st.radio("分析モデルを選択", ("グラビティモデル", "小売引力モデル", "エントロピー最大化モデル"))

# =============================================================================
# 5) 欠損補完関数
# =============================================================================
def fix_data(df):
    numeric_cols = ["Distance", "Population_Origin", "Population_Destination", "Flow"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Distance"] <= 0, "Distance"] = np.nan
    df.loc[df["Flow"] < 0, "Flow"] = np.nan
    df.loc[df["Population_Origin"] < 1, "Population_Origin"] = np.nan
    df.loc[df["Population_Destination"] < 1, "Population_Destination"] = np.nan

    na_before = df.isna().sum().sum()
    df = df.interpolate(method='linear', axis=0)
    df = df.fillna(df.mean(numeric_only=True))
    na_after = df.isna().sum().sum()
    fixed_count = na_before - na_after
    return df, fixed_count

# =============================================================================
# 6) 列名バリデーション
# =============================================================================
def validate_columns(df):
    required_columns = [
        "Origin", "Destination", "Distance",
        "Population_Origin", "Population_Destination", "Flow"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"次の列が欠落しています: {', '.join(missing_columns)}")
        return False
    return True

# =============================================================================
# 7) エントロピー最大化モデル
# =============================================================================
def entropy_model_regression(df):
    origins = df["Origin"].unique()
    destinations = df["Destination"].unique()
    origin_flows = df.groupby("Origin")["Flow"].sum().to_dict()
    nobs = len(df)

    def objective(beta):
        sse = 0.0
        for o in origins:
            denom = 0.0
            for d in destinations:
                if ((df["Origin"] == o) & (df["Destination"] == d)).any():
                    dist = df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Distance"].values[0]
                    denom += math.exp(-beta * dist)
            for d in destinations:
                if ((df["Origin"] == o) & (df["Destination"] == d)).any():
                    dist = df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Distance"].values[0]
                    obs = df.loc[(df["Origin"] == o) & (df["Destination"] == d), "Flow"].values[0]
                    pred = (
                        origin_flows[o] * math.exp(-beta * dist) / denom
                        if denom != 0 else 0
                    )
                    sse += (obs - pred) ** 2
        return sse

    res = minimize(objective, x0=np.array([0.1]), method="BFGS")
    beta_opt = res.x[0]

    var_beta = np.nan
    if hasattr(res, "hess_inv"):
        hess_inv = res.hess_inv  
        if isinstance(hess_inv, np.ndarray):
            var_beta = hess_inv[0, 0]
        else:
            var_beta = hess_inv

    if var_beta is not None and var_beta > 0:
        se_beta = np.sqrt(var_beta)
    else:
        se_beta = np.nan

    df_resid = nobs - 1
    t_val = np.nan
    p_val = np.nan
    if not np.isnan(se_beta) and se_beta > 0:
        t_val = beta_opt / se_beta
        p_val = 2 * (1 - t_dist.cdf(abs(t_val), df_resid))

    return beta_opt, se_beta, t_val, p_val, nobs, df_resid, res

# =============================================================================
# 8) メイン処理
# =============================================================================
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if not validate_columns(df):
        st.stop()

    df_fixed, fixed_count = fix_data(df)
    if fixed_count > 0:
        st.warning(f"{int(fixed_count)} 個のデータに不備があったため調整しました（補完・平均埋めなど）。")

    # --- アップロード後のデータ表示 ---
    st.write("### アップロードされたデータ（補完後）")
    st.write(df_fixed)

    # ===============================
    # (A) モデル式を先に表示
    # ===============================
    st.markdown("### 予測されたモデル式")

    if model_choice == "グラビティモデル":
        # グラビティモデル
        st.latex(r'''
            \log(\text{Flow})
            = \beta_0
              + \beta_1 \,\log(\text{Distance})
              + \beta_2 \,\log(\text{Population\_Origin})
              + \beta_3 \,\log(\text{Population\_Destination})
            \quad\longrightarrow\quad
            \text{Flow}
            = \exp(\beta_0)\,\times
              \text{Distance}^{\beta_1}\,\times
              \text{Population\_Origin}^{\beta_2}\,\times
              \text{Population\_Destination}^{\beta_3}.
        ''')
    elif model_choice == "小売引力モデル":
        # 小売引力モデル
        st.latex(r'''
            \log(\text{Flow})
            = \beta_0
              + \beta_1 \,\log(\text{Population\_Destination})
              + \beta_2 \,\log(\text{Distance})
            \quad\longrightarrow\quad
            \text{Flow}
            = \exp(\beta_0)\,\times
              \text{Population\_Destination}^{\beta_1}\,\times
              \text{Distance}^{\beta_2}.
        ''')
    else:
        # エントロピー最大化モデル
        st.latex(r'''
            \text{FlowPred}(i \to j)
            = T_i\,\times\,
              \frac{\exp\bigl(-\beta \times \text{Distance}_{ij}\bigr)}
                   {\sum_{k}\exp\bigl(-\beta \times \text{Distance}_{ik}\bigr)}
        ''')
        st.markdown(r"ここで、\( T_i = \sum_{j}\text{Flow}(i \to j) \) である。")

    # -----------------------------------------------------------
    # (1) グラビティモデル (OLS)
    # -----------------------------------------------------------
    if model_choice == "グラビティモデル":
        st.subheader("グラビティモデルの結果")

        df_model = df_fixed[
            (df_fixed["Flow"] > 0)
            & (df_fixed["Distance"] > 0)
            & (df_fixed["Population_Origin"] > 0)
            & (df_fixed["Population_Destination"] > 0)
        ].copy()

        df_model["log_Flow"] = np.log(df_model["Flow"])
        df_model["log_Distance"] = np.log(df_model["Distance"])
        df_model["log_PopO"] = np.log(df_model["Population_Origin"])
        df_model["log_PopD"] = np.log(df_model["Population_Destination"])

        X = df_model[["log_Distance", "log_PopO", "log_PopD"]]
        X = sm.add_constant(X)
        y = df_model["log_Flow"]

        model = sm.OLS(y, X).fit()
        df_model["log_Flow_pred"] = model.predict(X)
        df_model["Flow_pred"] = np.exp(df_model["log_Flow_pred"])

        st.write(model.summary())

        # 評価指標
        residuals = df_model["Flow"] - df_model["Flow_pred"]
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(residuals))
        ss_tot = np.sum((df_model["Flow"] - np.mean(df_model["Flow"]))**2)
        ss_res = np.sum(residuals**2)
        r2 = 1 - (ss_res / ss_tot)

        st.markdown("#### 評価指標")
        st.write(f"- MSE:  {mse:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE:  {mae:.3f}")
        st.write(f"- R²:   {r2:.3f}")

        st.write("#### 予測結果")
        st.write(df_model[["Origin", "Destination", "Flow", "Flow_pred"]])

        # --- Excelダウンロード (in-memory 例) ---
        buffer = io.BytesIO()
        df_model.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="結果をダウンロード",
            data=buffer,
            file_name="gravity_model_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -----------------------------------------------------------
    # (2) 小売引力モデル (OLS)
    # -----------------------------------------------------------
    elif model_choice == "小売引力モデル":
        st.subheader("小売引力モデルの結果")

        df_model = df_fixed[
            (df_fixed["Flow"] > 0)
            & (df_fixed["Distance"] > 0)
            & (df_fixed["Population_Destination"] > 0)
        ].copy()

        df_model["log_Flow"] = np.log(df_model["Flow"])
        df_model["log_Distance"] = np.log(df_model["Distance"])
        df_model["log_PopD"] = np.log(df_model["Population_Destination"])

        X = df_model[["log_PopD", "log_Distance"]]
        X = sm.add_constant(X)
        y = df_model["log_Flow"]

        model = sm.OLS(y, X).fit()
        df_model["log_Flow_pred"] = model.predict(X)
        df_model["Flow_pred"] = np.exp(df_model["log_Flow_pred"])

        st.write(model.summary())

        residuals = df_model["Flow"] - df_model["Flow_pred"]
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(residuals))
        ss_tot = np.sum((df_model["Flow"] - np.mean(df_model["Flow"]))**2)
        ss_res = np.sum(residuals**2)
        r2 = 1 - (ss_res / ss_tot)

        st.markdown("#### 評価指標")
        st.write(f"- MSE:  {mse:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE:  {mae:.3f}")
        st.write(f"- R²:   {r2:.3f}")

        st.write("#### 予測結果")
        st.write(df_model[["Origin", "Destination", "Flow", "Flow_pred"]])

        buffer = io.BytesIO()
        df_model.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="結果をダウンロード",
            data=buffer,
            file_name="retail_gravity_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -----------------------------------------------------------
    # (3) エントロピー最大化モデル
    # -----------------------------------------------------------
    else:
        st.subheader("エントロピー最大化モデルの結果")

        df_model = df_fixed[
            (df_fixed["Flow"] > 0)
            & (df_fixed["Distance"] > 0)
        ].copy()

        beta_opt, se_beta, t_val, p_val, nobs, df_resid, res = entropy_model_regression(df_model)

        origins = df_model["Origin"].unique()
        destinations = df_model["Destination"].unique()
        origin_flows = df_model.groupby("Origin")["Flow"].sum().to_dict()

        def calculate_predicted_flow(row):
            o = row["Origin"]
            denom = sum(
                math.exp(-beta_opt * val)
                for val in df_model.loc[df_model["Origin"] == o, "Distance"]
            )
            return origin_flows[o] * math.exp(-beta_opt * row["Distance"]) / denom if denom!=0 else 0

        df_model["Flow_pred"] = df_model.apply(calculate_predicted_flow, axis=1)
        df_model["Residual"] = df_model["Flow"] - df_model["Flow_pred"]

        # 評価指標
        sse = np.sum(df_model["Residual"]**2)
        mse = np.mean(df_model["Residual"]**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(df_model["Residual"]))
        ss_tot = np.sum((df_model["Flow"] - np.mean(df_model["Flow"]))**2)
        r2 = 1 - (sse / ss_tot)

        # 近似的な回帰サマリ表示
        st.markdown("**詳細な結果（近似的な回帰サマリ）**")
        st.write("※ バージョン差異により最適化メソッドや反復回数は省略")

        st.markdown("---")
        st.markdown("**パラメータ推定値 (β)**")
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        col1.metric("beta", f"{beta_opt:.4f}")
        col2.metric("std err", f"{se_beta:.4f}" if not np.isnan(se_beta) else "NaN")
        col3.metric("t 値", f"{t_val:.3f}" if not np.isnan(t_val) else "NaN")
        col4.metric("p 値", f"{p_val:.3f}" if not np.isnan(p_val) else "NaN")

        st.markdown("---")
        st.markdown("**モデル全体の評価指標**")
        st.write(f"- No. Observations: {nobs}")
        st.write(f"- SSE: {sse:.3f}")
        st.write(f"- MSE: {mse:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE: {mae:.3f}")
        st.write(f"- R²:  {r2:.3f}")

        st.write("#### 予測結果")
        st.write(df_model[["Origin", "Destination", "Flow", "Flow_pred"]])

        buffer = io.BytesIO()
        df_model.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="結果をダウンロード",
            data=buffer,
            file_name="entropy_model_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# フッター
st.markdown("""
<hr />
<small>
  本ツール使用による成果物を公表する際は、以下の例のように引用していただけると嬉しいです。<br/>
  DOI, Takumu (2025). Flow Analysis Tool [Computer software]. Usage date: YYYY-MM-DD.
</small>
""", unsafe_allow_html=True)
