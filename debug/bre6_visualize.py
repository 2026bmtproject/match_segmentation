"""簡化版 bre6 視覺化工具。

功能:
1) 時間序列折線圖（Difference Score）
2) 低於 threshold 的點標記
3) log histogram + peak 標記 + threshold 線

用法:
    python bre6_visualize.py test3_bre6.csv
    python bre6_visualize.py input.csv --method valley
    python bre6_visualize.py input.csv --threshold 2200 --save plot.html
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture


def sec_to_mmss(x: float) -> str:
    if np.isnan(x):
        return "00:00"
    x = max(0.0, float(x))
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes:02d}:{seconds:02d}"


def find_threshold_gmm(scores: np.ndarray) -> float:
    """用 Gaussian Mixture Model 自動找分界。"""
    log_scores = np.log10(np.maximum(scores.astype(float), 1.0))
    x = log_scores.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(x)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]

    v1 = max(v1, 1e-12)
    v2 = max(v2, 1e-12)
    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    a = 1.0 / (2.0 * v1) - 1.0 / (2.0 * v2)
    b = m2 / v2 - m1 / v1
    c = (m1**2) / (2.0 * v1) - (m2**2) / (2.0 * v2) + np.log((s2 * w1) / (s1 * w2))

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            thresh_log = float((m1 + m2) / 2.0)
            return float(10.0**thresh_log)
        thresh_log = float(-c / b)
        return float(10.0**thresh_log)

    roots = np.roots([a, b, c])
    real_roots = np.real(roots[np.isreal(roots)])
    between = real_roots[(real_roots > m1) & (real_roots < m2)]
    thresh_log = float(between[0]) if between.size > 0 else float((m1 + m2) / 2.0)
    return float(10.0**thresh_log)


def find_threshold_valley(scores: np.ndarray, bins: int = 300) -> float:
    """在 log histogram 的兩主峰中間找谷底。"""
    log_scores = np.log10(np.maximum(scores.astype(float), 1.0))
    hist, edges = np.histogram(log_scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # 自動調整 prominence，避免資料集大小差異太大時抓不到峰
    prom = max(5.0, float(hist.max()) * 0.03)
    peaks, props = find_peaks(hist, prominence=prom, distance=max(5, bins // 40))

    if peaks.size < 2:
        return find_threshold_gmm(scores)

    prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
    top2_idx = np.argsort(prominences)[-2:]
    p1, p2 = sorted(peaks[top2_idx])
    if p2 <= p1 + 1:
        return find_threshold_gmm(scores)

    valley_idx = p1 + int(np.argmin(hist[p1:p2]))
    return float(10.0 ** centers[valley_idx])


def resolve_threshold(df: pd.DataFrame, method: str, manual: float | None) -> float:
    scores = df["Difference_Score"].astype(float).to_numpy()
    if manual is not None:
        return float(manual)
    if method == "csv":
        if "Threshold" not in df.columns:
            raise ValueError("CSV 中沒有 Threshold 欄位，請改用 --method gmm/valley 或 --threshold")
        return float(df["Threshold"].iloc[0])
    if method == "gmm":
        return find_threshold_gmm(scores)
    return find_threshold_valley(scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="視覺化 bre6 CSV")
    parser.add_argument("csv_path", help="bre6 產生的 CSV 路徑")
    parser.add_argument("--save", default="", help="輸出圖檔路徑（建議 .html）")
    parser.add_argument("--no-show", action="store_true", help="只輸出檔案，不開互動視窗")
    parser.add_argument("--threshold", type=float, default=None, help="手動指定 threshold（優先）")
    parser.add_argument(
        "--method",
        choices=["csv", "gmm", "valley"],
        default="csv",
        help="threshold 來源：csv=讀取欄位、gmm=自動混合高斯、valley=雙峰谷底",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    required_cols = {"Time_Sec", "Difference_Score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位: {sorted(missing)}")

    times = df["Time_Sec"].astype(float).to_numpy()
    scores = df["Difference_Score"].astype(float).to_numpy()
    threshold = resolve_threshold(df, args.method, args.threshold)
    is_low = scores < threshold

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=times,
            y=scores,
            mode="lines",
            name="Difference_Score",
            line=dict(color="#3567B8", width=1.2),
            customdata=[sec_to_mmss(t) for t in times],
            hovertemplate="時間: %{customdata}<br>分數: %{y:,.0f}<extra></extra>",
        )
    )

    if np.any(is_low):
        fig.add_trace(
            go.Scattergl(
                x=times[is_low],
                y=scores[is_low],
                mode="markers",
                name="Low Points",
                marker=dict(color="#1E8E3E", size=5, opacity=0.65),
                customdata=[sec_to_mmss(t) for t in times[is_low]],
                hovertemplate="時間: %{customdata}<br>分數: %{y:,.0f}<br>狀態: Low<extra></extra>",
            )
        )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#D94E41",
        line_width=2,
        annotation_text=f"Threshold: {threshold:,.1f}",
        annotation_position="top left",
    )

    total_sec = float(times[-1]) if times.size else 0.0
    low_ratio = float(is_low.mean() * 100.0) if is_low.size else 0.0
    fig.update_layout(
        title=(
            "Frame Difference Visualization"
            f"<br><sup>Duration: {total_sec/60:.1f} min | Low ratio: {low_ratio:.1f}% | Method: {args.method}</sup>"
        ),
        template="plotly_white",
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=60, r=25, t=90, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    tick_step = 30.0
    tick_vals = np.arange(0.0, (float(times[-1]) if times.size else 0.0) + tick_step, tick_step)
    fig.update_xaxes(
        title_text="Time (mm:ss)",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=[sec_to_mmss(v) for v in tick_vals],
        rangeslider=dict(visible=True),
        showgrid=True,
    )
    fig.update_yaxes(title_text="Difference Score", showgrid=True)

    if args.save:
        fig.write_html(args.save, include_plotlyjs="cdn")
        print(f"已儲存圖檔: {args.save}")

    if not args.no_show:
        fig.show(config={"scrollZoom": True, "displaylogo": False})


if __name__ == "__main__":
    main()
