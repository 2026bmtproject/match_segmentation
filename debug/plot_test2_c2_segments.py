import sys
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


# 預設輸入檔
CSV_FILE = "test2_c2_2_segments.csv"

if len(sys.argv) > 1:
    CSV_FILE = sys.argv[1]


def find_threshold_gmm(scores):
    """
    用 Gaussian Mixture Model (3群) 自動找分界
    """
    log_scores = np.log10(np.maximum(scores, 1e-12))
    X = log_scores.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(X)

    means = np.asarray(gmm.means_, dtype=float).ravel()
    variances = np.asarray(gmm.covariances_, dtype=float).ravel()
    weights = np.asarray(gmm.weights_, dtype=float).ravel()

    order = np.argsort(means)
    # GMM=3 時，取低值群與中值群的交點作為主要分界
    m1, m2 = means[order][:2]
    v1, v2 = variances[order][:2]
    w1, w2 = weights[order][:2]

    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    a = 1 / (2 * v1) - 1 / (2 * v2)
    b = m2 / v2 - m1 / v1
    c = m1**2 / (2 * v1) - m2**2 / (2 * v2) + np.log((s2 * w1) / (s1 * w2))

    roots = np.roots([a, b, c])
    valid_roots = np.real(roots[(roots > m1) & (roots < m2)])

    if len(valid_roots) > 0:
        thresh_log = valid_roots[0]
    else:
        thresh_log = (m1 + m2) / 2

    return 10 ** thresh_log


def main():
    try:
        data = pd.read_csv(CSV_FILE)

        required_cols = [
            "Cross_Diff_Avg",
        ]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            print(f"Error: 缺少必要欄位: {', '.join(missing)}")
            return

        # 參考 plot_csv：針對 Cross_Diff_Avg 做 log 尺度直方圖統計
        scores = np.asarray(data["Cross_Diff_Avg"], dtype=float)
        log_scores = np.log10(np.maximum(scores, 1e-12))
        hist, edges = np.histogram(log_scores, bins=300)
        centers = (edges[:-1] + edges[1:]) / 2

        peaks, props = find_peaks(hist, prominence=30, distance=10)
        threshold_gmm = find_threshold_gmm(scores)

        threshold_valley = None
        if len(peaks) >= 2:
            prominences = props["prominences"]
            top2_idx = np.argsort(prominences)[-2:]
            p1, p2 = sorted(peaks[top2_idx])
            valley_idx = p1 + np.argmin(hist[p1:p2])
            threshold_valley = 10 ** centers[valley_idx]

        hist_fig = px.bar(
            x=centers,
            y=hist,
            labels={"x": "Log10(Cross_Diff_Avg)", "y": "Frequency"},
            title="Log-Scaled Cross_Diff_Avg Histogram",
        )

        if len(peaks) > 0:
            hist_fig.add_scatter(
                x=centers[peaks],
                y=hist[peaks],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Peaks",
            )

        hist_fig.add_vline(
            x=np.log10(threshold_gmm),
            line_dash="dash",
            line_color="green",
            annotation_text=f"GMM Threshold: {threshold_gmm:.2f}",
            annotation_position="top left",
        )

        if threshold_valley is not None:
            hist_fig.add_vline(
                x=np.log10(threshold_valley),
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Valley Threshold: {threshold_valley:.2f}",
                annotation_position="top right",
            )

        # 保留趨勢線，方便對照時間變化
        if "Start_Sec" in data.columns:
            line_fig = px.line(
                data,
                x="Start_Sec",
                y="Cross_Diff_Avg",
                title="Cross_Diff_Avg vs Start Time",
                labels={"Start_Sec": "Start Time (秒)", "Cross_Diff_Avg": "Cross_Diff_Avg"},
            )
            line_fig.update_traces(mode="lines+markers")
            line_fig.show()

        hist_fig.show()

    except FileNotFoundError:
        print(f"Error: 找不到檔案 {CSV_FILE}")
    except pd.errors.EmptyDataError:
        print(f"Error: 檔案 {CSV_FILE} 是空的或格式不正確")
    except Exception as e:
        print(f"Error: 發生未知錯誤 - {e}")


if __name__ == "__main__":
    main()
