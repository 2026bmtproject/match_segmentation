import pandas as pd
import plotly.express as px
import sys
import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

# 預設 CSV 檔案名稱
CSV_FILE = 'frame_diff_analysis2.csv'

if len(sys.argv) > 1:
    CSV_FILE = sys.argv[1]

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def find_threshold_gmm(scores):
    """
    用 Gaussian Mixture Model 自動找 rally / non-rally 分界
    """
    # log scale (和你原本方法一樣)
    log_scores = np.log10(np.maximum(scores, 1))

    # reshape for sklearn
    X = log_scores.reshape(-1, 1)

    # fit GMM (兩個高斯)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=0
    )

    gmm.fit(X)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # 排序 (確保第一個是rally)
    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]

    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    # 解兩個Gaussian相等的位置
    a = 1/(2*v1) - 1/(2*v2)
    b = m2/v2 - m1/v1
    c = m1**2/(2*v1) - m2**2/(2*v2) + np.log((s2*w1)/(s1*w2))

    roots = np.roots([a, b, c])

    # 取落在兩個mean之間的root
    thresh_log = np.real(roots[(roots > m1) & (roots < m2)])[0]

    threshold = 10 ** thresh_log

    return threshold

def main():
    try:
        # 讀取 CSV 檔案
        data = pd.read_csv(CSV_FILE)

        # 確保必要的欄位存在
        if 'Time_Sec' not in data.columns or 'Difference_Score' not in data.columns:
            print("Error: CSV 檔案中缺少必要的欄位 'Time_Sec' 或 'Difference_Score'")
            return

        # 固定 x 軸為秒
        x_data = 'Time_Sec'
        x_label = 'Time (秒)'

        # 計算 log 尺度的直方圖
        scores = data['Difference_Score'].values
        log_scores = np.log10(np.maximum(scores, 1))
        hist, edges = np.histogram(log_scores, bins=300)
        centers = (edges[:-1] + edges[1:]) / 2

        # 找主峰
        peaks, props = find_peaks(hist, prominence=30, distance=10)

        # 使用 GMM 找分界值
        threshold_gmm = find_threshold_gmm(scores)

        # 使用 prominences 找另一個分界值
        prominences = props['prominences']
        top2_idx = np.argsort(prominences)[-2:]
        p1, p2 = sorted(peaks[top2_idx])

        # 在兩峰之間找谷底
        valley_idx = p1 + np.argmin(hist[p1:p2])
        threshold_valley = 10 ** centers[valley_idx]

        # 使用 Plotly 繪製互動式圖表
        fig = px.line(data, x=x_data, y='Difference_Score', title=f'{x_label} vs Difference Score', labels={x_data: x_label, 'Difference_Score': 'Difference Score'})
        fig.update_layout(xaxis_title=x_label, yaxis_title='Difference Score')

        # 繪製直方圖
        hist_fig = px.bar(x=centers, y=hist, labels={'x': 'Log10(Difference Score)', 'y': 'Frequency'}, title='Log-Scaled Difference Score Histogram')
        hist_fig.add_scatter(x=centers[peaks], y=hist[peaks], mode='markers', marker=dict(color='red', size=10), name='Peaks')

        # 在直方圖中標記分界值
        hist_fig.add_vline(x=np.log10(threshold_gmm), line_dash="dash", line_color="green", annotation_text=f'GMM Threshold: {threshold_gmm:.2f}', annotation_position="top left")
        hist_fig.add_vline(x=np.log10(threshold_valley), line_dash="dot", line_color="blue", annotation_text=f'Valley Threshold: {threshold_valley:.2f}', annotation_position="top right")

        # 顯示互動式圖表
        fig.show()
        hist_fig.show()

    except FileNotFoundError:
        print(f"Error: 找不到檔案 {CSV_FILE}")
    except pd.errors.EmptyDataError:
        print(f"Error: 檔案 {CSV_FILE} 是空的或格式不正確")
    except Exception as e:
        print(f"Error: 發生未知錯誤 - {e}")

if __name__ == "__main__":
    main()