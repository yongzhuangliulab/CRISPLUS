import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# =========================
# 1. 基本配置
# =========================

METHODS = [
    {
        "name": "CRISPLUS",
        "root": Path("../results/OP3_complete"),
        "run_prefix": "OP3_complete",
        "color": "#d95f4a",
    },
    {
        "name": "CRISP",
        "root": Path("../results/OP3_complete_CRISP"),
        "run_prefix": "OP3_complete",
        "color": "#4c78a8",
    },
    {
        "name": "CRISPLUSv0.1",
        "root": Path("../results/OP3_complete_CRISPLUSv0.1"),
        "run_prefix": "OP3_complete",
        "color": "#54a24b",
    },
    {
        "name": "CRISPLUSv0.2",
        "root": Path("../results/OP3_complete_CRISPLUSv0.2"),
        "run_prefix": "OP3_complete",
        "color": "#b279a2",
    },
]

METHOD_ORDER = [m["name"] for m in METHODS]
METHOD_COLOR = {m["name"]: m["color"] for m in METHODS}

# 参考方法：你希望证明它强于所有其他方法
REFERENCE_METHOD = "CRISPLUS"

# 是否显示 REFERENCE_METHOD 优于其他方法的 p-value
SHOW_PVALUE = True

# 你想画的 OOD 指标
METRICS = [
    "pearson_delta_de",
    "r2score_de",
    "sinkhorn_de",
]

METRIC_TITLES = {
    "pearson_delta_de": r"Pearson $\Delta$ top 50 DE genes (↑)",
    "r2score_de": r"R$^2$ top 50 DE genes (↑)",
    "sinkhorn_de": r"Sinkhorn top 50 DE genes (↓)",
}

# 指标方向：
# higher: 越大越好，检验 REFERENCE_METHOD > other_method
# lower : 越小越好，检验 REFERENCE_METHOD < other_method
METRIC_BETTER_DIRECTION = {
    "pearson_delta_de": "higher",
    "r2score_de": "higher",
    "sinkhorn_de": "lower",
}

FIGURE_DIR = Path("../figures/OP3_complete")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FIGURE_DIR / "ood_metrics_multi_methods_long.csv"
OUT_PNG = FIGURE_DIR / "ood_metrics_multi_methods_boxplot.png"
OUT_JPG = FIGURE_DIR / "ood_metrics_multi_methods_boxplot.jpg"
OUT_PDF = FIGURE_DIR / "ood_metrics_multi_methods_boxplot.pdf"


# =========================
# 2. 从 log 中提取最后一次 evaluation_stats
# =========================

def extract_last_eval_stats_from_log(log_path: Path):
    """
    从 log.txt 中提取最后一次出现的 evaluation_stats 字典。
    返回:
        {
            "iid": {...},
            "ood": {...}
        }
    """
    text = log_path.read_text(errors="ignore")

    key = "'evaluation_stats'"
    positions = [m.start() for m in re.finditer(re.escape(key), text)]

    if not positions:
        print(f"[Warning] No evaluation_stats found in {log_path}")
        return None

    # 取最后一次 evaluation_stats
    pos = positions[-1]

    # 找到包含 evaluation_stats 的最外层字典开头
    start = text.rfind("{", 0, pos)
    if start == -1:
        print(f"[Warning] Cannot find dict start in {log_path}")
        return None

    # 花括号配对，找到这个 dict 的结尾
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        print(f"[Warning] Cannot find dict end in {log_path}")
        return None

    block = text[start:end]

    try:
        parsed = ast.literal_eval(block)
    except Exception as e:
        print(f"[Warning] Failed to parse {log_path}: {e}")
        return None

    return parsed.get("evaluation_stats", None)


def parse_split_seed_from_dirname(dirname: str, run_prefix: str):
    """
    解析目录名:
        OP3_complete_split1_1327

    得到:
        split1, 1327

    其中:
        run_prefix = OP3_complete
    """
    pattern = rf"{re.escape(run_prefix)}_(split\d+)_(\d+)"
    m = re.match(pattern, dirname)

    if m is None:
        return None, None

    return m.group(1), int(m.group(2))


def collect_ood_metrics_for_method(method_cfg):
    rows = []

    method_name = method_cfg["name"]
    result_root = method_cfg["root"]
    run_prefix = method_cfg["run_prefix"]

    # 如果方法还没填目录，跳过
    if str(result_root) == "." or str(result_root) == "" or run_prefix == "":
        print(
            f"[Warning] Method {method_name} has empty root or run_prefix. "
            f"Please fill them before collecting this method."
        )
        return pd.DataFrame()

    log_paths = sorted(result_root.glob(f"{run_prefix}_split*_*/log.txt"))

    if len(log_paths) == 0:
        print(f"[Warning] No log.txt found for method={method_name} under {result_root}")
        return pd.DataFrame()

    for log_path in log_paths:
        run_dir = log_path.parent.name
        split, seed = parse_split_seed_from_dirname(run_dir, run_prefix)

        if split is None:
            print(f"[Warning] Skip unexpected directory name: {run_dir}")
            continue

        eval_stats = extract_last_eval_stats_from_log(log_path)
        if eval_stats is None:
            continue

        if "ood" not in eval_stats:
            print(f"[Warning] No ood evaluation in {log_path}")
            continue

        ood_stats = eval_stats["ood"]

        for metric, value in ood_stats.items():
            rows.append({
                "method": method_name,
                "split": split,
                "seed": seed,
                "run": f"{split}_{seed}",
                "setting": "ood",
                "metric": metric,
                "value": float(value),
                "log_path": str(log_path),
            })

    return pd.DataFrame(rows)


def collect_all_methods_ood_metrics(methods):
    dfs = []

    for method_cfg in methods:
        df_m = collect_ood_metrics_for_method(method_cfg)
        if not df_m.empty:
            dfs.append(df_m)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =========================
# 3. One-tail paired t-test:
#    检验 REFERENCE_METHOD 是否优于其他方法
# =========================

def format_pvalue(p):
    if p is None or np.isnan(p):
        return "NA"

    if p < 1e-4:
        return "****"

    return f"{p:.4f}"


def compute_reference_better_pvalue(df_metric, other_method, reference_method, metric):
    """
    One-tail paired t-test.

    这里 reference_method 是你希望证明更强的方法，例如 CRISPLUS。

    对相同 run 的结果做配对检验：

    higher-better 指标:
        H0: reference_method <= other_method
        H1: reference_method >  other_method

    lower-better 指标:
        H0: reference_method >= other_method
        H1: reference_method <  other_method

    返回值:
        p-value, test description
    """
    ref = (
        df_metric[df_metric["method"] == reference_method][["run", "value"]]
        .dropna()
        .copy()
    )
    other = (
        df_metric[df_metric["method"] == other_method][["run", "value"]]
        .dropna()
        .copy()
    )

    if ref.empty or other.empty:
        return np.nan, "NA"

    merged = ref.merge(
        other,
        on="run",
        suffixes=("_reference", "_other"),
    )

    if len(merged) < 2:
        print(
            f"[Warning] Not enough paired runs for {reference_method} better than {other_method} "
            f"on metric={metric}. Only {len(merged)} paired runs found."
        )
        return np.nan, "not_enough_pairs"

    direction = METRIC_BETTER_DIRECTION.get(metric, "higher")

    if direction == "higher":
        # 检验 reference_method > other_method
        alternative = "greater"
    elif direction == "lower":
        # 检验 reference_method < other_method
        alternative = "less"
    else:
        raise ValueError(f"Unknown direction for metric={metric}: {direction}")

    try:
        stat, p = ttest_rel(
            merged["value_reference"].values,
            merged["value_other"].values,
            alternative=alternative,
        )
        return p, f"one_tail_paired_ttest_{reference_method}_better_than_{other_method}_{alternative}"
    except Exception as e:
        print(
            f"[Warning] paired t-test failed for {reference_method} better than {other_method} "
            f"on metric={metric}: {e}"
        )
        return np.nan, "failed"


# =========================
# 4. 画图
# =========================

def plot_ood_boxplot_multi_methods(df: pd.DataFrame):
    df_plot = df[df["metric"].isin(METRICS)].copy()

    if df_plot.empty:
        raise ValueError("No data available for selected METRICS.")

    methods_present = [
        method for method in METHOD_ORDER
        if method in set(df_plot["method"])
    ]

    if len(methods_present) == 0:
        raise ValueError("No configured methods were found in df.")

    fig_height = max(2.8, 0.55 * len(methods_present) + 1.0)
    fig_width = 4.2 * len(METRICS)

    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(fig_width, fig_height),
        sharey=True,
    )

    if len(METRICS) == 1:
        axes = [axes]

    y_positions = np.arange(len(methods_present))[::-1]
    method_to_y = dict(zip(methods_present, y_positions))

    rng = np.random.default_rng(0)

    for ax, metric in zip(axes, METRICS):
        sub = df_plot[df_plot["metric"] == metric].copy()

        # 背景条纹
        for i, method in enumerate(methods_present):
            y = method_to_y[method]
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#eee6d8", zorder=0)
            else:
                ax.axhspan(y - 0.5, y + 0.5, color="#f7f2ea", zorder=0)

        all_values = sub["value"].dropna().values
        if len(all_values) == 0:
            continue

        x_min = np.nanmin(all_values)
        x_max = np.nanmax(all_values)
        x_range = max(x_max - x_min, 1e-8)

        # 画每个方法
        for method in methods_present:
            values = sub.loc[sub["method"] == method, "value"].dropna().values
            y = method_to_y[method]

            if len(values) == 0:
                continue

            color = METHOD_COLOR.get(method, "#cccccc")

            ax.boxplot(
                values,
                positions=[y],
                vert=False,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.9,
                ),
                medianprops=dict(
                    color="black",
                    linewidth=1.2,
                ),
                whiskerprops=dict(
                    color="black",
                    linewidth=0.8,
                ),
                capprops=dict(
                    color="black",
                    linewidth=0.8,
                ),
                whis = (0, 100),
            )

            # 叠加每个 split+seed 的点
            jitter = rng.normal(0, 0.04, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), y) + jitter,
                marker="x",
                s=18,
                linewidths=0.7,
                color="black",
                alpha=0.85,
                zorder=3,
            )

        # p-value 标注：
        # 标在其他方法行上，含义是 REFERENCE_METHOD 是否显著优于该行方法
        if SHOW_PVALUE and REFERENCE_METHOD in methods_present:
            text_x = x_max + 0.08 * x_range

            for method in methods_present:
                if method == REFERENCE_METHOD:
                    continue

                y = method_to_y[method]

                p, test_type = compute_reference_better_pvalue(
                    sub,
                    other_method=method,
                    reference_method=REFERENCE_METHOD,
                    metric=metric,
                )

                txt = format_pvalue(p)

                ax.text(
                    text_x,
                    y,
                    txt,
                    color="#d62728",
                    fontsize=11,
                    va="center",
                    ha="left",
                    fontweight="bold" if txt == "****" else "normal",
                )

            ax.set_xlim(x_min - 0.08 * x_range, x_max + 0.35 * x_range)
        else:
            ax.set_xlim(x_min - 0.08 * x_range, x_max + 0.08 * x_range)

        ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=12)

        ax.grid(axis="x", linestyle="-", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(methods_present, fontsize=11)

    fig.text(
        0.02,
        0.5,
        "OP3_complete unseen cell type",
        rotation=90,
        va="center",
        fontsize=13,
    )

    plt.tight_layout(rect=[0.06, 0.03, 1, 0.95])

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_JPG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")

    plt.show()


# =========================
# 5. 主程序
# =========================

if __name__ == "__main__":
    df = collect_all_methods_ood_metrics(METHODS)

    if df.empty:
        raise RuntimeError(
            "No metrics were collected. "
            "Please check METHODS config, result directories, run_prefix, and log files."
        )

    df.to_csv(OUT_CSV, index=False)

    print(f"[Info] Collected {len(df)} metric records.")
    print(f"[Info] Number of methods: {df['method'].nunique()}")
    print(f"[Info] Methods: {sorted(df['method'].unique())}")

    print("\n[Info] Number of runs per method:")
    print(df.groupby("method")["run"].nunique())

    print("\n[Info] OOD metrics summary:")
    summary = (
        df[df["metric"].isin(METRICS)]
        .groupby(["method", "metric"])["value"]
        .agg(["count", "mean", "std", "min", "max"])
    )
    print(summary)

    if SHOW_PVALUE and REFERENCE_METHOD in set(df["method"]):
        print(
            f"\n[Info] One-tail paired t-test p-values: "
            f"testing whether {REFERENCE_METHOD} is better than each other method."
        )

        for metric in METRICS:
            direction = METRIC_BETTER_DIRECTION.get(metric, "higher")
            if direction == "higher":
                hypothesis_text = f"{REFERENCE_METHOD} > other_method"
            elif direction == "lower":
                hypothesis_text = f"{REFERENCE_METHOD} < other_method"
            else:
                hypothesis_text = "unknown"

            sub = df[df["metric"] == metric]
            print(f"\nMetric: {metric}")
            print(f"  Alternative hypothesis: {hypothesis_text}")

            for method in METHOD_ORDER:
                if method == REFERENCE_METHOD:
                    continue

                if method not in set(sub["method"]):
                    continue

                p, test_type = compute_reference_better_pvalue(
                    sub,
                    other_method=method,
                    reference_method=REFERENCE_METHOD,
                    metric=metric,
                )

                print(
                    f"  {REFERENCE_METHOD} better than {method}: "
                    f"p={p:.6g}, test={test_type}"
                )

    plot_ood_boxplot_multi_methods(df)

    print(f"\n[Info] Saved CSV: {OUT_CSV}")
    print(f"[Info] Saved PNG: {OUT_PNG}")
    print(f"[Info] Saved JPG: {OUT_JPG}")
    print(f"[Info] Saved PDF: {OUT_PDF}")