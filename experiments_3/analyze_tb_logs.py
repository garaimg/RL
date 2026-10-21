"""
Análisis de los logs de TensorBoard de los experimentos 3.x.

Para cada entorno lee logs/<entorno>/<config>/<algo>_run<i>_*/events.*, interpola la curva
train/reward de cada ejecución sobre una rejilla común de pasos y calcula, por (config, algoritmo):
  - final_reward: recompensa media de entrenamiento en el último 10% de los pasos
  - auc_reward:   recompensa media durante todo el entrenamiento (velocidad + calidad)
  - steps_to_thr: primer paso en que la curva media supera un umbral (None si nunca lo hace)
  - final_len:    longitud media de episodio en el último 10%
Genera también una figura por entorno (results/<entorno>_curves.png) con un panel por configuración.

Uso: python analyze_tb_logs.py [entorno ...]
"""
import os
import sys
import glob
import csv
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Umbral de recompensa de entrenamiento (media de 100 episodios) para medir la velocidad de aprendizaje
THRESHOLDS = {"taxi": 0.0, "cliffwalking": -30.0, "frozenlake16_shaped": 1.5, "frozenlake16_sparse": 0.5}

ALGO_COLORS = {"sarsa": "#2a78d6", "qlearning": "#eb6834", "esarsa": "#1baf7a"}
ALGO_NAMES = {"sarsa": "SARSA", "qlearning": "Q-Learning", "esarsa": "Expected SARSA"}
CONFIG_ORDER = ["baseline", "lr=0.01", "lr=0.5", "lr=0.5_decay", "dr=0.9", "dr=1.0", "eps=0.01", "eps=0.3", "eps_decay"]


def load_scalar(run_dir, tag):
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None, None
    events = acc.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def load_run(run_dir):
    config = os.path.basename(os.path.dirname(run_dir))
    algo = os.path.basename(run_dir).split("_run")[0]
    steps, rewards = load_scalar(run_dir, "train/reward")
    len_steps, lens = load_scalar(run_dir, "train/episode_len")
    return config, algo, steps, rewards, len_steps, lens


def load_env(env_name, n_points=200):
    """Devuelve {(config, algo): {"grid":..., "reward": array(runs, n_points), "len": ...}}"""
    env_dir = os.path.join(BASE_DIR, "logs", env_name)
    data = defaultdict(lambda: {"reward": [], "len": []})
    max_step = 0
    with Pool(os.cpu_count()) as pool:  # leer los ficheros de eventos es lento: se hace en paralelo
        raw = [r for r in pool.map(load_run, sorted(glob.glob(os.path.join(env_dir, "*", "*_run*"))))
               if r[2] is not None and len(r[2]) > 0]
    for r in raw:
        max_step = max(max_step, r[2][-1])

    grid = np.linspace(0, max_step, n_points)
    for config, algo, steps, rewards, len_steps, lens in raw:
        data[(config, algo)]["reward"].append(np.interp(grid, steps, rewards))
        data[(config, algo)]["len"].append(np.interp(grid, len_steps, lens))
    for value in data.values():
        value["reward"] = np.array(value["reward"])
        value["len"] = np.array(value["len"])
        value["grid"] = grid
    return dict(data)


def summarize(env_name, data):
    threshold = THRESHOLDS.get(env_name)
    rows = []
    for (config, algo), d in data.items():
        mean_curve = d["reward"].mean(axis=0)
        last = d["grid"] >= 0.9 * d["grid"][-1]
        above = np.nonzero(mean_curve >= threshold)[0] if threshold is not None else []
        rows.append({"config": config, "algo": algo, "n_runs": len(d["reward"]),
                     "final_reward": mean_curve[last].mean(),
                     "final_reward_std": d["reward"][:, last].mean(axis=1).std(),
                     "auc_reward": mean_curve.mean(),
                     "steps_to_thr": int(d["grid"][above[0]]) if len(above) else None,
                     "final_len": d["len"].mean(axis=0)[last].mean()})
    rows.sort(key=lambda r: (CONFIG_ORDER.index(r["config"]) if r["config"] in CONFIG_ORDER else 99, r["algo"]))

    out_file = os.path.join(BASE_DIR, "results", f"{env_name}_tensorboard.csv")
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {env_name}: curvas de entrenamiento (TensorBoard, train/reward), umbral={threshold} ===")
    print(f"{'config':<14} {'algo':<10} {'final':>9} {'±':>7} {'AUC':>9} {'pasos->umbral':>14} {'len final':>10}")
    for r in rows:
        thr = f"{r['steps_to_thr']:,}" if r["steps_to_thr"] is not None else "nunca"
        print(f"{r['config']:<14} {r['algo']:<10} {r['final_reward']:9.2f} {r['final_reward_std']:7.2f} "
              f"{r['auc_reward']:9.2f} {thr:>14} {r['final_len']:10.1f}")
    return rows


def plot(env_name, data):
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in data)]
    n_cols = 3
    n_rows = int(np.ceil(len(configs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), sharex=True, sharey=True, squeeze=False)

    # Eje y común recortado a percentiles para que las primeras fases (muy negativas) no aplasten las curvas
    all_means = np.concatenate([d["reward"].mean(axis=0) for d in data.values()])
    y_low, y_high = np.percentile(all_means, 5), np.max(all_means)
    margin = 0.05 * (y_high - y_low)
    threshold = THRESHOLDS.get(env_name)

    for ax, config in zip(axes.flat, configs):
        for algo in ALGO_COLORS:
            if (config, algo) not in data:
                continue
            d = data[(config, algo)]
            mean, std = d["reward"].mean(axis=0), d["reward"].std(axis=0)
            ax.fill_between(d["grid"], mean - std, mean + std, color=ALGO_COLORS[algo], alpha=0.15, linewidth=0)
            ax.plot(d["grid"], mean, color=ALGO_COLORS[algo], linewidth=2, label=ALGO_NAMES[algo])
        if threshold is not None:
            ax.axhline(threshold, color="#9a9994", linewidth=1, linestyle="--")
        ax.set_title(config, fontsize=11, color="#0b0b0b")
        ax.set_ylim(y_low - margin, y_high + margin)
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.tick_params(colors="#52514e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#d4d3cd")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    for ax in list(axes.flat)[len(configs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("pasos de entrenamiento", color="#52514e", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("train/reward (media 100 ep.)", color="#52514e", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"{env_name}: recompensa de entrenamiento (media ± std de las ejecuciones)", fontsize=12)
    fig.tight_layout()
    out_file = os.path.join(BASE_DIR, "results", f"{env_name}_curves.png")
    fig.savefig(out_file, dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Figura guardada en {out_file}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    env_names = sys.argv[1:] or sorted(os.listdir(os.path.join(BASE_DIR, "logs")))
    for env_name in env_names:
        data = load_env(env_name)
        if not data:
            print(f"No hay logs para {env_name}")
            continue
        summarize(env_name, data)
        plot(env_name, data)
