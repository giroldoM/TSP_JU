# tuner.py
import time
import random
import itertools as it
import statistics as stats
import pandas as pd
import numpy as np
import io, contextlib
import importlib
import main  

# =================== CONFIG ===================
TRIALS_PER_COMBO = 10
GA_VERBOSE = False
BASE_SEED = 1234

# grade de parâmetros
GRID = {
    "mutation_rate":  [0.01, 0.03, 0.05, 0.08, 0.10],
    "num_paths":      [100, 250, 400],
    "num_generations":[500, 1000, 2000],
    "num_best":       [2, 4],
    "k_tournament":   [3, 5, 7],
}

# Se você souber o ótimo da instância (Berlin52 = 7542), ative:
OPT_KNOWN = True
OPT_VALUE = 7542
# sucesso = distância <= OPT_VALUE*(1+EPS)
SUCCESS_EPS = 0.2  # 2% do ótimo

# arquivos de saída
TRIALS_CSV  = "tuning_trials.csv"
SUMMARY_CSV = "tuning_summary.csv"

# Estratégia para escolher "melhor combinação"
# Opções: "min_mean", "mean+lambda_std", "ucb", "median_q75", "pareto", "custom"
BEST_STRATEGY = "mean+lambda_std"
LAMBDA_STD = 0.50  
UCB_Z = 1.96       
TIME_WEIGHT = 0.0  
# ==============================================


def get_distance_matrix():
    importlib.reload(main)
    return main.df.values.tolist()

def run_single_trial(distance_matrix, params, seed):
    random.seed(seed)

    suppress = contextlib.nullcontext() if GA_VERBOSE else contextlib.redirect_stdout(io.StringIO())
    t0 = time.time()
    with suppress:
        best_path, best_distance = main.genetic_algorithm_tsp(
            distance_matrix=distance_matrix,
            num_paths=params["num_paths"],
            num_generations=params["num_generations"],
            mutation_rate=params["mutation_rate"],
            num_best=params["num_best"],
            k_tournament=params["k_tournament"],
        )
    dt = time.time() - t0

    return {
        **params,
        "seed": seed,
        "best_distance": float(best_distance),
        "best_path": best_path,
        "time_sec": dt,
    }

def all_param_combos(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [{k:v[i] for i,k in enumerate(keys)} for v in it.product(*vals)]

def summarize_trials(df):

    param_cols = [c for c in df.columns if c not in ("seed","best_distance","best_path","time_sec")]


    idx_min = df.groupby(param_cols)["best_distance"].idxmin()
    best_rows = df.loc[idx_min, param_cols + ["best_distance","best_path"]].rename(
        columns={"best_distance":"best_distance_min", "best_path":"best_path_of_best"}
    )

   
    def q(series, p): 
        return float(np.quantile(series.to_numpy(), p))

    agg = df.groupby(param_cols).agg(
        trials            = ("best_distance","count"),
        best_distance_mean=("best_distance","mean"),
        best_distance_std = ("best_distance","std"),
        best_distance_med = ("best_distance","median"),
        q10               = ("best_distance", lambda s: q(s,0.10)),
        q25               = ("best_distance", lambda s: q(s,0.25)),
        q75               = ("best_distance", lambda s: q(s,0.75)),
        q90               = ("best_distance", lambda s: q(s,0.90)),
        time_sec_mean     = ("time_sec","mean"),
    ).reset_index()


    if OPT_KNOWN:
        thr = OPT_VALUE * (1.0 + SUCCESS_EPS)
        succ = df.assign(success = df["best_distance"] <= thr)
        succ_rate = (succ.groupby(param_cols)["success"].mean()*100.0).reset_index().rename(columns={"success":"success_rate_pct"})
        agg = agg.merge(succ_rate, on=param_cols, how="left")
    else:
        agg["success_rate_pct"] = np.nan


    summary = agg.merge(best_rows, on=param_cols, how="left")

 
    order = param_cols + [
        "trials","best_distance_min","best_distance_mean","best_distance_med","best_distance_std",
        "q10","q25","q75","q90","time_sec_mean","success_rate_pct","best_path_of_best"
    ]
    return summary[order]

def choose_best(summary):
    s = summary.copy()


    if TIME_WEIGHT != 0.0:
        time_norm = (s["time_sec_mean"] - s["time_sec_mean"].min()) / (s["time_sec_mean"].max() - s["time_sec_mean"].min() + 1e-9)
    else:
        time_norm = 0.0

    if BEST_STRATEGY == "min_mean":
        s = s.sort_values(["best_distance_mean","best_distance_std","time_sec_mean"], ascending=[True,True,True])

    elif BEST_STRATEGY == "mean+lambda_std":
        score = s["best_distance_mean"] + LAMBDA_STD*s["best_distance_std"].fillna(0.0) + TIME_WEIGHT*(time_norm if isinstance(time_norm, np.ndarray) else 0.0)
        s = s.assign(score=score).sort_values(["score","best_distance_mean","time_sec_mean"], ascending=[True,True,True])

    elif BEST_STRATEGY == "ucb":
        
        s_eff = s["best_distance_std"].fillna(0.0) / np.sqrt(s["trials"].clip(lower=1))
        ucb = s["best_distance_mean"] + UCB_Z * s_eff + TIME_WEIGHT*(time_norm if isinstance(time_norm, np.ndarray) else 0.0)
        s = s.assign(ucb=ucb).sort_values(["ucb","time_sec_mean"], ascending=[True,True])

    elif BEST_STRATEGY == "median_q75":
        
        score = 0.7*s["best_distance_med"] + 0.3*s["q75"] + TIME_WEIGHT*(time_norm if isinstance(time_norm, np.ndarray) else 0.0)
        s = s.assign(score=score).sort_values(["score","time_sec_mean"], ascending=[True,True])

    elif BEST_STRATEGY == "pareto":
        # frente de Pareto em (mean, std, time). Mantém só não-dominados e ordena pelo mean
        arr = s[["best_distance_mean","best_distance_std","time_sec_mean"]].to_numpy()
        dominated = np.zeros(len(s), dtype=bool)
        for i in range(len(s)):
            if dominated[i]: 
                continue
            for j in range(len(s)):
                if i==j: 
                    continue
                # j domina i se é <= em tudo e < em pelo menos um
                if (arr[j] <= arr[i]).all() and (arr[j] < arr[i]).any():
                    dominated[i] = True
                    break
        s = s.assign(pareto=~dominated)
        s = s[s["pareto"]].sort_values(["best_distance_mean","best_distance_std","time_sec_mean"], ascending=[True,True,True])

    elif BEST_STRATEGY == "custom":
        
        s_ok = s[s["success_rate_pct"].fillna(0) >= 50]
        if len(s_ok)==0:
            s_ok = s
        score = s_ok["best_distance_mean"] + 0.5*s_ok["best_distance_std"].fillna(0.0) + TIME_WEIGHT*(time_norm if isinstance(time_norm, np.ndarray) else 0.0)
        s = s_ok.assign(score=score).sort_values(["score","time_sec_mean"], ascending=[True,True])

    else:
        raise ValueError(f"Estratégia desconhecida: {BEST_STRATEGY}")

    return s.iloc[0].to_dict(), s

def main_run():
    print("== TSP GA Parameter Tuning (robusto) ==")
    print(f"Trials/combo: {TRIALS_PER_COMBO} | Estratégia: {BEST_STRATEGY}")
    if OPT_KNOWN:
        print(f"Ótimo conhecido: {OPT_VALUE} | success_eps: {SUCCESS_EPS*100:.1f}%\n")

    dm = get_distance_matrix()
    combos = all_param_combos(GRID)
    print(f"Total de combinações: {len(combos)} | Execuções totais: {len(combos)*TRIALS_PER_COMBO}\n")

    rows = []
    for ci, params in enumerate(combos, 1):
        print(f"[{ci}/{len(combos)}] {params}")
        for t in range(TRIALS_PER_COMBO):
            seed = BASE_SEED + t  
            r = run_single_trial(dm, params, seed)
            rows.append(r)
            print(f"  - trial {t+1}/{TRIALS_PER_COMBO}: dist={r['best_distance']:.2f} time={r['time_sec']:.2f}s (seed={seed})")

    df_trials = pd.DataFrame(rows)
    df_trials.to_csv(TRIALS_CSV, index=False)
    print(f"\nSalvo: {TRIALS_CSV}")

    df_summary = summarize_trials(df_trials)
    df_summary = df_summary.sort_values(["best_distance_mean","best_distance_std","time_sec_mean"], ascending=[True,True,True]).reset_index(drop=True)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Salvo: {SUMMARY_CSV}\n")

    best, ranked = choose_best(df_summary)
    print("== MELHOR COMBINAÇÃO (segundo a estratégia) ==")
    print({k:best[k] for k in ["mutation_rate","num_paths","num_generations","num_best","k_tournament"]})
    print(f"trials={best['trials']}, mean={best['best_distance_mean']:.2f}, std={0.0 if pd.isna(best['best_distance_std']) else best['best_distance_std']:.2f}, "
          f"median={best['best_distance_med']:.2f}, q75={best['q75']:.2f}, min={best['best_distance_min']:.2f}, "
          f"time_mean={best['time_sec_mean']:.2f}s, success%={best['success_rate_pct'] if not pd.isna(best['success_rate_pct']) else 'NA'}")
    print("\nDica pro report:")
    print("- Justificamos a escolha por uma métrica robusta a sorte (ex.: mean+λ·std / UCB / Pareto).")
    print("- Mantivemos seeds comuns entre combos (BASE_SEED+i), reduzindo variância de comparação.")
    print("- Reportamos média, variância, quantis e taxa de sucesso (≤ ótimo·(1+ε)).")

if __name__ == "__main__":
    main_run()
