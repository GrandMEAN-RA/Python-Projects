
# fico_bucketing.py
import numpy as np
import pandas as pd
from pathlib import Path

# --------- Helper functions ----------
def compress_scores_and_counts(fico_scores, defaults):
    arr = np.vstack([fico_scores, defaults]).T
    arr = arr[arr[:,0].argsort()]
    unique_vals, counts = np.unique(arr[:,0], return_counts=True)
    defaults_sum = []
    for v in unique_vals:
        defaults_sum.append(int(arr[arr[:,0] == v][:,1].sum()))
    defaults_sum = np.array(defaults_sum, dtype=int)
    return unique_vals.astype(float), counts.astype(int), defaults_sum

def precompute_interval_loglik(counts, defaults_sum, eps=1e-6):
    n = len(counts)
    prefix_counts = np.concatenate([[0], np.cumsum(counts)])
    prefix_defaults = np.concatenate([[0], np.cumsum(defaults_sum)])
    loglik = np.full((n, n), -np.inf)
    for i in range(n):
        for j in range(i, n):
            ni = prefix_counts[j+1] - prefix_counts[i]
            ki = prefix_defaults[j+1] - prefix_defaults[i]
            p_hat = (ki + eps) / (ni + 2*eps)
            p_hat = np.clip(p_hat, eps, 1 - eps)
            loglik[i, j] = ki * np.log(p_hat) + (ni - ki) * np.log(1 - p_hat)
    return loglik

def optimal_buckets_loglik(values, counts, defaults_sum, K, eps=1e-6):
    n = len(values)
    loglik = precompute_interval_loglik(counts, defaults_sum, eps)
    NEG_INF = -1e18
    dp = np.full((K+1, n+1), NEG_INF)
    prev = np.full((K+1, n+1), -1, dtype=int)
    dp[0,0] = 0.0
    for k in range(1, K+1):
        for j in range(1, n+1):
            best_val = NEG_INF
            best_i = -1
            # iterate possible split points
            for i in range(0, j):
                val = dp[k-1, i] + loglik[i, j-1]
                if val > best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            prev[k, j] = best_i
    # backtrack to get buckets
    boundaries = []
    j = n
    for k in range(K, 0, -1):
        i = prev[k, j]
        if i is None or i < 0:
            i = 0
        boundaries.append((i, j-1))
        j = i
    boundaries = list(reversed(boundaries))
    bucket_value_bounds = [(values[s], values[e]) for (s, e) in boundaries]
    return boundaries, bucket_value_bounds

def map_score_to_rating(score, bucket_value_bounds):
    K = len(bucket_value_bounds)
    for idx, (low, high) in enumerate(bucket_value_bounds):
        if low <= score <= high:
            return K - idx  # invert so 1 = best (highest FICO)
    if score < bucket_value_bounds[0][0]:
        return K
    else:
        return 1

# --------- Main procedure ----------
def run_fico_bucketing(csv_path, K=5, fico_col='fico_score', default_col='default', out_prefix='fico_buckets'):
    df = pd.read_csv(csv_path)
    if fico_col not in df.columns or default_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{fico_col}' and '{default_col}'. Found: {df.columns.tolist()}")
    fico_scores = df[fico_col].astype(float).values
    defaults = df[default_col].astype(int).values
    values, counts, defaults_sum = compress_scores_and_counts(fico_scores, defaults)
    boundaries_idx, bucket_bounds = optimal_buckets_loglik(values, counts, defaults_sum, K)
    # Build summary table
    prefix_counts = np.concatenate([[0], np.cumsum(counts)])
    prefix_defaults = np.concatenate([[0], np.cumsum(defaults_sum)])
    rows = []
    for b_idx, (s_idx, e_idx) in enumerate(boundaries_idx):
        low = values[s_idx]; high = values[e_idx]
        ni = int(prefix_counts[e_idx+1] - prefix_counts[s_idx])
        ki = int(prefix_defaults[e_idx+1] - prefix_defaults[s_idx])
        pd_obs = ki / ni if ni>0 else 0.0
        rating = K - b_idx
        rows.append({
            'bucket': b_idx+1,
            'score_low': float(low),
            'score_high': float(high),
            'count': ni,
            'defaults': ki,
            'observed_pd': pd_obs,
            'rating (1=best)': rating
        })
    bucket_table = pd.DataFrame(rows)
    bucket_table = bucket_table[['bucket','score_low','score_high','count','defaults','observed_pd','rating (1=best)']]
    # Add rating to original df
    df['rating'] = df[fico_col].apply(lambda x: map_score_to_rating(x, bucket_bounds))
    # Save outputs
    out_table_csv = f"{out_prefix}_table_K{K}.csv"
    out_map_csv = f"{out_prefix}_mapped_K{K}.csv"
    bucket_table.to_csv(out_table_csv, index=False)
    df.to_csv(out_map_csv, index=False)
    print(f"Saved bucket table to: {out_table_csv}")
    print(f"Saved full mapped dataset to: {out_map_csv}")
    print("\nBucket table:")
    print(bucket_table.to_string(index=False))
    return bucket_table, bucket_bounds, df

# --------- Usage example ----------
if __name__ == "__main__":
    # change filename/path if needed
    csv_file = r"C:\Users\EBUNOLUWASIMI\Dropbox\Study Materials\Python\JP Morgan Chase\Task 3 and 4_Loan_Data.csv"
    K = 5  # default number of buckets; change if you prefer
    bucket_table, bucket_bounds, df_mapped = run_fico_bucketing(csv_file, K=K)
