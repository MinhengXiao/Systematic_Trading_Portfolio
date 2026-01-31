import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

def uniq_dates(index: pd.MultiIndex):
    return (pd.Index(pd.to_datetime(index.get_level_values('date')))
            .sort_values().unique().tolist())

def subset_by_dates(obj, dates):
    if isinstance(obj, pd.Series):
        return obj.loc[(pd.to_datetime(dates), slice(None))].sort_index()
    else:
        return obj.loc[(pd.to_datetime(dates), slice(None)), :].sort_index()


@dataclass
class Split:
    train_dates: list
    valid_dates: list

def make_rolling_day_splits(
    all_dates: List[pd.Timestamp],
    train_days: int = 252,
    valid_days: int = 10,
    step_days: int = 1,
    purge_days: int = 10,
    embargo_days: int = 0,
) -> List[Split]:
    didx = pd.DatetimeIndex(pd.to_datetime(all_dates)).unique().sort_values()
    if len(didx) == 0:
        return []

    pos = pd.Series(np.arange(len(didx)), index=didx)

    train_days = int(train_days)
    valid_days = int(valid_days)
    step_days = int(step_days)
    purge_days = int(purge_days)
    embargo_days = int(embargo_days)

    splits: List[Split] = []
    embargo_excluded: set[pd.Timestamp] = set()

    i = train_days
    while i + valid_days <= len(didx):
        val_days = list(didx[i:i + valid_days])
        if len(val_days) == 0:
            i += step_days
            continue

        train_days_list = list(didx[i - train_days:i])

        if purge_days > 0:
            v0 = val_days[0]
            v0_pos = int(pos[v0])
            cutoff_pos = max(0, v0_pos - purge_days)
            train_days_list = [d for d in train_days_list if int(pos[d]) < cutoff_pos]

        if embargo_excluded:
            train_days_list = [d for d in train_days_list if d not in embargo_excluded]

        splits.append(Split(train_dates=train_days_list, valid_dates=val_days))

        if embargo_days > 0:
            v1 = val_days[-1]
            v1_pos = int(pos[v1])
            emb_end_pos = min(len(didx) - 1, v1_pos + embargo_days)
            emb_days = list(didx[v1_pos + 1: emb_end_pos + 1])
            embargo_excluded.update(emb_days)

        i += step_days

    return splits


def make_target(
    y_cont: pd.Series,
    n_bins = 7
):
    if n_bins == 5:
        edges = (0.10, 0.30, 0.70, 0.90)
        labels = (0.00, 0.25, 0.50, 0.75, 1.00)

    elif n_bins == 7:
        edges = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
        labels = (0.00, 1/6, 2/6, 3/6, 4/6, 5/6, 1.00)

    else:
        raise ValueError("n_bins illegal!")

    if isinstance(y_cont, pd.DataFrame):
        y_cont = y_cont.squeeze("columns")
    assert isinstance(y_cont.index, pd.MultiIndex) and y_cont.index.names[:2] == ['date','code']

    pct = y_cont.groupby(level='date').transform(
        lambda s: s.rank(pct=True, method='average')
    )

    bins = [0.0] + list(edges) + [1.0 + 1e-9]
    tgt = pd.cut(pct, bins=bins, labels=labels, include_lowest=True, right=True).astype('float32')

    tgt[pct.isna()] = np.nan
    return tgt.rename('target')

def era_corr_safe(y_true: pd.Series, y_pred: pd.Series, method='spearman'):
    df = pd.DataFrame({'y': y_true, 'p': y_pred}).dropna()
    corrs = (
        df.groupby(level='date')
          .apply(lambda g: g['y'].corr(g['p'], method=method))
          .dropna()
    )
    return corrs


def _to_1d_series(x, name=None, index=None):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name or 'input'} has {x.shape[1]} columns; expected single-column")
        s = x.iloc[:, 0]
        s.name = name or s.name
        return s
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = name or s.name
        return s
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, index=index, name=name)


def _newey_west_t(per_day: pd.Series, max_lag: int) -> float:
    import numpy as np

    r = per_day.dropna().values
    n = len(r)
    if n <= 1:
        return np.nan

    mu = r.mean()
    x = r - mu

    gamma0 = np.dot(x, x) / n
    var_hat = gamma0

    L = min(max_lag, n - 1)
    for lag in range(1, L + 1):
        cov = np.dot(x[lag:], x[:-lag]) / n
        weight = 1.0 - lag / (L + 1.0)
        var_hat += 2.0 * weight * cov

    se = np.sqrt(var_hat / n)
    if se == 0 or not np.isfinite(se):
        return np.nan
    return mu / se


def long_only_stats_safe(y_cont: pd.Series, 
                    score: pd.Series,       # the prediction
                    p_long: float = 0.05,   # percentage of long
                    min_n_per_day: int = 20,
                    nw_lags: int = None  # Newey-West
                   ):
    y_cont = _to_1d_series(y_cont, name='ret')
    score  = _to_1d_series(
        score, name='s',
        index=y_cont.index if not isinstance(score, (pd.Series, pd.DataFrame)) else None
    )
    df = pd.concat([y_cont, score], axis=1, join='inner').dropna()

    def _one(g: pd.DataFrame):
        n = len(g)
        if n < min_n_per_day:
            return np.nan
        kL = max(1, int(np.floor(n * p_long)))
        return g.nlargest(kL, 's')['ret'].mean()

    per_day = df.groupby(level='date', sort=False).apply(_one)

    if per_day.count() > 1 and per_day.std(ddof=1) > 0:
        if nw_lags is None or nw_lags <= 0:
            ir = per_day.mean() / per_day.std(ddof=1) * np.sqrt(per_day.count())
        else:
            ir = _newey_west_t(per_day, max_lag=nw_lags)
    else:
        ir = np.nan

    return per_day, per_day.mean(), ir

import json
from pathlib import Path
from tqdm import tqdm

def normalize_and_align_y(y_raw, target_index: pd.MultiIndex, y_col: str = "y") -> pd.Series:
    if isinstance(y_raw, pd.DataFrame):
        if y_col in y_raw.columns:
            s = y_raw[y_col]
        elif y_raw.shape[1] == 1:
            s = y_raw.iloc[:, 0]
        elif ("date" in y_raw.columns) and ("code" in y_raw.columns):
            cols = [c for c in y_raw.columns if c not in ("date", "code")]
            if len(cols) != 1:
                raise ValueError("Cannot infer y column from y_raw.")
            tmp = y_raw.copy()
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
            tmp["code"] = tmp["code"].astype(str)
            s = pd.Series(tmp[cols[0]].values,
                          index=pd.MultiIndex.from_arrays([tmp["date"], tmp["code"]], names=["date", "code"]))
        else:
            raise ValueError("Unsupported y_raw DataFrame format.")
    else:
        s = y_raw

    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError("y must be MultiIndex(date,code) or provide date/code columns.")

    d0 = pd.to_datetime(s.index.get_level_values(0)).normalize()
    c0 = s.index.get_level_values(1).astype(str)
    s = s.copy()
    s.index = pd.MultiIndex.from_arrays([d0, c0], names=["date", "code"])
    if s.index.has_duplicates:
        s = s.groupby(level=["date", "code"], sort=False).last()
    s = pd.to_numeric(s, errors="coerce").astype(float)

    # align
    d1 = pd.to_datetime(target_index.get_level_values(0)).normalize()
    c1 = target_index.get_level_values(1).astype(str)
    tgt = pd.MultiIndex.from_arrays([d1, c1], names=["date", "code"])
    out = s.reindex(tgt)
    out.index = target_index
    return out


def _norm_pred_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["code"] = d["code"].astype(str)
    d = d[["date", "code", "score"]].rename(columns={"score": col_name})
    return d


def cs_median_fill_scores(df_wide: pd.DataFrame) -> pd.DataFrame:
    def _fill(g: pd.DataFrame) -> pd.DataFrame:
        med = g.median(axis=0, skipna=True)
        out = g.fillna(med)
        return out

    out = df_wide.groupby(level="date", sort=False).apply(_fill)
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 3:
        if out.index.names[0] == "date" and out.index.names[1] == "date":
            out.index = out.index.droplevel(0)
    out = out.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.5)  # score中性值
    return out

def cs_zscore_df(df_wide: pd.DataFrame) -> pd.DataFrame:
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        g = g.astype(float)
        mu = g.mean(axis=0)
        sd = g.std(axis=0, ddof=1).replace(0.0, np.nan)
        z = (g - mu) / sd
        return z
    z = df_wide.groupby(level="date", sort=False).apply(_z)
    if isinstance(z.index, pd.MultiIndex) and z.index.nlevels >= 3:
        if z.index.names[0] == "date" and z.index.names[1] == "date":
            z.index = z.index.droplevel(0)
    z = z.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z

def ridge_fit_predict(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, alpha: float = 10.0
) -> np.ndarray:
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float)
    Xva = np.asarray(Xva, float)

    m = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
    X = Xtr[m]
    y = ytr[m]

    # intercept (unpenalized)
    ones = np.ones((X.shape[0], 1))
    X1 = np.concatenate([ones, X], axis=1)
    ones_va = np.ones((Xva.shape[0], 1))
    Xva1 = np.concatenate([ones_va, Xva], axis=1)

    p = X1.shape[1]
    A = X1.T @ X1
    b = X1.T @ y

    P = np.zeros((p, p))
    P[1:, 1:] = np.eye(p - 1) * float(alpha)
    A = A + P

    try:
        coef = np.linalg.solve(A, b)
    except Exception:
        return np.full(Xva.shape[0], np.nan)

    return Xva1 @ coef


def train_ridge_fusion_dayrolling(
    xgb1: pd.DataFrame,
    xgb3: pd.DataFrame,
    xgb6: pd.DataFrame,
    y_full,
    *,
    merge_how: str = "inner",
    # rolling split
    train_days: int = 252,
    valid_days: int = 10,
    step_days: int = 10,
    purge_days: int = 10,
    embargo_days: int = 0,
    
    # target
    n_bins: int = 7,
    
    # ridge
    ridge_alpha: float = 10.0,
    
    # preprocessing
    do_cs_median_fill: bool = True,
    do_cs_zscore: bool = True,
    
    # evaluation
    p_long: float = 0.10,
    nw_lags: int = 2,
) -> Dict:
    # 1) merge predictions
    d1 = _norm_pred_df(xgb1, "s1")
    d3 = _norm_pred_df(xgb3, "s3")
    d6 = _norm_pred_df(xgb6, "s6")
    df = d1.merge(d3, on=["date", "code"], how=merge_how).merge(d6, on=["date", "code"], how=merge_how)

    df_wide = df.set_index(["date", "code"]).sort_index()
    X_feat = df_wide[["s1", "s3", "s6"]].astype(float)

    # 2) preprocess X
    if do_cs_median_fill:
        X_feat = cs_median_fill_scores(X_feat)
    if do_cs_zscore:
        X_feat = cs_zscore_df(X_feat)

    # 3) align y_full to X_feat.index
    y_cont_use = normalize_and_align_y(y_full, X_feat.index, y_col="y")

    # 4) build training target same as xgb training
    y_tgt = make_target(y_cont_use, n_bins=n_bins)

    # 5) rolling splits on available dates
    all_dates = pd.Index(pd.to_datetime(X_feat.index.get_level_values("date")).unique()).sort_values().tolist()
    splits = make_rolling_day_splits(
        all_dates,
        train_days=train_days,
        valid_days=valid_days,
        step_days=step_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    fold_rows = []
    preds_all = []

    for k, sp in tqdm(enumerate(splits, 1)):
        X_tr = subset_by_dates(X_feat, sp.train_dates)
        X_va = subset_by_dates(X_feat, sp.valid_dates)

        y_tr = subset_by_dates(y_tgt, sp.train_dates)
        y_va_cont = subset_by_dates(y_cont_use, sp.valid_dates)

        if X_tr.empty or X_va.empty:
            continue

        yhat_va = ridge_fit_predict(
            X_tr.to_numpy(dtype=float),
            pd.to_numeric(_to_1d_series(y_tr), errors="coerce").to_numpy(dtype=float),
            X_va.to_numpy(dtype=float),
            alpha=float(ridge_alpha),
        )
        pred_va = pd.Series(yhat_va, index=X_va.index, name="pred")
        preds_all.append(pred_va)

        # eval
        corr_s = era_corr_safe(_to_1d_series(y_va_cont), pred_va, method="spearman")
        corr_mean = float(corr_s.mean()) if len(corr_s) > 0 else np.nan
        if len(corr_s) > 1 and float(corr_s.std(ddof=1)) > 0:
            corr_ir = float(_newey_west_t(corr_s, max_lag=int(nw_lags))) if (nw_lags is not None and nw_lags > 0) else float(corr_mean / corr_s.std(ddof=1) * np.sqrt(len(corr_s)))
        else:
            corr_ir = np.nan

        per_day_ret, per_day_mean, per_day_t = long_only_stats_safe(
            y_va_cont, pred_va, p_long=float(p_long), nw_lags=nw_lags
        )

        fold_rows.append({
            "fold": k,
            "train_start": str(pd.to_datetime(sp.train_dates[0]).date()),
            "train_end": str(pd.to_datetime(sp.train_dates[-1]).date()),
            "valid_start": str(pd.to_datetime(sp.valid_dates[0]).date()),
            "valid_end": str(pd.to_datetime(sp.valid_dates[-1]).date()),
            "eras": int(len(corr_s)),
            "corr_mean": corr_mean,
            "corr_ir": corr_ir,
            "long_mean": float(per_day_mean) if np.isfinite(per_day_mean) else np.nan,
            "long_t": float(per_day_t) if np.isfinite(per_day_t) else np.nan,
            "ridge_alpha": float(ridge_alpha),
        })

        print(
            f"[Fold {k}] Train {fold_rows[-1]['train_start']}~{fold_rows[-1]['train_end']} | "
            f"Valid {fold_rows[-1]['valid_start']}~{fold_rows[-1]['valid_end']} | "
            f"IC_mean={corr_mean:.4f} IC_t={corr_ir:.2f} Eras={len(corr_s)} | "
            f"LongMean={float(per_day_mean) if np.isfinite(per_day_mean) else np.nan:.4e} Long_t={float(per_day_t) if np.isfinite(per_day_t) else np.nan:.2f}"
        )

    reports = pd.DataFrame(fold_rows)
    splits_df = reports[["fold", "train_start", "train_end", "valid_start", "valid_end"]].copy()

    if not preds_all:
        return dict(pred_fused=pd.Series(dtype=float), reports=reports, splits_df=splits_df,
                    score_df=pd.DataFrame(), long_df=pd.DataFrame())

    pred_fused = pd.concat(preds_all).sort_index()
    score_df = pred_fused.to_frame("score").reset_index()

    def _select_top(g: pd.DataFrame):
        n = len(g)
        if n == 0:
            return g.iloc[0:0]
        k_top = max(1, int(np.ceil(float(p_long) * n)))
        return g.nlargest(k_top, "score")

    long_df = (
        score_df.groupby("date", sort=False, group_keys=False)
                .apply(_select_top)
                .reset_index(drop=True)
    )

    return dict(
        pred_fused=pred_fused,
        reports=reports,
        splits_df=splits_df,
        score_df=score_df,
        long_df=long_df,
    )


res_fuse_day = train_ridge_fusion_dayrolling(
    xgb1, xgb3, xgb6,
    y_full=y,
    train_days=252,
    valid_days=1,
    step_days=1,
    purge_days=5,
    ridge_alpha=980.0,
    n_bins=7,
    do_cs_median_fill=False,
    do_cs_zscore=False,
    p_long=0.05,
    nw_lags=2,
)