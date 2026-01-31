import os
import numpy as np
import pandas as pd
import optuna
from typing import List
import xgboost as xgb
from dataclasses import dataclass
from pandas import IndexSlice as idx
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from build_dataset import build_range, CFG

# global setting
cfg = CFG(
    LOCAL_ROOT = "..",
    ALPHA101_DIR = "../get_all_alpha_101/daily",
    ALPHA191_DIR = "../get_all_alpha_191/daily",
    JQF_DIR = "../get_factor_values/daily",
    PRIVATE_DIR = "..",
    INDUSTRY_DIR = "../get_industry",
    TARGET_DIR = "../label_mode=open_next/k=3/bench=000905.XSHG",
)

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

def make_monthly_splits(
    all_dates: List[pd.Timestamp],
    train_months: int = 12,            
    step_months: int = 1,              
    purge_days: int = 5,               
    embargo_days: int = 0,             
    train_first_day_only: bool = False
) -> List[Split]:

    didx = pd.DatetimeIndex(pd.to_datetime(all_dates)).unique().sort_values()
    if len(didx) != len(all_dates):
        all_dates = list(didx)

    pos = pd.Series(np.arange(len(didx)), index=didx)

    mons = didx.to_period('M')
    unique_mons = pd.Index(pd.unique(mons))
    month2days = {m: list(didx[mons == m]) for m in unique_mons}

    splits: List[Split] = []
    embargo_excluded: set[pd.Timestamp] = set()

    i = train_months
    while i < len(unique_mons):
        val_mon = unique_mons[i]
        val_days = month2days[val_mon]
        if len(val_days) == 0:
            i += step_months
            continue
        
        train_mons = unique_mons[i - train_months : i]

        if train_first_day_only:
            train_days = [month2days[m][0] for m in train_mons if len(month2days[m]) > 0]
        else:
            train_days = [d for m in train_mons for d in month2days[m]]

        if purge_days > 0:
            v0 = val_days[0]
            v0_pos = int(pos[v0])
            cutoff_pos = max(0, v0_pos - purge_days)
            train_days = [d for d in train_days if int(pos[d]) < cutoff_pos]

        if embargo_excluded:
            train_days = [d for d in train_days if d not in embargo_excluded]

        splits.append(Split(train_dates=train_days, valid_dates=val_days))

        if embargo_days > 0:
            v1 = val_days[-1]
            v1_pos = int(pos[v1])
            emb_end_pos = min(len(didx) - 1, v1_pos + embargo_days)
            emb_days = list(didx[v1_pos + 1 : emb_end_pos + 1])
            embargo_excluded.update(emb_days)

        i += step_months

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


def make_time_decay(index: pd.MultiIndex, half_life=126, ref_date=None):
    d_unique = pd.Index(pd.to_datetime(index.get_level_values('date'))).unique().sort_values()
    pos = pd.Series(np.arange(len(d_unique)), index=d_unique)
    if ref_date is None:
        ref_date = d_unique.max()
    age = (pos[ref_date] - pos).clip(lower=0)
    w_day = (0.5) ** (age / float(half_life))
    return w_day.reindex(pd.to_datetime(index.get_level_values('date'))).set_axis(index)


def make_U_weight(y_tgt: pd.Series, strength: float = 1.0) -> pd.Series:
    y = y_tgt.copy()
    mask = y.notna()
    if not mask.any() or strength <= 0:
        return pd.Series(1.0, index=y.index, dtype='float32')

    levels = np.sort(y[mask].unique())
    K = len(levels)

    pos = np.arange(K, dtype='float32')
    mid = (K - 1) / 2.0
    d = np.abs(pos - mid) / mid
    w_levels = 1.0 + strength * d

    level_to_weight = {lvl: w_levels[i] for i, lvl in enumerate(levels)}
    w = y.map(level_to_weight).astype('float32')
    w[~mask] = 1.0
    return w



def make_reg_weights(index: pd.MultiIndex, 
                     per_day_normalize: bool = True, 
                     time_half_life: int = 126, 
                     ref_date=None,
                     *,
                     y_tgt: pd.Series = None,
                     ext_multiplier: float = 1.0,
                     ):

    w = pd.Series(1.0, index=index, dtype='float32')
    if y_tgt is not None and ext_multiplier is not None and ext_multiplier > 0:
        y_loc = y_tgt.reindex(index)
        w = make_U_weight(y_loc, strength=ext_multiplier)

    if per_day_normalize:
        day_sum = w.groupby(level='date').transform('sum').astype('float32')
        day_sum = day_sum.replace(0.0, np.nan)
        w = (w / day_sum).fillna(0.0)

    if time_half_life is not None:
        w_time = make_time_decay(index, half_life=time_half_life, ref_date=ref_date).astype('float32')
        w = w * w_time

    return w.astype('float32')


def era_corr(y_true: pd.Series, y_pred: pd.Series, method='spearman'):
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


def long_only_stats(y_cont: pd.Series, 
                    score: pd.Series, 
                    p_long: float = 0.05,
                    min_n_per_day: int = 20,
                    nw_lags: int = None
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


def fit_eval_one_fold_reg_xgb(
    X_tr, y_tgt_tr,
    X_va, y_tgt_va,
    y_cont_va,
    params,
    num_boost_round: int = 400,
    verbose_eval: int = 100,
    sample_weight_tr = None,
    p_long: float = 0.05,
    nw_lags: int = 2 
):
    reg = xgb.XGBRegressor(
        **params,
        n_estimators=num_boost_round,
    )
    
    reg.fit(
        X_tr, y_tgt_tr,
        sample_weight=sample_weight_tr,
        eval_set=[(X_va, y_tgt_va)],
        verbose=verbose_eval,
    )

    pred = pd.Series(reg.predict(X_va), index=X_va.index, name='pred')

    y_cont_1d = _to_1d_series(y_cont_va)
    corr_s = era_corr(y_cont_1d, pred, method='spearman')
    corr_mean = corr_s.mean()

    if len(corr_s) > 1 and corr_s.std(ddof=1) > 0:
        if nw_lags is None or nw_lags <= 0:
            corr_ir = corr_mean / corr_s.std(ddof=1) * np.sqrt(len(corr_s))
        else:
            corr_ir = _newey_west_t(corr_s, max_lag=nw_lags)
    else:
        corr_ir = np.nan

    per_day_ret, per_day_mean, per_day_t = long_only_stats(
        y_cont_va, pred, p_long=p_long, nw_lags=nw_lags
    )

    summary = dict(
        eras=len(corr_s),
        corr_mean=corr_mean,
        corr_ir=corr_ir,
        per_day_ret_mean=per_day_mean,
        per_day_t_stat=per_day_t
    )
    return reg, pred, per_day_ret, summary


def train_regression_wfo_xgb(
    X: pd.DataFrame,
    y_cont: pd.Series,
    *,
    n_bins: int = 7,
    train_months: int = 36,
    step_months: int = 1,
    purge_days: int = 10,
    embargo_days: int = 0,
    params=None,
    num_boost_round: int = 300,
    verbose_eval: int = 100,
    per_day_normalize: bool = True,
    time_half_life: int = 250,
    ext_multiplier: float = 1.5,
    p_long: float = 0.05,
    nw_lags: int = 2,
    score_csv_path: str  = None, 
    long_csv_path: str  = None,
):

    assert X.index.equals(y_cont.index), 'Feature Dates Do Not Match Target Dates'

    y_tgt = make_target(y_cont, n_bins=n_bins)

    all_dates = uniq_dates(X.index)
    splits = make_monthly_splits(
        all_dates=all_dates,
        train_months=train_months,
        step_months=step_months,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    if params is None:
        params = dict(
            objective='reg:squarederror',
            eval_metric='rmse',
            learning_rate=0.1,
            max_depth=5,
            subsample=0.73,
            colsample_bytree=0.95,
            reg_alpha=0.13,
            reg_lambda=5.53,
            n_jobs=-1,
            tree_method='gpu_hist',
            random_state=2025,
            device='cuda'
        )

    models, fold_summaries, val_scores = [], [], []
    for k, sp in enumerate(splits, 1):
        X_tr = subset_by_dates(X, sp.train_dates)
        X_va = subset_by_dates(X, sp.valid_dates)

        y_cont_tr = subset_by_dates(y_cont, sp.train_dates)
        y_cont_va = subset_by_dates(y_cont, sp.valid_dates)

        y_tgt_tr  = subset_by_dates(y_tgt,  sp.train_dates)
        y_tgt_va  = subset_by_dates(y_tgt,  sp.valid_dates)

        ref_day = pd.to_datetime(sp.train_dates[-1])
        w_tr = make_reg_weights(
            X_tr.index,
            per_day_normalize=per_day_normalize,
            time_half_life=time_half_life,
            ref_date=ref_day,
            y_tgt=y_tgt_tr,
            ext_multiplier=ext_multiplier,
        )

        reg, pred, per_day_ret, summ = fit_eval_one_fold_reg_xgb(
            X_tr, y_tgt_tr,
            X_va, y_tgt_va,
            y_cont_va,
            params,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            sample_weight_tr=w_tr,
            p_long=p_long,
            nw_lags=nw_lags,
        )

        long_codes = {}
        for d, s_day in pred.groupby(level='date'):
            s_day = s_day.droplevel('date')
            n = len(s_day)
            if n == 0:
                continue
            k_top = max(1, int(np.ceil(p_long * n)))
            top_codes = s_day.nlargest(k_top).index.tolist()
            long_codes[str(pd.to_datetime(d).date())] = top_codes

        models.append(reg)
        val_scores.append(pred)
        fold_info = dict(
            fold=k,
            train_start=str(sp.train_dates[0]),
            train_end=str(sp.train_dates[-1]),
            valid_start=str(sp.valid_dates[0]),
            valid_end=str(sp.valid_dates[-1]),
            eras=summ['eras'],
            corr_mean=summ['corr_mean'],
            corr_ir=summ['corr_ir'],
            long_mean=summ['per_day_ret_mean'],
            long_t=summ['per_day_t_stat'],
            long_codes=long_codes, 
        )
        fold_summaries.append(fold_info)

        print(
            f"[Fold {k}] Train {sp.train_dates[0]}~{sp.train_dates[-1]} | "
            f"Valid {sp.valid_dates[0]}~{sp.valid_dates[-1]} | "
            f"IC_mean={summ['corr_mean']:.4f}  IC_t={summ['corr_ir']:.2f}  Eras={summ['eras']} | "
            f"LongMean={summ['per_day_ret_mean']:.4e}  Long_t={summ['per_day_t_stat']:.2f}"
        )

    val_scores = pd.concat(val_scores).sort_index()
    reports = pd.DataFrame(fold_summaries)

    score_df = val_scores.to_frame('score')
    
    score_df = score_df.reset_index()   # columns: ['date','code','score']

    def _select_top(g: pd.DataFrame):
        n = len(g)
        if n == 0:
            return g.iloc[0:0]
        k_top = max(1, int(np.ceil(p_long * n)))
        return g.nlargest(k_top, 'score')

    long_df = (
        score_df
        .groupby('date', sort=False, group_keys=False)
        .apply(_select_top)
        .reset_index(drop=True)
    )
    
    if score_csv_path is not None:
        score_df.to_csv(score_csv_path, index=False)
        print(f"[INFO] score_df saved to {score_csv_path}")

    if long_csv_path is not None:
        long_df.to_csv(long_csv_path, index=False)
        print(f"[INFO] long_df (top {p_long:.0%}) saved to {long_csv_path}")

    return dict(
        models=models,
        splits=splits,
        val_scores=val_scores,
        reports=reports,
        score_df=score_df,
        long_df=long_df,
    )


# from xgboost_find_params
params = {'objective': 'reg:squarederror', 
          'eval_metric': 'rmse', 
          'learning_rate': 0.1622839170779374, 
          'max_depth': 6, 
          'subsample': 0.8140249099519008, 
          'colsample_bytree': 0.5926997376568158, 
          'reg_alpha': 0.00833214616268351,
          'reg_lambda': 0.24634602207186726,
          'gamma': 0.007383495322524604,
          'random_state': 2025,
          'n_jobs':-1,
          'tree_method': 'gpu_hist',
          'device': 'cuda'}

res = train_regression_wfo_xgb(
    X,
    y.y,
    n_bins = 7,
    train_months = 36,
    step_months = 1,
    purge_days = 5,
    embargo_days = 0,
    params = params,
    num_boost_round = 200,
    verbose_eval = 100,
    per_day_normalize = True,
    time_half_life = 120,
    ext_multiplier = 1.5,
    p_long = 0.05,
    nw_lags = 2,
    score_csv_path = 'xgboost_3yr_pred.csv', # for model stacking only
    long_csv_path = 'xgboost_3yr_long_list.csv')

