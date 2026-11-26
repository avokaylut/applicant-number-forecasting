"""Team forecast: default is phase_weekends (user-reported public score 0.53116).
Previous models are retained for reproducibility.
"""
import argparse
import hashlib
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


SEASON_START = "06-20"

SEASON_END = "08-31"

COVID_YEAR = 2020

SUNDAY_VALUE = 0.3

SEED = 0

FEATURE_MODE = 'phase_weekends'

def rmsle(actual, pred):
    """Метрика соревнования."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))

def add_calendar(df):
    df = df.copy()
    df["year"] = df.date.dt.year
    df["dow"] = df.date.dt.dayofweek
    df["doy"] = (df.date - pd.to_datetime(df.year.astype(str) + "-" + SEASON_START)).dt.days  # день кампании
    return df

def make_calendar(year):
    d = pd.DataFrame({"date": pd.date_range(f"{year}-{SEASON_START}", f"{year}-{SEASON_END}")})
    return add_calendar(d)

def _features(d):
    # день кампании, день недели, флаги воскресенья/субботы, период после дедлайна 25.07
    base = np.c_[d.doy, d.dow, (d.dow == 6), (d.dow == 5), d.doy >= 36]
    if FEATURE_MODE != 'calendar':
        return base
    dates = d.date.to_numpy(dtype='datetime64[D]')
    starts = pd.to_datetime(d.year.astype(str) + '-06-20').to_numpy(dtype='datetime64[D]')
    start_dow = pd.to_datetime(starts).dayofweek.to_numpy()
    weekday_index = np.busday_count(starts, dates)
    saturday_number = np.maximum(0, (d.doy.to_numpy() - (5-start_dow) % 7)//7 + 1)
    first_weekday = np.where(start_dow < 5, 0, 7-start_dow)
    first_monday = (7-start_dow) % 7
    # Calendar weekdays, not an assumption about the admissions office's actual opening hours.
    return np.c_[base, weekday_index, saturday_number,
                 d.doy.to_numpy() == first_weekday,
                 (d.doy.to_numpy() == first_monday) & (d.doy.to_numpy() < 7)]

def _gbm(depth=3, it=150, lr=0.05, leaf=8):
    return HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it,
                                         min_samples_leaf=leaf, random_state=SEED)

def model_gbm(tr, te, k=6, decay=0.25, w_covid=0.0):
    """Бустинг на всех днях. Веса затухают к прошлому, 2020 год отключён."""
    t = tr[tr.year.isin(sorted(tr.year.unique())[-k:])]
    w = np.exp(decay * (t.year - t.year.max()).values)
    w = np.where(t.year == COVID_YEAR, w * w_covid, w)
    m = _gbm().fit(_features(t), np.log1p(t.y), sample_weight=w)
    p = np.expm1(m.predict(_features(te))).clip(0)
    return np.where(te.dow.values == 6, SUNDAY_VALUE, p)

def model_gbm_saturday(tr, te, k=8, decay=0.25, sat_w=3.0, sat_year=2021):
    """Бустинг без воскресений; субботы свежих лет усилены, старые - ослаблены
    (доля субботы от будней выросла с ~0.35 до ~0.59)."""
    t = tr[tr.year.isin(sorted(tr.year.unique())[-k:]) & (tr.dow != 6)]
    w = np.exp(decay * (t.year - t.year.max()).values)
    w = np.where(t.year == COVID_YEAR, 0, w)
    w = np.where((t.dow == 5) & (t.year >= sat_year), w * sat_w, w)
    w = np.where((t.dow == 5) & (t.year < sat_year), w * 0.3, w)
    m = _gbm().fit(_features(t), np.log1p(t.y), sample_weight=w)
    p = np.expm1(m.predict(_features(te))).clip(0)
    return np.where(te.dow.values == 6, SUNDAY_VALUE, p)

def _dow_effects(t):
    """Лог-эффект дня недели (относительно Пн-Пт) по каждому году на пике кампании."""
    out = {}
    for y, g in t.groupby("year"):
        g = g[(g.doy >= 8) & (g.doy <= 38)]
        l = np.log1p(g.y)
        e = l.groupby(g.dow).mean() - l[g.dow < 5].mean()
        out[y] = e.reindex(range(6)).fillna(0)
    return pd.DataFrame(out)

def model_hybrid(tr, te, k=7, decay=0.25, sat_k=3):
    """Профиль по дню кампании (по будням, очищенный от дня недели) + эффект дня недели
    по последним sat_k годам (без 2020)."""
    t = tr[tr.year.isin(sorted(tr.year.unique())[-k:]) & (tr.dow != 6)].copy()
    E = _dow_effects(t)
    good = [y for y in E.columns if y != COVID_YEAR]
    Et = E[good[-sat_k:]].mean(axis=1)
    t["l"] = np.log1p(t.y) - [E.loc[d, y] if d < 6 else 0 for d, y in zip(t.dow, t.year)]
    f = t[t.dow < 5]
    w = np.exp(decay * (f.year - f.year.max()).values)
    w = np.where(f.year == COVID_YEAR, 0, w)
    m = _gbm().fit(f[["doy"]].values, f.l.values, sample_weight=w)
    base = m.predict(te[["doy"]].values)
    de = te.dow.map(lambda d: Et[d] if d < 6 else 0).values
    p = np.expm1(base + de).clip(0)
    return np.where(te.dow.values == 6, SUNDAY_VALUE, p)

ENS_K, ENS_DECAY, ENS_SAT_K = 8, 0.4, 2

EARLY_DAYS, EARLY_W, RECENT_DECAY = 10, 0.5, 5.0

DECAYS = (0.25, 0.6, 1.2)

BAG_CONFIGS = (dict(depth=3, it=150), dict(depth=2, it=250), dict(depth=4, it=120, leaf=12),
               dict(depth=3, it=100, lr=0.08, l2=1.0), dict(depth=3, it=150, decay=0.2),
               dict(depth=3, it=150, decay=0.7, k=6))

def model_ensemble(tr, te, decay=ENS_DECAY):
    """Взвешенное среднее трёх моделей в log1p-пространстве."""
    a = np.log1p(model_gbm(tr, te, k=ENS_K, decay=decay))
    b = np.log1p(model_gbm_saturday(tr, te, k=ENS_K, decay=decay))
    c = np.log1p(model_hybrid(tr, te, k=ENS_K, decay=decay, sat_k=ENS_SAT_K))
    return np.expm1(0.4 * a + 0.4 * b + 0.2 * c)

def _gbm_variant(tr, te, k=8, decay=0.4, depth=3, it=150, lr=0.05, leaf=8, l2=0.0, sat_w=3.0):
    """Вариант бустинга с другими гиперпараметрами (для бэггинга)."""
    t = tr[tr.year.isin(sorted(tr.year.unique())[-k:]) & (tr.dow != 6)]
    w = np.exp(decay * (t.year - t.year.max()).values)
    w = np.where(t.year == COVID_YEAR, 0, w)
    w = np.where((t.dow == 5) & (t.year >= 2021), w * sat_w, w)
    w = np.where((t.dow == 5) & (t.year < 2021), w * 0.3, w)
    m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=leaf,
                                      l2_regularization=l2, random_state=SEED)
    m.fit(_features(t), np.log1p(t.y), sample_weight=w)
    p = np.expm1(m.predict(_features(te))).clip(0)
    return np.where(te.dow.values == 6, SUNDAY_VALUE, p)

def model_bagged(tr, te):
    """Среднее (в log1p) шести бустингов с разными гиперпараметрами - снижает дисперсию."""
    return np.expm1(np.mean([np.log1p(_gbm_variant(tr, te, **c)) for c in BAG_CONFIGS], axis=0))

def smooth_weekdays(te, p, win=3):
    """Сглаживание прогноза по будням (скользящее среднее в log1p): убирает ступеньки деревьев."""
    l = np.log1p(p)
    idx = np.where(te.dow.values < 5)[0]
    l[idx] = pd.Series(l[idx]).rolling(win, center=True, min_periods=1).mean().values
    return np.expm1(l)

def _predict_weekends(tr, te):
    parts = [np.log1p(model_ensemble(tr, te, decay=d)) for d in DECAYS]
    parts.append(np.log1p(model_bagged(tr, te)))
    base = np.mean(parts, axis=0)
    smoothed = np.log1p(smooth_weekdays(te, np.expm1(base)))
    early = te.doy.to_numpy() <= EARLY_DAYS
    recent = np.log1p(model_ensemble(tr, te, decay=RECENT_DECAY))
    # Reuse the same fitted forecasts for smoothed and unsmoothed branches.
    base[early] = (1-EARLY_W)*base[early] + EARLY_W*recent[early]
    smoothed[early] = (1-EARLY_W)*smoothed[early] + EARLY_W*recent[early]
    result = .5*smoothed + .5*base
    history = tr[(tr.year != COVID_YEAR) & (tr.dow == 5)]
    for i, row in enumerate(te.itertuples()):
        if row.dow != 5 or not 7 <= row.doy < 36:
            continue
        delta = history.doy.to_numpy()-row.doy
        weights = np.exp(-.2*(row.year-1-history.year.to_numpy()))
        phase = np.digitize(history.doy.to_numpy(), [7, 36])
        weights *= np.exp(-.5*(delta/8)**2)*np.where(phase == 1, 1., .03)
        result[i] = np.average(np.log1p(history.y), weights=weights)
    return np.maximum(np.expm1(result), 0)


def predict(tr, te, baseline=None):
    global FEATURE_MODE
    baseline = _predict_weekends(tr, te) if baseline is None else baseline
    if FEATURE_MODE == 'cat_history':
        try:
            FEATURE_MODE = 'phase_weekends'
            champion = predict(tr, te, baseline=baseline)
        finally:
            FEATURE_MODE = 'cat_history'
        return predict_history_boost(tr, te, champion)
    if FEATURE_MODE not in ('spline', 'phase', 'phase_weekends', 'soft_transition'):
        return baseline
    history = tr[(tr.year != COVID_YEAR) & (tr.dow < 5)]
    profiles = []
    # Average a small fixed family of smooth curves instead of selecting one knot count.
    for knots in (8, 14, 22):
        basis = SplineTransformer(n_knots=knots, degree=3, include_bias=False,
                                  extrapolation='constant')
        train_x = basis.fit_transform(history[['doy']])
        test_x = basis.transform(te[['doy']])
        if FEATURE_MODE in ('phase', 'phase_weekends', 'soft_transition'):
            # Separate campaign phases without forcing continuity at the boundary.
            train_late = (history.doy.to_numpy() >= 36)[:, None]
            test_late = (te.doy.to_numpy() >= 36)[:, None]
            if FEATURE_MODE == 'soft_transition':
                train_late = (1/(1+np.exp(-(history.doy.to_numpy()-36)/4)))[:, None]
                test_late = (1/(1+np.exp(-(te.doy.to_numpy()-36)/4)))[:, None]
            train_x = np.c_[train_x * (1-train_late), train_x * train_late, train_late]
            test_x = np.c_[test_x * (1-test_late), test_x * test_late, test_late]
        train_x = np.c_[train_x, np.eye(7)[history.dow.to_numpy()][:, :4]]
        test_x = np.c_[test_x, np.eye(7)[te.dow.to_numpy()][:, :4]]
        for decay in (.25, .75):
            weights = np.exp(-decay * (te.year.iloc[0]-1-history.year.to_numpy()))
            model = Ridge(alpha=.1 if FEATURE_MODE in ('phase', 'phase_weekends', 'soft_transition') else 1.)
            model.fit(train_x, np.log1p(history.y), sample_weight=weights)
            profiles.append(np.maximum(model.predict(test_x), 0))
    smooth = np.mean(profiles, axis=0)
    result = np.log1p(baseline)
    weekdays = te.dow.to_numpy() < 5
    result[weekdays] = .5*result[weekdays] + .5*smooth[weekdays]
    if FEATURE_MODE in ('phase_weekends', 'soft_transition'):
        history = tr[(tr.year != COVID_YEAR) & (tr.dow == 5)]
        for i, row in enumerate(te.itertuples()):
            if row.dow != 5:
                continue
            if row.doy >= 66:
                end = history[history.doy >= 66]
                weights = np.exp(-.2*(row.year-1-end.year.to_numpy()))
                result[i] = np.average(np.log1p(end.y), weights=weights)
            elif 7 <= row.doy < 36:
                weights = np.exp(-.8*(row.year-1-history.year.to_numpy()))
                phase = np.digitize(history.doy.to_numpy(), [7, 36])
                weights *= np.exp(-.5*((history.doy.to_numpy()-row.doy)/8)**2)
                weights *= np.where(phase == 1, 1., .03)
                result[i] = np.average(np.log1p(history.y), weights=weights)
    return np.maximum(np.expm1(result), 0)


def historical_features(data, year):
    calendar = make_calendar(year)
    history = data[(data.year < year) & data.year.ne(COVID_YEAR)]
    if history.empty:
        raise ValueError('Historical features require an earlier season')
    day, dow = calendar.doy.to_numpy(), calendar.dow.to_numpy()
    x = pd.DataFrame({'day': day, 'dow': dow, 'month': calendar.date.dt.month,
                      'day_month': calendar.date.dt.day, 'late': day >= 36,
                      'early': day <= 10, 'tail': day >= 66})
    distance = history.doy.to_numpy()[None, :] - day[:, None]
    same_dow = history.dow.to_numpy()[None, :] == dow[:, None]
    same_phase = (np.digitize(history.doy, [7, 36, 66])[None, :]
                  == np.digitize(day, [7, 36, 66])[:, None])
    for decay, label in ((.3, 'history'), (1., 'recent')):
        age = np.exp(-decay*(year-1-history.year.to_numpy()))
        for bandwidth in (3, 10):
            weights = np.exp(-.5*(distance/bandwidth)**2)*age*same_dow
            weights *= np.where(same_phase, 1., .05)
            x[f'{label}_{bandwidth}'] = (weights @ np.log1p(history.y.to_numpy()))/weights.sum(axis=1)
    return x.astype(float)


def predict_history_boost(tr, te, champion):
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise RuntimeError('cat_history requires: python -m pip install catboost==1.2.10') from error
    target_year = int(te.year.iloc[0])
    years = [int(y) for y in sorted(tr.year.unique())
             if y != COVID_YEAR and y > tr.year.min() and y < target_year]
    x = pd.concat([historical_features(tr, year) for year in years], ignore_index=True)
    train = tr[tr.year.isin(years)].sort_values('date')
    if len(x) != len(train):
        raise ValueError('Historical feature rows must align with complete seasons')
    target = np.log1p(train.y.to_numpy()) - x.history_10.to_numpy()
    weights = np.exp(.25*(train.year.to_numpy()-train.year.max()))
    model = CatBoostRegressor(iterations=500, depth=4, learning_rate=.035, l2_leaf_reg=10,
                              loss_function='RMSE', random_seed=42, thread_count=2,
                              verbose=False, allow_writing_files=False)
    model.fit(x, target, sample_weight=weights)
    xt = historical_features(tr, target_year).iloc[te.doy.to_numpy().astype(int)]
    log_prediction = np.maximum(model.predict(xt)+xt.history_10.to_numpy(), 0.)
    return np.expm1(.75*np.log1p(champion)+.25*log_prediction)


def load_daily(path):
    raw = pd.read_csv(path, parse_dates=['date'])
    column = 'registrations' if 'registrations' in raw else 'target'
    data = raw.loc[raw.in_submission_window.eq(1), ['date',column]]
    data = add_calendar(data.rename(columns={column:'y'}).sort_values('date').reset_index(drop=True))
    if data.date.duplicated().any() or not data.groupby('year').size().eq(73).all():
        raise ValueError('Expected 73 unique daily dates per season')
    if not np.isfinite(data.y).all() or data.y.lt(0).any():
        raise ValueError('Invalid historical counts')
    return data


def main():
    global FEATURE_MODE
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily', type=Path, default=here/'data/daily_registered.csv')
    parser.add_argument('--out-dir', type=Path, default=here/'output')
    parser.add_argument('--model', choices=['calendar', 'baseline', 'spline', 'phase', 'phase_weekends', 'soft_transition', 'cat_history'], default='phase_weekends',
                        help='phase_weekends: current best public score; other modes: previous versions')
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--validate', dest='validate', action='store_true')
    validation.add_argument('--no-validate', dest='validate', action='store_false',
                            help='Skip historical RMSLE for a faster forecast')
    parser.set_defaults(validate=True)
    args = parser.parse_args()
    FEATURE_MODE = args.model
    print(f'Model: {FEATURE_MODE}', flush=True)
    data = load_daily(args.daily)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.validate:
        print('Historical RMSLE: train on earlier years, test on the displayed year.', flush=True)
        rows=[]
        for year in (2018,2019,2021,2022,2023,2020):
            test=data[data.year.eq(year)]
            train = data[data.year.lt(year)]
            baseline = _predict_weekends(train, test)
            prediction=predict(train, test, baseline=baseline)
            score=rmsle(test.y,prediction)
            row = {'year':year,'rmsle':score}
            if args.model in ('spline', 'phase', 'phase_weekends', 'soft_transition', 'cat_history'):
                row['baseline_rmsle'] = rmsle(test.y, baseline)
                row['delta'] = score-row['baseline_rmsle']
                print(f'{year}: baseline={row["baseline_rmsle"]:.5f}, {args.model}={score:.5f}, delta={row["delta"]:+.5f}', flush=True)
                if args.model == 'phase_weekends':
                    FEATURE_MODE = 'phase'
                    reference = predict(train, test, baseline=baseline)
                    FEATURE_MODE = args.model
                    row['phase_rmsle'] = rmsle(test.y, reference)
                    row['delta_vs_phase'] = score-row['phase_rmsle']
                    print(f'  phase={row["phase_rmsle"]:.5f}, delta_vs_phase={row["delta_vs_phase"]:+.5f}', flush=True)
                if args.model in ('soft_transition', 'cat_history'):
                    FEATURE_MODE = 'phase_weekends'
                    reference = predict(train, test, baseline=baseline)
                    FEATURE_MODE = args.model
                    row['champion_rmsle'] = rmsle(test.y, reference)
                    row['delta_vs_champion'] = score-row['champion_rmsle']
                    print(f'  champion={row["champion_rmsle"]:.5f}, delta_vs_champion={row["delta_vs_champion"]:+.5f}', flush=True)
            else:
                print(f'{year}: RMSLE={score:.5f}',flush=True)
            rows.append(row)
        scores=pd.DataFrame(rows)
        label = 'weekends' if args.model == 'baseline' else args.model
        validation_name = f'validation_{label}.csv'
        scores.to_csv(args.out_dir/validation_name,index=False)
        mean=scores[scores.year.isin([2018,2019,2021,2022])].rmsle.mean()
        print(f'Mean yearly RMSLE (2018/2019/2021/2022): {mean:.5f}',flush=True)
        print(f'2023 RMSLE: {scores.loc[scores.year.eq(2023), "rmsle"].iloc[0]:.5f}', flush=True)
        if args.model in ('spline', 'phase', 'phase_weekends', 'soft_transition', 'cat_history'):
            ordinary = scores[scores.year.ne(COVID_YEAR)]
            print(f'Mean five non-2020 years: baseline={ordinary.baseline_rmsle.mean():.5f}, {args.model}={ordinary.rmsle.mean():.5f}', flush=True)
            if args.model == 'phase_weekends':
                print(f'Mean five non-2020 years: phase={ordinary.phase_rmsle.mean():.5f}, candidate={ordinary.rmsle.mean():.5f}', flush=True)
            if args.model in ('soft_transition', 'cat_history'):
                print(f'Mean five non-2020 years: champion={ordinary.champion_rmsle.mean():.5f}, candidate={ordinary.rmsle.mean():.5f}', flush=True)
            recent = scores[scores.year.isin([2021,2022,2023])]
            print(f'Mean 2021-2023: baseline={recent.baseline_rmsle.mean():.5f}, {args.model}={recent.rmsle.mean():.5f}', flush=True)
        print('2020 is reported separately as a stress year, excluded from this mean.', flush=True)
        print('2024 RMSLE: unknown locally; actual 2024 counts are hidden.', flush=True)
    calendar=make_calendar(2024)
    forecast=predict(data[data.year.lt(2024)],calendar)
    output=calendar[['date']].copy()
    output['forecast']=np.round(forecast,2)
    if not np.isfinite(forecast).all() or (forecast<0).any():
        raise ValueError('Invalid forecast')
    label = 'weekends' if args.model == 'baseline' else args.model
    filename = f'submission_{label}_2024.csv'
    path=args.out_dir/filename
    output.to_csv(path,index=False)
    print(f'Saved: {path}',flush=True)
    print(f'SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
