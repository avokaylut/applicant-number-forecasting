"""Verified reproduction of the repository v3; no leaderboard inputs."""
import argparse
import hashlib
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
SEASON_START = '06-20'
SEASON_END = '08-31'
COVID_YEAR = 2020
SUNDAY_VALUE = 0.3
SEED = 0
FEATURE_MODE = 'phase_weekends'

def rmsle(actual, pred):
    """Метрика соревнования."""
    actual, pred = (np.asarray(actual, float), np.asarray(pred, float))
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))

def add_calendar(df):
    df = df.copy()
    df['year'] = df.date.dt.year
    df['dow'] = df.date.dt.dayofweek
    df['doy'] = (df.date - pd.to_datetime(df.year.astype(str) + '-' + SEASON_START)).dt.days
    return df

def make_calendar(year):
    d = pd.DataFrame({'date': pd.date_range(f'{year}-{SEASON_START}', f'{year}-{SEASON_END}')})
    return add_calendar(d)

def _features(d):
    base = np.c_[d.doy, d.dow, d.dow == 6, d.dow == 5, d.doy >= 36]
    if FEATURE_MODE != 'calendar':
        return base
    dates = d.date.to_numpy(dtype='datetime64[D]')
    starts = pd.to_datetime(d.year.astype(str) + '-06-20').to_numpy(dtype='datetime64[D]')
    start_dow = pd.to_datetime(starts).dayofweek.to_numpy()
    weekday_index = np.busday_count(starts, dates)
    saturday_number = np.maximum(0, (d.doy.to_numpy() - (5 - start_dow) % 7) // 7 + 1)
    first_weekday = np.where(start_dow < 5, 0, 7 - start_dow)
    first_monday = (7 - start_dow) % 7
    return np.c_[base, weekday_index, saturday_number, d.doy.to_numpy() == first_weekday, (d.doy.to_numpy() == first_monday) & (d.doy.to_numpy() < 7)]

def _gbm(depth=3, it=150, lr=0.05, leaf=8):
    return HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=leaf, random_state=SEED)

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
    for y, g in t.groupby('year'):
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
    t['l'] = np.log1p(t.y) - [E.loc[d, y] if d < 6 else 0 for d, y in zip(t.dow, t.year)]
    f = t[t.dow < 5]
    w = np.exp(decay * (f.year - f.year.max()).values)
    w = np.where(f.year == COVID_YEAR, 0, w)
    m = _gbm().fit(f[['doy']].values, f.l.values, sample_weight=w)
    base = m.predict(te[['doy']].values)
    de = te.dow.map(lambda d: Et[d] if d < 6 else 0).values
    p = np.expm1(base + de).clip(0)
    return np.where(te.dow.values == 6, SUNDAY_VALUE, p)
ENS_K, ENS_DECAY, ENS_SAT_K = (8, 0.4, 2)
EARLY_DAYS, EARLY_W, RECENT_DECAY = (10, 0.5, 5.0)
DECAYS = (0.25, 0.6, 1.2)
BAG_CONFIGS = (dict(depth=3, it=150), dict(depth=2, it=250), dict(depth=4, it=120, leaf=12), dict(depth=3, it=100, lr=0.08, l2=1.0), dict(depth=3, it=150, decay=0.2), dict(depth=3, it=150, decay=0.7, k=6))

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
    m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=leaf, l2_regularization=l2, random_state=SEED)
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
    base[early] = (1 - EARLY_W) * base[early] + EARLY_W * recent[early]
    smoothed[early] = (1 - EARLY_W) * smoothed[early] + EARLY_W * recent[early]
    result = 0.5 * smoothed + 0.5 * base
    history = tr[(tr.year != COVID_YEAR) & (tr.dow == 5)]
    for i, row in enumerate(te.itertuples()):
        if row.dow != 5 or not 7 <= row.doy < 36:
            continue
        delta = history.doy.to_numpy() - row.doy
        weights = np.exp(-0.2 * (row.year - 1 - history.year.to_numpy()))
        phase = np.digitize(history.doy.to_numpy(), [7, 36])
        weights *= np.exp(-0.5 * (delta / 8) ** 2) * np.where(phase == 1, 1.0, 0.03)
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
    for knots in (8, 14, 22):
        basis = SplineTransformer(n_knots=knots, degree=3, include_bias=False, extrapolation='constant')
        train_x = basis.fit_transform(history[['doy']])
        test_x = basis.transform(te[['doy']])
        if FEATURE_MODE in ('phase', 'phase_weekends', 'soft_transition'):
            train_late = (history.doy.to_numpy() >= 36)[:, None]
            test_late = (te.doy.to_numpy() >= 36)[:, None]
            if FEATURE_MODE == 'soft_transition':
                train_late = (1 / (1 + np.exp(-(history.doy.to_numpy() - 36) / 4)))[:, None]
                test_late = (1 / (1 + np.exp(-(te.doy.to_numpy() - 36) / 4)))[:, None]
            train_x = np.c_[train_x * (1 - train_late), train_x * train_late, train_late]
            test_x = np.c_[test_x * (1 - test_late), test_x * test_late, test_late]
        train_x = np.c_[train_x, np.eye(7)[history.dow.to_numpy()][:, :4]]
        test_x = np.c_[test_x, np.eye(7)[te.dow.to_numpy()][:, :4]]
        for decay in (0.25, 0.75):
            weights = np.exp(-decay * (te.year.iloc[0] - 1 - history.year.to_numpy()))
            model = Ridge(alpha=0.1 if FEATURE_MODE in ('phase', 'phase_weekends', 'soft_transition') else 1.0)
            model.fit(train_x, np.log1p(history.y), sample_weight=weights)
            profiles.append(np.maximum(model.predict(test_x), 0))
    smooth = np.mean(profiles, axis=0)
    result = np.log1p(baseline)
    weekdays = te.dow.to_numpy() < 5
    result[weekdays] = 0.5 * result[weekdays] + 0.5 * smooth[weekdays]
    if FEATURE_MODE in ('phase_weekends', 'soft_transition'):
        history = tr[(tr.year != COVID_YEAR) & (tr.dow == 5)]
        for i, row in enumerate(te.itertuples()):
            if row.dow != 5:
                continue
            if row.doy >= 66:
                end = history[history.doy >= 66]
                weights = np.exp(-0.2 * (row.year - 1 - end.year.to_numpy()))
                result[i] = np.average(np.log1p(end.y), weights=weights)
            elif 7 <= row.doy < 36:
                weights = np.exp(-0.8 * (row.year - 1 - history.year.to_numpy()))
                phase = np.digitize(history.doy.to_numpy(), [7, 36])
                weights *= np.exp(-0.5 * ((history.doy.to_numpy() - row.doy) / 8) ** 2)
                weights *= np.where(phase == 1, 1.0, 0.03)
                result[i] = np.average(np.log1p(history.y), weights=weights)
    return np.maximum(np.expm1(result), 0)

def historical_features(data, year):
    calendar = make_calendar(year)
    history = data[(data.year < year) & data.year.ne(COVID_YEAR)]
    if history.empty:
        raise ValueError('Historical features require an earlier season')
    day, dow = (calendar.doy.to_numpy(), calendar.dow.to_numpy())
    x = pd.DataFrame({'day': day, 'dow': dow, 'month': calendar.date.dt.month, 'day_month': calendar.date.dt.day, 'late': day >= 36, 'early': day <= 10, 'tail': day >= 66})
    distance = history.doy.to_numpy()[None, :] - day[:, None]
    same_dow = history.dow.to_numpy()[None, :] == dow[:, None]
    same_phase = np.digitize(history.doy, [7, 36, 66])[None, :] == np.digitize(day, [7, 36, 66])[:, None]
    for decay, label in ((0.3, 'history'), (1.0, 'recent')):
        age = np.exp(-decay * (year - 1 - history.year.to_numpy()))
        for bandwidth in (3, 10):
            weights = np.exp(-0.5 * (distance / bandwidth) ** 2) * age * same_dow
            weights *= np.where(same_phase, 1.0, 0.05)
            x[f'{label}_{bandwidth}'] = weights @ np.log1p(history.y.to_numpy()) / weights.sum(axis=1)
    return x.astype(float)

def predict_history_boost(tr, te, champion):
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise RuntimeError('cat_history requires: python -m pip install catboost==1.2.10') from error
    target_year = int(te.year.iloc[0])
    years = [int(y) for y in sorted(tr.year.unique()) if y != COVID_YEAR and y > tr.year.min() and (y < target_year)]
    x = pd.concat([historical_features(tr, year) for year in years], ignore_index=True)
    train = tr[tr.year.isin(years)].sort_values('date')
    if len(x) != len(train):
        raise ValueError('Historical feature rows must align with complete seasons')
    target = np.log1p(train.y.to_numpy()) - x.history_10.to_numpy()
    weights = np.exp(0.25 * (train.year.to_numpy() - train.year.max()))
    model = CatBoostRegressor(iterations=500, depth=4, learning_rate=0.035, l2_leaf_reg=10, loss_function='RMSE', random_seed=42, thread_count=2, verbose=False, allow_writing_files=False)
    model.fit(x, target, sample_weight=weights)
    xt = historical_features(tr, target_year).iloc[te.doy.to_numpy().astype(int)]
    log_prediction = np.maximum(model.predict(xt) + xt.history_10.to_numpy(), 0.0)
    return np.expm1(0.75 * np.log1p(champion) + 0.25 * log_prediction)

from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor
import json
import platform


def phase_design(frame, basis, cut=36):
    splines = basis.transform(frame[['doy']].to_numpy())
    late = (frame.doy.to_numpy() >= cut)[:, None]
    phase = np.digitize(frame.doy.to_numpy(), [7, cut, 55])
    indicators = np.zeros((len(frame), 28))
    indicators[np.arange(len(frame)), frame.dow.to_numpy()*4+phase] = 1
    return np.c_[splines*(1-late), splines*late, indicators]


def saturday_ridge(train, test):
    year = int(test.year.iloc[0])
    history = train[train.year.ne(COVID_YEAR)]
    basis = SplineTransformer(n_knots=8, degree=3, extrapolation='constant').fit(np.arange(73)[:, None])
    weights = np.exp(-.2*(year-1-history.year.to_numpy()))
    model = Ridge(alpha=.1).fit(phase_design(history, basis), np.log1p(history.y), sample_weight=weights)
    return np.maximum(model.predict(phase_design(test, basis)), 0.)


def hurdle_profile(train, test):
    """Reproduce v3's approximate hurdle model, not an exact truncated-NB MLE."""
    year = int(test.year.iloc[0])
    history = train[train.year.ne(COVID_YEAR)]
    weights = np.exp(-.3*(year-1-history.year.to_numpy()))
    history_phase = np.digitize(history.doy, [7, 36, 55])
    test_phase = np.digitize(test.doy, [7, 36, 55])
    probability = []
    for dow, phase in zip(test.dow, test_phase):
        if dow < 5:
            probability.append(1.)
            continue
        same = (history.dow.to_numpy() == dow) & (history_phase == phase)
        prior = .1 if dow == 6 else .7
        probability.append((weights[same] @ (history.y.to_numpy()[same] > 0)+2*prior)/(weights[same].sum()+2))
    positive = history.y.gt(0).to_numpy()
    basis = SplineTransformer(n_knots=10, extrapolation='constant').fit(np.arange(73)[:, None])
    model = PoissonRegressor(alpha=1e-3, max_iter=3000).fit(
        phase_design(history[positive], basis), history.y[positive], sample_weight=weights[positive])
    fitted = model.predict(phase_design(history[positive], basis))
    dispersion = max(np.average(((history.y[positive]-fitted)**2-fitted)/fitted**2,
                                weights=weights[positive]), 1e-3)
    intensity = np.maximum(model.predict(phase_design(test, basis)), 1e-3)
    conditional_log = np.empty(len(test))
    r = 1/dispersion
    for i, mu in enumerate(intensity):
        support = np.arange(int(6*mu+60))
        mass = nbinom.pmf(support, r, r/(r+mu))
        conditional_log[i] = mass[1:] @ np.log1p(support[1:])/max(1-mass[0], 1e-12)
    return np.array(probability)*conditional_log


def forecast_v3(train, test):
    global FEATURE_MODE
    if train.empty or test.year.nunique() != 1 or train.year.max() >= test.year.iloc[0]:
        raise ValueError('Forecast must use strictly earlier training years')
    expected = make_calendar(int(test.year.iloc[0]))
    if not np.array_equal(test.date.to_numpy(), expected.date.to_numpy()):
        raise ValueError('Forecast requires all 73 season dates in chronological order')
    # Only known calendar fields of the test season enter model predictions.
    calendar = expected
    FEATURE_MODE = 'phase_weekends'
    champion = predict(train, calendar)
    cat_mix = predict_history_boost(train, calendar, champion)
    result = .5*np.log1p(champion)+.5*np.log1p(cat_mix)
    saturday = calendar.dow.eq(5).to_numpy()
    regular = saturday & calendar.doy.ge(7).to_numpy()
    ridge = saturday_ridge(train, calendar)
    result[regular] = .7*result[regular]+.3*ridge[regular]
    first = saturday & calendar.doy.lt(7).to_numpy()
    hurdle = hurdle_profile(train, calendar)
    result[first] = hurdle[first]
    return {'v3': np.maximum(np.expm1(result), 0.),
            'phase_weekends': champion, 'cat_history': cat_mix}


def read_history(csv_path, raw_path=None):
    if raw_path is not None:
        raw = pd.read_excel(raw_path)
        dates = pd.to_datetime(raw['reg_ts'], errors='raise').dt.normalize()
        if dates.isna().any():
            raise ValueError('Missing raw registration timestamps')
        counts = raw.groupby(dates).size()
        index = pd.DatetimeIndex(np.concatenate([make_calendar(y).date.to_numpy() for y in range(2014, 2024)]))
        frame = counts.reindex(index, fill_value=0).rename_axis('date').reset_index(name='y')
    else:
        frame = pd.read_csv(csv_path, parse_dates=['date'])
        if 'y' not in frame:
            frame = frame.rename(columns={'target': 'y'})
        frame = frame[['date', 'y']]
    frame = add_calendar(frame.sort_values('date').reset_index(drop=True))
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError('Invalid or duplicate dates')
    if not np.isfinite(frame.y).all() or frame.y.lt(0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if set(frame.year.unique()) != set(range(2014, 2024)):
        raise ValueError('Expected complete history for 2014-2023, not 2024 actuals')
    for year, group in frame.groupby('year'):
        if not np.array_equal(group.date.to_numpy(), make_calendar(year).date.to_numpy()):
            raise ValueError(f'Incomplete season {year}')
    return frame


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Reproducible v3: historical data only, no leaderboard fitting')
    parser.add_argument('--data', type=Path, default=here/'data/daily_registered.csv')
    parser.add_argument('--raw', type=Path, help='Optional original registered_2014_2023.xlsx instead of prepared CSV')
    parser.add_argument('--out-dir', type=Path, default=here/'output')
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()
    history = read_history(args.data, args.raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_validate:
        rows = []
        predictions = []
        for year in (2017, 2018, 2019, 2021, 2022, 2023, 2020):
            train = history[history.year.lt(year)]
            test = history[history.year.eq(year)].reset_index(drop=True)
            variants = forecast_v3(train, test)
            row = {'year': year, **{name: rmsle(test.y, p) for name, p in variants.items()}}
            rows.append(row)
            print(f'{year}: v3={row["v3"]:.6f}; previous={row["phase_weekends"]:.6f}', flush=True)
            for name, p in variants.items():
                predictions.extend({'date': date, 'model': name, 'actual': actual, 'forecast': value}
                                   for date, actual, value in zip(test.date, test.y, p))
        scores = pd.DataFrame(rows)
        scores.to_csv(args.out_dir/'validation.csv', index=False)
        pd.DataFrame(predictions).to_csv(args.out_dir/'historical_predictions.csv', index=False)
        ordinary = scores[scores.year.ne(COVID_YEAR)]
        print('Mean yearly RMSLE, six non-2020 years:', ordinary.drop(columns='year').mean().round(6).to_dict(), flush=True)
        print('2020 is a separate stress test. These historical years were used in model development.', flush=True)
        print('2024 RMSLE is unknown locally: actual counts are hidden.', flush=True)
    calendar = make_calendar(2024)
    forecast = forecast_v3(history, calendar)['v3']
    if not np.isfinite(forecast).all() or np.any(forecast < 0):
        raise ValueError('Invalid forecast')
    result = calendar[['date']].copy()
    result['forecast'] = np.round(forecast, 2)
    destination = args.out_dir/'submission_v3_2024.csv'
    result.to_csv(destination, index=False)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    import sklearn, catboost, scipy
    metadata = {'model': 'verified_v3', 'rows': len(result), 'sha256': digest,
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'catboost': catboost.__version__, 'scipy': scipy.__version__,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_sha256': hashlib.sha256((args.raw or args.data).read_bytes()).hexdigest()}
    (args.out_dir/'run_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved: {destination}\nSHA256: {digest}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
