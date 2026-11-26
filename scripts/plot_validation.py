"""Visualize locally generated backtests, without publishing competition data."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('artifacts/v3'))
    args = parser.parse_args()
    scores = pd.read_csv(args.results / 'validation.csv').sort_values('year')
    predictions = pd.read_csv(args.results / 'historical_predictions.csv', parse_dates=['date'])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
    scores.set_index('year')[['phase_weekends', 'cat_history', 'v3']].plot.bar(ax=axes[0])
    axes[0].set(ylabel='RMSLE (lower is better)', title='Historical backtests — development years, not an untouched test')
    axes[0].tick_params(axis='x', rotation=0)
    latest = predictions[predictions.date.dt.year.eq(predictions.date.dt.year.max())]
    actual = latest.drop_duplicates('date').sort_values('date')
    axes[1].plot(actual.date, actual.actual, color='black', label='Actual registrations')
    for model, group in latest.groupby('model'):
        axes[1].plot(group.date, group.forecast, label=model, alpha=.8)
    axes[1].set(ylabel='Daily registrations', title='Latest historical season (2023)')
    axes[1].legend()
    fig.savefig(args.results / 'validation.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
