"""Prepare local daily registration counts; never write applicant identifiers."""
import argparse
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def prepare(archive, destination):
    with zipfile.ZipFile(archive) as z:
        candidates = []
        for entry in z.infolist():
            if entry.filename.startswith('__MACOSX/'):
                continue
            name = entry.filename
            try:
                name = name.encode('cp437').decode('utf-8')
            except (UnicodeError, LookupError):
                pass
            if Path(name).name.startswith('Зарегистрированы') and name.endswith('.xlsx'):
                candidates.append(entry)
        if len(candidates) != 1:
            raise ValueError('Expected exactly one registered-applicants workbook')
        raw = pd.read_excel(io.BytesIO(z.read(candidates[0])), usecols=['reg_ts'])
    dates = pd.to_datetime(raw.reg_ts, errors='raise').dt.normalize()
    if dates.isna().any() or not dates.dt.year.between(2014, 2023).all():
        raise ValueError('Missing timestamps or dates outside 2014–2023')
    index = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f'{y}-06-20', f'{y}-08-31').to_numpy() for y in range(2014, 2024)]))
    counts = dates.value_counts().reindex(index, fill_value=0).sort_index()
    daily = counts.rename_axis('date').reset_index(name='target')
    # Both original loaders accept this schema.
    daily['in_submission_window'] = 1
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(destination, index=False)
    print(f'Saved {len(daily)} daily rows; excluded {len(raw)-int(counts.sum())} registrations outside season.')
    return daily


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/processed/daily_registered.csv'))
    args = parser.parse_args()
    prepare(args.archive, args.out)
