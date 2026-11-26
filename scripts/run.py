"""Strict input checks around the two unchanged historical solutions."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['clean', 'v3'], required=True)
    parser.add_argument('--daily', type=Path, default=Path('data/processed/daily_registered.csv'))
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('extended', ROOT / 'solutions/extended_v3.py')
    extended = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extended)
    # The clean loader checks row counts, but not exact season coverage.
    extended.read_history(args.daily)
    output = args.out_dir or Path('artifacts') / args.version
    if args.version == 'clean':
        command = [sys.executable, str(ROOT / 'solutions/forecast_clean.py'),
                   '--daily', str(args.daily), '--model', 'phase_weekends']
    else:
        command = [sys.executable, str(ROOT / 'solutions/extended_v3.py'), '--data', str(args.daily)]
    command += ['--out-dir', str(output)]
    if args.no_validate:
        command += ['--no-validate']
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
