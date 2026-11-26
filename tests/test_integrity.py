"""Synthetic checks only: no competition data required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'solutions' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3 = load('extended_v3')
        cls.clean = load('forecast_clean')
        cls.history = pd.concat([cls.v3.make_calendar(y) for y in range(2014, 2024)], ignore_index=True)
        cls.history['y'] = np.where(cls.history.dow.eq(6), 0,
                                   20 + cls.history.doy % 17 + (cls.history.year-2014)*2)

    def test_calendar_and_metric(self):
        self.assertEqual(len(self.v3.make_calendar(2024)), 73)
        self.assertAlmostEqual(self.v3.rmsle([0, 3], [0, 3]), 0)
        self.assertAlmostEqual(self.v3.rmsle([0], [1]), np.log(2))

    def test_historical_features_ignore_current_and_future_targets(self):
        original = self.v3.historical_features(self.history, 2021)
        changed = self.history.copy()
        changed.loc[changed.year.ge(2021), 'y'] = 999999
        pd.testing.assert_frame_equal(original, self.v3.historical_features(changed, 2021))

    def test_strict_reader_rejects_wrong_calendar(self):
        frame = self.history[['date', 'y']].copy()
        frame.loc[0, 'date'] = pd.Timestamp('2014-06-19')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'daily.csv'
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Incomplete season'):
                self.v3.read_history(path)

    def test_v3_rejects_overlapping_years(self):
        with self.assertRaisesRegex(ValueError, 'strictly earlier'):
            self.v3.forecast_v3(self.history, self.v3.make_calendar(2023))

    def test_forecasts_ignore_test_targets_and_share_base(self):
        train = self.history[self.history.year.lt(2017)]
        test = self.history[self.history.year.eq(2017)].reset_index(drop=True)
        with threadpool_limits(limits=2):
            original = self.v3.forecast_v3(train, test)
            changed = test.copy()
            changed['y'] = 999999
            repeated = self.v3.forecast_v3(train, changed)
            self.clean.FEATURE_MODE = 'phase_weekends'
            clean = self.clean.predict(train, test)
            clean_changed = self.clean.predict(train, changed)
        np.testing.assert_allclose(clean, original['phase_weekends'], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(clean, clean_changed)
        for name, values in original.items():
            np.testing.assert_array_equal(values, repeated[name])
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values >= 0).all())


if __name__ == '__main__':
    unittest.main()
