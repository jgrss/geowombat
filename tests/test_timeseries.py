"""Tests for geowombat.core.timeseries (input hygiene helpers)."""

import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import geowombat as gw
from geowombat.core.timeseries import (
    GeoWombatInputWarning,
    TimeSeriesFiller,
    audit_input,
    fill_time,
    mask_clouds,
    missing_report,
    to_float_with_nan,
)
from geowombat.data import l8_224078_20200518

NT, NB, NY, NX = 4, 2, 10, 10


def _make_da(nodata=None, with_time=True, seed=0):
    rng = np.random.default_rng(seed)
    shape = (NT, NB, NY, NX) if with_time else (NB, NY, NX)
    arr = rng.uniform(0.1, 1.0, shape).astype('float32')
    coords = {
        'band': [f'b{i}' for i in range(NB)],
        'y': np.arange(NY)[::-1] * 10.0,
        'x': np.arange(NX) * 10.0,
    }
    dims = ('band', 'y', 'x')
    if with_time:
        coords['time'] = np.arange(NT)
        dims = ('time',) + dims
    attrs = {'res': (10.0, 10.0)}
    if nodata is not None:
        attrs['nodatavals'] = (nodata,) * NB
    return xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)


def _caught(func, *args, **kwargs):
    """Run func and return the list of GeoWombatInputWarning messages."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = func(*args, **kwargs)
    msgs = [
        str(w.message)
        for w in rec
        if issubclass(w.category, GeoWombatInputWarning)
    ]
    return out, msgs


class TestToFloatWithNan(unittest.TestCase):
    def test_nodata_value_becomes_nan_and_attrs_kept(self):
        da_ = _make_da(nodata=9999.0)
        da_[3, 0, :1, :5] = 9999.0
        out = to_float_with_nan(da_)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(int(np.isnan(out.values).sum()), 5)
        self.assertTrue(np.isnan(out.values[3, 0, 0, :5]).all())
        self.assertEqual(out.attrs['res'], (10.0, 10.0))

    def test_nodata_mask_coord_applies_to_all_time_and_bands(self):
        da_ = _make_da()
        mask = np.zeros((NY, NX), dtype=bool)
        mask[2, 3] = True
        da_ = da_.assign_coords(
            _nodata_mask=xr.DataArray(mask, dims=('y', 'x'))
        )
        out = to_float_with_nan(da_)
        self.assertTrue(np.isnan(out.values[:, :, 2, 3]).all())
        self.assertEqual(int(np.isnan(out.values).sum()), NT * NB)

    def test_no_nodata_declared_is_a_plain_cast(self):
        da_ = _make_da()
        out = to_float_with_nan(da_)
        self.assertFalse(np.isnan(out.values).any())


class TestMissingReport(unittest.TestCase):
    def test_fractions_per_time_and_band(self):
        da_ = _make_da(nodata=9999.0)
        da_[1, 0, :4, :] = np.nan  # 40 of 100 pixels
        da_[2, 1, 0, :10] = 0.0  # 10 of 100 pixels
        da_[3, 0, 0, :5] = 9999.0  # 5 of 100 pixels
        rep = missing_report(da_)
        self.assertIsInstance(rep, pd.DataFrame)
        self.assertEqual(len(rep), NT * NB)
        self.assertEqual(
            list(rep.columns),
            [
                'time',
                'band',
                'n_pixels',
                'frac_nan',
                'frac_nodata',
                'frac_zero',
                'frac_missing',
            ],
        )
        rep = rep.set_index(['time', 'band'])
        self.assertEqual(rep.loc[(1, 'b0'), 'n_pixels'], 100)
        self.assertAlmostEqual(rep.loc[(1, 'b0'), 'frac_nan'], 0.4)
        self.assertAlmostEqual(rep.loc[(1, 'b0'), 'frac_missing'], 0.4)
        self.assertAlmostEqual(rep.loc[(2, 'b1'), 'frac_zero'], 0.1)
        self.assertAlmostEqual(rep.loc[(2, 'b1'), 'frac_missing'], 0.0)
        self.assertAlmostEqual(rep.loc[(3, 'b0'), 'frac_nodata'], 0.05)
        self.assertAlmostEqual(rep.loc[(3, 'b0'), 'frac_missing'], 0.05)
        self.assertAlmostEqual(rep.loc[(0, 'b1'), 'frac_missing'], 0.0)

    def test_zero_declared_as_nodata_counts_as_nodata_not_zero(self):
        da_ = _make_da(nodata=0)
        da_[0, 0, 0, :10] = 0.0
        rep = missing_report(da_).set_index(['time', 'band'])
        self.assertAlmostEqual(rep.loc[(0, 'b0'), 'frac_nodata'], 0.1)
        self.assertAlmostEqual(rep.loc[(0, 'b0'), 'frac_zero'], 0.0)

    def test_footprint_nodata_is_excluded_from_fractions(self):
        # Pixels missing in every (time, band) are outside the footprint
        # and must not count as gaps; one extra gap at time 1 still does.
        da_ = _make_da()
        da_[:, :, :5, :] = np.nan  # 50 footprint pixels
        da_[1, 0, 5, :] = np.nan  # 10 real gaps among the 50 valid
        rep = missing_report(da_).set_index(['time', 'band'])
        self.assertEqual(rep.loc[(0, 'b0'), 'n_pixels'], 50)
        self.assertAlmostEqual(rep.loc[(0, 'b0'), 'frac_missing'], 0.0)
        self.assertAlmostEqual(rep.loc[(1, 'b0'), 'frac_missing'], 0.2)

    def test_nodata_mask_coord_counts_as_nodata(self):
        # I4: the accessor report must see what the classifiers see
        da_ = _make_da()
        mask = np.zeros((NY, NX), dtype=bool)
        mask[0, :3] = True
        da_ = da_.assign_coords(
            _nodata_mask=xr.DataArray(mask, dims=('y', 'x'))
        )
        da_[1, 0, 5, :5] = np.nan  # 5 real gaps among 97 valid pixels
        rep = missing_report(da_).set_index(['time', 'band'])
        self.assertEqual(rep.loc[(0, 'b0'), 'n_pixels'], 97)
        self.assertAlmostEqual(rep.loc[(1, 'b0'), 'frac_missing'], 5 / 97)

    def test_report_without_time(self):
        da_ = _make_da(with_time=False)
        da_[1, :5, :] = np.nan
        rep = missing_report(da_)
        self.assertEqual(len(rep), NB)
        self.assertTrue(rep['time'].isna().all())
        self.assertAlmostEqual(
            rep.set_index('band').loc['b1', 'frac_nan'], 0.5
        )


class TestAuditInput(unittest.TestCase):
    def test_warns_on_heavy_missingness(self):
        da_ = _make_da()
        da_[1, 0, :4, :] = np.nan
        rep, msgs = _caught(audit_input, da_, stage='fit', fill='train_median')
        self.assertIsInstance(rep, pd.DataFrame)
        self.assertTrue(any('missing' in m.lower() for m in msgs), msgs)
        self.assertTrue(any('train_median' in m for m in msgs), msgs)

    def test_clean_data_with_cloud_tag_is_silent(self):
        da_ = _make_da()
        da_.attrs['cloud_masked'] = True
        _, msgs = _caught(audit_input, da_, stage='fit')
        self.assertEqual(msgs, [])

    def test_all_missing_slice_at_fit_raises(self):
        da_ = _make_da()
        da_[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'time'):
            audit_input(da_, stage='fit', policy='warn')

    def test_warns_on_suspicious_zeros(self):
        da_ = _make_da()
        da_.attrs['cloud_masked'] = True
        da_[0, 1, 0, :2] = 0.0  # 2% zeros, 0 not declared nodata
        _, msgs = _caught(audit_input, da_, stage='fit')
        self.assertTrue(any('zero' in m.lower() for m in msgs), msgs)

    def test_zeros_declared_nodata_do_not_trigger_zero_warning(self):
        da_ = _make_da(nodata=0)
        da_.attrs['cloud_masked'] = True
        da_[0, 1, 0, :2] = 0.0
        _, msgs = _caught(audit_input, da_, stage='fit')
        self.assertFalse(any('zero' in m.lower() for m in msgs), msgs)

    def test_predict_shift_warning_names_time_step(self):
        clean = _make_da()
        clean.attrs['cloud_masked'] = True
        ref = audit_input(clean, stage='fit', policy='warn')
        pred = _make_da(seed=1)
        pred[1, :, :4, :] = np.nan
        _, msgs = _caught(
            audit_input,
            pred,
            stage='predict',
            reference=ref,
            fill='train_median',
        )
        shift = [m for m in msgs if 'training' in m.lower()]
        self.assertTrue(shift, msgs)
        self.assertIn('1', shift[0])

    def test_predict_without_shift_is_silent(self):
        clean = _make_da()
        clean.attrs['cloud_masked'] = True
        ref = audit_input(clean, stage='fit', policy='warn')
        pred = _make_da(seed=1)
        pred.attrs['cloud_masked'] = True
        _, msgs = _caught(audit_input, pred, stage='predict', reference=ref)
        self.assertEqual(msgs, [])

    def test_unmasked_cloud_heuristic(self):
        da_ = _make_da()  # 4 dates, no gaps, no tag
        _, msgs = _caught(audit_input, da_, stage='fit')
        self.assertTrue(any('cloud' in m.lower() for m in msgs), msgs)
        da_.attrs['cloud_masked'] = True
        _, msgs = _caught(audit_input, da_, stage='fit')
        self.assertFalse(any('cloud' in m.lower() for m in msgs), msgs)

    def test_all_missing_slice_guard_lives_in_the_filler(self):
        # I1: the guarantee must not depend on the audit policy
        da_ = _make_da()
        da_[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'time'):
            TimeSeriesFiller('train_median').fit(da_)
        # interpolation can bridge a fully missing step
        TimeSeriesFiller('interpolate').fit(da_)

    def test_policy_off_returns_empty_report_without_warnings(self):
        # I7: 'off' must not scan the raster
        da_ = _make_da()
        da_[1, 0, :4, :] = np.nan
        rep, msgs = _caught(audit_input, da_, stage='fit', policy='off')
        self.assertEqual(msgs, [])
        self.assertEqual(len(rep), 0)
        self.assertIn('frac_missing', rep.columns)

    def test_predict_shift_compares_by_position(self):
        # I3: a different year with the same number of steps is not a shift
        clean = _make_da()
        clean.attrs['cloud_masked'] = True
        clean[1, :, 0, :] = np.nan  # 10% missing at fit, step 1
        ref = audit_input(clean, stage='fit', policy='warn')
        other_year = _make_da(seed=1).assign_coords(time=np.arange(10, 14))
        other_year[1, :, 0, :] = np.nan  # same 10%: not a shift
        _, msgs = _caught(
            audit_input, other_year, stage='predict', reference=ref
        )
        self.assertFalse(any('training' in m.lower() for m in msgs), msgs)
        other_year[1, :, :4, :] = np.nan  # now 40%
        _, msgs = _caught(
            audit_input, other_year, stage='predict', reference=ref
        )
        shift = [m for m in msgs if 'training' in m.lower()]
        self.assertTrue(shift, msgs)
        self.assertIn('11', shift[0])
        self.assertIn('10% missing at fit', shift[0])

    def test_messages_render_plain_labels(self):
        da_ = _make_da().assign_coords(time=np.arange(4, dtype='float64'))
        da_[1, 0, :4, :] = np.nan
        _, msgs = _caught(audit_input, da_, stage='fit', fill='train_median')
        heavy = [m for m in msgs if 'missing' in m.lower()][0]
        self.assertNotIn('np.', heavy)
        self.assertIn("time=1, band='b0'", heavy)

    def test_policy_off_and_raise(self):
        da_ = _make_da()
        da_[1, 0, :4, :] = np.nan
        _, msgs = _caught(audit_input, da_, stage='fit', policy='off')
        self.assertEqual(msgs, [])
        with self.assertRaises(ValueError):
            audit_input(da_, stage='fit', policy='raise')
        with self.assertRaises(ValueError):
            audit_input(da_, stage='fit', policy='bogus')


class TestTimeSeriesFiller(unittest.TestCase):
    def _gappy(self):
        da_ = _make_da()
        da_[1, 0, :4, :] = np.nan
        da_[3, 1, 2, 2] = np.nan
        return da_

    def test_train_median_matches_nanmedian_and_fills_only_nan(self):
        da_ = self._gappy()
        filler = TimeSeriesFiller('train_median').fit(da_)
        expected = np.nanmedian(da_.values, axis=(2, 3))  # (time, band)
        np.testing.assert_allclose(
            filler.fill_values_.transpose('time', 'band').values, expected
        )
        out = filler.transform(da_).compute()
        self.assertFalse(np.isnan(out.values).any())
        self.assertAlmostEqual(
            float(out.values[1, 0, 0, 0]), float(expected[1, 0]), places=6
        )
        # untouched elsewhere
        np.testing.assert_array_equal(out.values[0], da_.values[0])

    def test_train_median_applies_to_new_data(self):
        train = self._gappy()
        filler = TimeSeriesFiller('train_median').fit(train)
        test = _make_da(seed=5)
        test[2, 1, 0, 0] = np.nan
        out = filler.transform(test).compute()
        self.assertAlmostEqual(
            float(out.values[2, 1, 0, 0]),
            float(np.nanmedian(train.values[2, 1])),
            places=6,
        )

    def test_interpolate_linear_with_end_fill(self):
        da_ = _make_da()
        da_[:, 0, 0, 0] = [0.2, np.nan, 0.6, 0.8]
        da_[:, 1, 0, 0] = [np.nan, 0.3, 0.5, np.nan]
        out = TimeSeriesFiller('interpolate').fit(da_).transform(da_).compute()
        self.assertAlmostEqual(float(out.values[1, 0, 0, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(out.values[0, 1, 0, 0]), 0.3, places=5)
        self.assertAlmostEqual(float(out.values[3, 1, 0, 0]), 0.5, places=5)

    def test_train_median_fills_positionally_when_labels_differ(self):
        # C1: predicting another year must still use the fit medians
        train = self._gappy()
        filler = TimeSeriesFiller('train_median').fit(train)
        test = _make_da(seed=5).assign_coords(time=np.arange(10, 14))
        test[2, 1, 0, 0] = np.nan
        out = filler.transform(test).compute()
        self.assertAlmostEqual(
            float(out.values[2, 1, 0, 0]),
            float(np.nanmedian(train.values[2, 1])),
            places=6,
        )
        short = _make_da(seed=5).isel(time=slice(0, 3))
        with self.assertRaisesRegex(ValueError, 'time'):
            filler.transform(short)

    def test_transform_keeps_footprint_nan(self):
        # I5: pixels missing everywhere stay NaN after filling
        da_ = _make_da()
        da_[:, :, :2, :] = np.nan
        da_[1, 0, 5, 5] = np.nan
        for strategy in ('train_median', 'zero'):
            filler = TimeSeriesFiller(strategy).fit(da_)
            out, _ = _caught(filler.transform, da_)
            out = out.compute()
            self.assertTrue(np.isnan(out.values[:, :, :2, :]).all())
            self.assertFalse(np.isnan(out.values[1, 0, 5, 5]))

    def test_interpolate_single_valid_observation_is_propagated(self):
        # I6: one valid value must be carried to every step
        da_ = _make_da()
        da_[:, 0, 0, 0] = [np.nan, np.nan, 0.5, np.nan]
        da_[:, 1, 0, 0] = np.nan
        out = TimeSeriesFiller('interpolate').fit(da_).transform(da_).compute()
        np.testing.assert_allclose(out.values[:, 0, 0, 0], 0.5)
        self.assertTrue(np.isnan(out.values[:, 1, 0, 0]).all())

    def test_interpolate_requires_time(self):
        da_ = _make_da(with_time=False)
        with self.assertRaisesRegex(ValueError, 'time'):
            TimeSeriesFiller('interpolate').fit(da_).transform(da_)

    def test_zero_warns_and_fills(self):
        da_ = self._gappy()
        filler = TimeSeriesFiller('zero').fit(da_)
        out, msgs = _caught(filler.transform, da_)
        self.assertTrue(msgs)
        self.assertEqual(float(out.values[1, 0, 0, 0]), 0.0)

    def test_none_leaves_nan(self):
        da_ = self._gappy()
        out = TimeSeriesFiller('none').fit(da_).transform(da_)
        self.assertTrue(np.isnan(out.values[1, 0, 0, 0]))

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            TimeSeriesFiller('bogus')

    def test_residual_mask_flags_all_nan_pixels(self):
        da_ = _make_da()
        da_[:, 0, 4, 4] = np.nan  # whole series missing in band 0
        da_[1, 1, 0, 0] = np.nan  # ordinary gap
        filler = TimeSeriesFiller('interpolate').fit(da_)
        filled = filler.transform(da_)
        mask = filler.residual_mask(filled).compute()
        self.assertEqual(mask.dims, ('y', 'x'))
        self.assertTrue(bool(mask.values[4, 4]))
        self.assertFalse(bool(mask.values[0, 0]))
        self.assertEqual(int(mask.values.sum()), 1)

    def test_fill_3d_train_median(self):
        da_ = _make_da(with_time=False)
        da_[0, 0, 0] = np.nan
        filler = TimeSeriesFiller('train_median').fit(da_)
        self.assertEqual(filler.fill_values_.dims, ('band',))
        out = filler.transform(da_).compute()
        self.assertFalse(np.isnan(out.values).any())


class TestFillTime(unittest.TestCase):
    def test_interpolate_with_max_gap(self):
        da_ = _make_da()
        da_[:, 0, 0, 0] = [0.2, np.nan, np.nan, 0.8]  # gap of 2
        da_[:, 1, 0, 0] = [0.2, np.nan, 0.6, 0.8]  # gap of 1
        out = fill_time(da_, method='interpolate', max_gap=1).compute()
        self.assertTrue(np.isnan(out.values[1, 0, 0, 0]))
        self.assertAlmostEqual(float(out.values[1, 1, 0, 0]), 0.4, places=5)
        self.assertEqual(out.attrs['res'], (10.0, 10.0))

    def test_median_method(self):
        da_ = _make_da()
        da_[1, 0, 0, 0] = np.nan
        out = fill_time(da_, method='median').compute()
        self.assertAlmostEqual(
            float(out.values[1, 0, 0, 0]),
            float(np.nanmedian(da_.values[1, 0])),
            places=6,
        )

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            fill_time(_make_da(), method='bogus')


class TestMaskClouds(unittest.TestCase):
    def _code_mask(self, with_time=True):
        shape = (1, NT, NY, NX) if with_time else (1, NY, NX)
        codes = np.zeros(shape, dtype='uint8')
        dims = ('band', 'time', 'y', 'x') if with_time else ('band', 'y', 'x')
        coords = {
            'band': ['mask'],
            'y': np.arange(NY)[::-1] * 10.0,
            'x': np.arange(NX) * 10.0,
        }
        if with_time:
            coords['time'] = np.arange(NT)
        return xr.DataArray(codes, dims=dims, coords=coords)

    def test_qamasker_style_time_mask(self):
        da_ = _make_da()
        mask = self._code_mask()
        mask[0, 2, 1, 1] = 4  # cloud at time 2
        mask[0, 0, 3, 3] = 2  # shadow at time 0
        out = mask_clouds(da_, mask).compute()
        self.assertTrue(np.isnan(out.values[2, :, 1, 1]).all())
        self.assertTrue(np.isnan(out.values[0, :, 3, 3]).all())
        self.assertFalse(np.isnan(out.values[1]).any())
        self.assertEqual(int(np.isnan(out.values).sum()), 2 * NB)
        self.assertTrue(out.attrs['cloud_masked'])
        self.assertEqual(out.attrs['res'], (10.0, 10.0))

    def test_single_date_mask_requires_broadcast_flag(self):
        da_ = _make_da()
        mask = self._code_mask(with_time=False)
        mask[0, 1, 1] = 4
        with self.assertRaisesRegex(ValueError, 'broadcast_time'):
            mask_clouds(da_, mask)
        out = mask_clouds(da_, mask, broadcast_time=True).compute()
        self.assertTrue(np.isnan(out.values[:, :, 1, 1]).all())

    def test_boolean_mask(self):
        da_ = _make_da()
        cloud = xr.DataArray(
            np.zeros((NT, NY, NX), dtype=bool),
            dims=('time', 'y', 'x'),
            coords={'time': da_.time, 'y': da_.y, 'x': da_.x},
        )
        cloud[3, 5, 5] = True
        out = mask_clouds(da_, cloud).compute()
        self.assertTrue(np.isnan(out.values[3, :, 5, 5]).all())
        self.assertEqual(int(np.isnan(out.values).sum()), NB)

    def test_mismatched_grid_is_rejected(self):
        # I2
        da_ = _make_da()
        mask = self._code_mask()
        shifted = mask.assign_coords(x=mask.x + 5000.0)
        with self.assertRaisesRegex(ValueError, 'grid'):
            mask_clouds(da_, shifted)

    def test_custom_clear_values(self):
        da_ = _make_da()
        mask = self._code_mask()
        mask[0, 1, 0, 0] = 1  # water, treated as clear here
        out = mask_clouds(da_, mask, clear_values=(0, 1)).compute()
        self.assertFalse(np.isnan(out.values).any())


class TestAccessor(unittest.TestCase):
    def test_accessor_wrappers_on_real_data(self):
        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518] * 3, stack_dim='time', nodata=0
            ) as src:
                rep = src.gw.missing_report()
                self.assertIsInstance(rep, pd.DataFrame)
                self.assertEqual(len(rep), 3 * src.gw.nbands)
                filled = src.gw.fill_time(method='interpolate')
                self.assertIsInstance(filled, xr.DataArray)
                self.assertEqual(filled.dims, src.dims)
                cloud = xr.zeros_like(src.isel(band=0), dtype=bool)
                masked = src.gw.mask_clouds(cloud)
                self.assertTrue(masked.attrs['cloud_masked'])
                self.assertEqual(masked.dims, src.dims)


if __name__ == '__main__':
    unittest.main()
