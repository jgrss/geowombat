# Changelog

<!--next-version-placeholder-->


## v2.0.8 (2022-09-21)
* Added a `nodataval` `DataArray` property ([#208](https://github.com/jgrss/geowombat/pull/208))

## v2.0.7 (2022-09-20)
* Fixed ML `nodata` and `dtype` ([#207](https://github.com/jgrss/geowombat/pull/207))

## v2.0.6 (2022-09-19)
* Changed behavior of `nodata` values ([#204](https://github.com/jgrss/geowombat/pull/204))

## v2.0.5 (2022-09-16)
* Fixed attribute lookup when co-registration is applied ([#203](https://github.com/jgrss/geowombat/pull/203))

## v2.0.4 (2022-09-15)
* Fixed 'filename' attribute when opening a NetCDF file ([#201](https://github.com/jgrss/geowombat/pull/201))

## v2.0.3 (2022-09-14)
* Pinned Python >= 3.8 in `setup.cfg` ([#200](https://github.com/jgrss/geowombat/pull/200))

## v2.0.2 (2022-09-13)
* Added CRS to WKT transformation for co-registration ([#199](https://github.com/jgrss/geowombat/pull/199))

## v2.0.1 (2022-09-13)
* Fixed ML tests ([#198](https://github.com/jgrss/geowombat/pull/198))

## v2.0.0 (2022-09-01)
* Added `geowombat.save()` method ([#189](https://github.com/jgrss/geowombat/pull/189))
* Warping methods now return `dask.Delayed` objects ([#189](https://github.com/jgrss/geowombat/pull/189))
* Better CRS checks ([#189](https://github.com/jgrss/geowombat/pull/189))

## v1.11.4 (2022-08-31)
* Fixed `to_raster()` ([#187](https://github.com/jgrss/geowombat/pull/187))

## v1.11.3 (2022-07-10)
* Added user proj bounds to return more specific bbox ([#180](https://github.com/jgrss/geowombat/issues/180))

## v1.11.2 (2022-07-10)
* Fixed CRS errors generated from the readthedocs build ([#178](https://github.com/jgrss/geowombat/issues/178))

## v1.11.1 (2022-07-09)
* Fixed CRS translation error of certain EPSG codes ([#177](https://github.com/jgrss/geowombat/issues/177))

## v1.11.0 (2022-07-09)
* Added Landsat 9 to metadata lookup

## v1.10.1 (2022-07-07)
* Removed imports from `geowombat.__init__`

## v1.10.0 (2022-07-07)
* Add STAC API to read Landsat and Sentinel-2 time series

## v1.9.1 (2022-06-18)
* Added support for Landsat 9

## v1.8.6 (2022-06-12)
### Fix
* Added token ([#148](https://github.com/jgrss/geowombat/issues/148)) ([`79b0243`](https://github.com/jgrss/geowombat/commit/79b0243df5765865ef913ab42b911960649ec511))
* Removed semantic version header ([#147](https://github.com/jgrss/geowombat/issues/147)) ([`529b02b`](https://github.com/jgrss/geowombat/commit/529b02bcf128ab31eecf52a7f2067626461cc6b7))
* Test github action release ([#146](https://github.com/jgrss/geowombat/issues/146)) ([`f99f6de`](https://github.com/jgrss/geowombat/commit/f99f6de714dcf355dca3cb82126c7fa4ff65952a))
* Pin min requests version ([#143](https://github.com/jgrss/geowombat/issues/143)) ([`98ad33a`](https://github.com/jgrss/geowombat/commit/98ad33aa15474d88f7396c32f765a33d7265021f))

## v1.8.5 (2022-05-31)
### Fix
* Jgrss/dependencies ([#134](https://github.com/jgrss/geowombat/issues/134)) ([`342bb2b`](https://github.com/jgrss/geowombat/commit/342bb2b518350ac1617dcca3329b9645862c17c9))

## v1.8.4 (2022-05-23)
### Fix
* Changed upload to PyPI to GitHub releases ([#113](https://github.com/jgrss/geowombat/issues/113)) ([`378f8ec`](https://github.com/jgrss/geowombat/commit/378f8ecd6671c6451d87e7d1949967a29f448be0))

## v1.8.3 (2022-05-23)
### Fix
* Added documentation describing how nodata is applied ([#110](https://github.com/jgrss/geowombat/issues/110)) ([`8bd7d3d`](https://github.com/jgrss/geowombat/commit/8bd7d3dc8cd6c1d8a3a3d8dbc391300ad7602a99))

### Documentation
* Replaced version badge ([#109](https://github.com/jgrss/geowombat/issues/109)) ([`537386d`](https://github.com/jgrss/geowombat/commit/537386df4daa4c8cfc567b75db12b555a957d5e8))

## v1.8.2 (2022-05-22)
### Fix
* Added exit ([#108](https://github.com/jgrss/geowombat/issues/108)) ([`000d6fd`](https://github.com/jgrss/geowombat/commit/000d6fd35828ea1625e068b3343a23bd98743987))

## v1.8.1 (2022-05-22)
### Fix
* Added wheel ([#107](https://github.com/jgrss/geowombat/issues/107)) ([`ce86863`](https://github.com/jgrss/geowombat/commit/ce8686389a4a6f94cc441d35c523c8db68057791))

## v1.8.0 (2022-05-22)
### Feature
* Merge pull request #98 from jgrss/semantic-release ([`25aa5f3`](https://github.com/jgrss/geowombat/commit/25aa5f3c0920ae8591578f30998d4aa65010b43a))

### Fix
* Test fingerprint ([#106](https://github.com/jgrss/geowombat/issues/106)) ([`d8919cc`](https://github.com/jgrss/geowombat/commit/d8919cce5e9a4d9cc0a7f13ff600a4c6c79b6f53))
* Comment ([#105](https://github.com/jgrss/geowombat/issues/105)) ([`92e642e`](https://github.com/jgrss/geowombat/commit/92e642e7c5bbc64a6d5cead59bb1237dfddc6d7b))
* Small change ([#104](https://github.com/jgrss/geowombat/issues/104)) ([`5c32d7e`](https://github.com/jgrss/geowombat/commit/5c32d7eeb92b53ff2041a5ce8c8121e835979dcd))
* Small change ([#103](https://github.com/jgrss/geowombat/issues/103)) ([`3512e90`](https://github.com/jgrss/geowombat/commit/3512e901b2dd0f15886651190bd85b9d0ca4e9f6))
* Added git config ([#102](https://github.com/jgrss/geowombat/issues/102)) ([`3e77a4d`](https://github.com/jgrss/geowombat/commit/3e77a4def2b8e3997becdb003ee245bd6b42e8a2))
* Added token ([#101](https://github.com/jgrss/geowombat/issues/101)) ([`67e9bb3`](https://github.com/jgrss/geowombat/commit/67e9bb3263be3d3250fc242461582bc218c605f2))
* Release ([#100](https://github.com/jgrss/geowombat/issues/100)) ([`1e3c0c8`](https://github.com/jgrss/geowombat/commit/1e3c0c862173bd8bc553771b149c966e73f2d3ae))
* Merge pull request #99 from jgrss/semantic2 ([`daf469b`](https://github.com/jgrss/geowombat/commit/daf469ba177c29ec413fa86b76148776c5f415ed))
