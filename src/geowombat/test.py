# %%
import os
import sys

os.chdir(r"/home/mmann1123/Documents/github/geowombat")
sys.path.append("/")
sys.path.append("/geowombat")
import geowombat as gw
import rasterio
from geowombat.data import l8_224077_20200518_B2, l8_224078_20200518_B2
import matplotlib.pyplot as plt
from geowombat.data import l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan

fig, ax = plt.subplots(dpi=200)

# with gw.config.update(
# ref_res=(10, 10),
# ):  #
with gw.open(
    # l8_224077_20200518_B2,
    [l8_224077_20200518_B2, l8_224078_20200518_B2],
    mosaic=True,
    bounds_by="union",
) as src:
    display(src)
    src.where(src != 0).gw.imshow(robust=True, ax=ax)
    src.gw.save(
        f"./test.tif",
        compress="lzw",
        overwrite=True,
        bigtiff="IF_NEEDED",
    )
# %%
fig, ax = plt.subplots(dpi=200)

with gw.open("./test.tif") as src:
    display(src.attrs["transform"])
    src.where(src != 0).gw.imshow(robust=True, ax=ax)

# %%
fig, ax = plt.subplots(dpi=200)

with gw.open([l8_224077_20200518_B2], mosaic=True) as src:
    attrs = src.attrs
    src = src.where(src > 0)
    print(src)
    src.attrs = attrs
    src.gw.save(
        "l8_224077_20200518_B2_nan.tif",
    )

src.gw.imshow(robust=True, ax=ax)
# %%
fig, ax = plt.subplots(dpi=200)

with gw.open(
    [l8_224078_20200518_B2],
    mosaic=True,
) as src:
    attrs = src.attrs
    src = src.where(src > 0)
    print(src)
    src.attrs = attrs
    src.gw.save(
        "l8_224078_20200518_B2_nan.tif",
    )
src.gw.imshow(robust=True, ax=ax)


# %%
fig, ax = plt.subplots(dpi=200)
import numpy as np

with gw.open(
    [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan],
    mosaic=True,
    bounds_by="union",
    nodata=np.nan,
    overlap="mean",
) as src:
    src
    # print(src)
    src.gw.imshow(robust=True, ax=ax)
# %%
from geowombat.core import coords_to_indices, lonlat_to_xy

with gw.open(l8_224077_20200518_B2_nan) as src:
    x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
    j, i = coords_to_indices(x, y, src)
    mid_values_77 = src[0, i : i + 3, j : j + 3].values


with gw.open(l8_224078_20200518_B2_nan) as src:
    x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
    j, i = coords_to_indices(x, y, src)
    mid_values_78 = src[0, i : i + 3, j : j + 3].values

    mean_answer = np.mean([mid_values_77, mid_values_78], axis=0)
    print(mean_answer)
    max_answer = np.max([mid_values_77, mid_values_78], axis=0)
    print(max_answer)

    min_answer = np.min([mid_values_77, mid_values_78], axis=0)
    print(min_answer)
# %%


filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
with gw.open(
    filenames,
    band_names=["blue"],
    mosaic=True,
    overlap="min",
    bounds_by="union",
) as src:
    start_values = src.values[
        0,
        0,
        0:10,
    ]
    x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
    j, i = coords_to_indices(x, y, src)
    mid_values = src[0, i : i + 3, j : j + 3].values

    end_values = src.values[
        0,
        -2,
        -10:,
    ]

    print(
        "mean answer",
        np.allclose(
            mid_values,
            np.array(
                [
                    [8385.5, 8183.0, 8049.5],
                    [7936.0, 7868.0, 7887.0],
                    [7861.5, 7827.0, 7721.0],
                ]
            ),
        ),
    )
    print(
        "min answer",
        np.allclose(
            mid_values,
            np.array(
                [
                    [8384.0, 8183.0, 8049.0],
                    [7934.0, 7867.0, 7885.0],
                    [7861.0, 7826.0, 7721.0],
                ]
            ),
        ),
    )
    print(
        "max answer",
        np.allclose(
            mid_values,
            np.array(
                [
                    [8387.0, 8183.0, 8050.0],
                    [7938.0, 7869.0, 7889.0],
                    [7862.0, 7828.0, 7721.0],
                ]
            ),
        ),
    )

    print(
        "start answer",
        np.allclose(
            start_values,
            np.array(
                [
                    8482.0,
                    8489.0,
                    8483.0,
                    8547.0,
                    8561.0,
                    8574.0,
                    8616.0,
                    8530.0,
                    8396.0,
                    8125.0,
                ]
            ),
        ),
    )

    print(
        "end answer",
        np.allclose(
            end_values,
            np.array(
                [
                    7409.0,
                    7427.0,
                    7490.0,
                    7444.0,
                    7502.0,
                    7472.0,
                    7464.0,
                    7443.0,
                    7406.0,
                    np.nan,
                ]
            ),
            equal_nan=True,
        ),
    )

    # %%

#     filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
#     with gw.open(
#         filenames,
#         band_names=["blue"],
#         mosaic=True,
#         overlap="min",
#         bounds_by="intersection",
#         nodata=0,
#     ) as src:
#         start_values = src.values[
#             0,
#             0,
#             0:10,
#         ]
#         end_values = src.values[
#             0,
#             -2,
#             -10:,
#         ]
#         x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
#         j, i = coords_to_indices(x, y, src)
#         mid_values = src[0, i : i + 3, j : j + 3].values
#         print(
#             np.allclose(
#                 start_values,
#                 np.array(
#                     [
#                         8482.0,
#                         8489.0,
#                         8483.0,
#                         8547.0,
#                         8561.0,
#                         8574.0,
#                         8616.0,
#                         8530.0,
#                         8396.0,
#                         8125.0,
#                     ]
#                 ),
#             ),
#         )
#         print(
#             np.allclose(
#                 mid_values,
#                 np.array(
#                     [
#                         [8384.0, 8183.0, 8049.0],
#                         [7934.0, 7867.0, 7885.0],
#                         [7861.0, 7826.0, 7721.0],
#                     ]
#                 ),
#             )
#         )
#         print(
#             np.allclose(
#                 end_values,
#                 np.array(
#                     [
#                         7409.0,
#                         7427.0,
#                         7490.0,
#                         7444.0,
#                         7502.0,
#                         7472.0,
#                         7464.0,
#                         7443.0,
#                         7406.0,
#                         np.nan,
#                     ]
#                 ),
#                 equal_nan=True,
#             ),
#         )
# # %%
# %%
os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/models"
)
files = sorted(
    [
        "../features/B2/B2_minimum_0000000000-0000000000-part1.tif",
        "../features/B2/B2_minimum_0000046592-0000046592.tif",
        "../features/B2/B2_minimum_0000000000-0000046592-part1.tif",
        "../features/B2/B2_minimum_0000000000-0000046592-part2.tif",
        "../features/B2/B2_minimum_0000000000-0000000000.tif",
        "../features/B2/B2_minimum_0000000000-0000093184.tif",
        "../features/B2/B2_minimum_0000046592-0000093184.tif",
        "../features/B2/B2_minimum_0000046592-0000000000.tif",
    ]
)
files


with gw.open(
    files[0:2],
    mosaic=True,
    overlap="max",
    bounds_by="union",
    nodata=0,
) as src:
    src.gw.to_raster(
        f"../final_model_features/B2_minimum.tif",
        compress="lzw",
        overwrite=True,
        separate=True,
        bigtiff="YES",
    )
# %%
import numpy as np

filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
with gw.open(
    filenames,
    band_names=["blue"],
    mosaic=True,
    bounds_by="union",
) as src:
    values = src.values[
        0,
        src.gw.nrows // 2,
        src.gw.ncols // 2 : src.gw.ncols // 2 + 10,
    ]

    np.allclose(
        values,
        np.array(
            [
                7524,
                7538,
                7573,
                7625,
                7683,
                7661,
                7643,
                7773,
                7697,
                7566,
            ]
        ),
    )
# %%
