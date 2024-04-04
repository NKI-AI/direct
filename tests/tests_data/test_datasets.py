# Copyright (c) DIRECT Contributors

"""Tests for the `direct.data.datasets` module."""

import pathlib
import tempfile

import h5py
import ismrmrd
import numpy as np
import pytest
from xsdata.formats.dataclass.serializers import XmlSerializer

from direct.data.datasets import (
    CalgaryCampinasDataset,
    CMRxReconDataset,
    ConcatDataset,
    FakeMRIBlobsDataset,
    FastMRIDataset,
    SheppLoganProtonDataset,
    SheppLoganT1Dataset,
    SheppLoganT2Dataset,
)


def create_fastmri_h5file(filename, shape, recon_shape):
    header = ismrmrd.xsd.ismrmrdHeader()
    encoding = ismrmrd.xsd.encodingType()

    ematrix = ismrmrd.xsd.matrixSizeType()
    rmatrix = ismrmrd.xsd.matrixSizeType()

    ematrix = ismrmrd.xsd.matrixSizeType()
    ematrix.x = shape[2]
    ematrix.y = shape[3]
    ematrix.z = shape[0]
    rmatrix = ismrmrd.xsd.matrixSizeType()
    rmatrix.x = recon_shape[1]
    rmatrix.y = recon_shape[2]
    rmatrix.z = recon_shape[0]

    espace = ismrmrd.xsd.encodingSpaceType()
    espace.matrixSize = ematrix

    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix

    # Set encoded and recon spaces
    encoding.encodedSpace = espace
    encoding.reconSpace = rspace

    # Encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()
    limits1 = ismrmrd.xsd.limitType()
    limits1.minimum = 0
    limits1.center = round(shape[3] / 2)
    limits1.maximum = shape[3] - 1
    limits.kspace_encoding_step_1 = limits1

    limits_rep = ismrmrd.xsd.limitType()
    limits_rep.minimum = 0
    limits_rep.center = 0
    limits_rep.maximum = 0
    limits.repetition = limits_rep

    limits_rest = ismrmrd.xsd.limitType()
    limits_rest.minimum = 0
    limits_rest.center = 0
    limits_rest.maximum = 0
    limits.kspace_encoding_step_0 = limits_rest
    limits.slice = limits_rest
    limits.average = limits_rest
    limits.contrast = limits_rest
    limits.kspaceEncodingStep2 = limits_rest
    limits.phase = limits_rest
    limits.segment = limits_rest
    limits.set = limits_rest

    encoding.encodingLimits = limits
    header.encoding.append(encoding)

    kspace = np.random.rand(*shape) + 1.0j * np.random.rand(*shape)
    rss = np.random.rand(shape[0], *shape[2:])
    h5file = h5py.File(filename, "w")
    h5file.create_dataset("kspace", data=kspace)
    h5file.create_dataset("reconstruction_rss", data=rss)

    # Serializing 'header' object to XML string.
    xml_string = XmlSerializer().render(header)
    h5file.create_dataset("ismrmrd_header", data=xml_string)

    h5file.attrs["norm"] = np.linalg.norm(kspace)
    h5file.attrs["max"] = np.abs(kspace).max()

    h5file.close()


@pytest.mark.parametrize(
    "num_samples",
    [3],
)
@pytest.mark.parametrize(
    "shape, recon_shape",
    [[(6, 12, 20, 10), (6, 15, 8)]],
)
@pytest.mark.parametrize(
    "transform",
    [None, lambda x: x],
)
@pytest.mark.parametrize(
    "filter",
    [None, ["file0.h5", "file1.h5"]],
)
def test_FastMRIDataset(num_samples, shape, recon_shape, transform, filter):
    FASTMRI_KEYS = {
        "kspace",
        "filename",
        "slice_no",
        "scaling_factor",
        "padding_left",
        "padding_right",
        "encoding_size",
        "reconstruction_size",
    }
    with tempfile.TemporaryDirectory() as tempdir:
        for _ in range(num_samples):
            create_fastmri_h5file(pathlib.Path(tempdir) / f"file{_}.h5", shape, recon_shape)
        if filter:
            f = open(pathlib.Path(tempdir) / "filter.lst", "w")
            for filename in filter:
                f.write(filename + "\n")
            f.close()
        dataset = FastMRIDataset(
            pathlib.Path(tempdir),
            filenames_filter=[pathlib.Path(pathlib.Path(tempdir) / f) for f in filter] if filter else None,
            transform=transform,
        )
        assert len(dataset) == (num_samples if not filter else len(filter)) * shape[0]
        assert all(FASTMRI_KEYS.issubset(dataset[_]) for _ in range(len(dataset)))

        # Test with filenames_lists
        if filter:
            dataset = FastMRIDataset(
                pathlib.Path(tempdir),
                filenames_filter=None,
                filenames_lists=["filter.lst"],
                filenames_lists_root=pathlib.Path(tempdir),
                transform=transform,
            )
            assert len(dataset) == len(filter) * shape[0]
            assert all("kspace" in _.keys() for _ in dataset)


@pytest.mark.parametrize(
    "num_samples",
    [3],
)
@pytest.mark.parametrize(
    "shape",
    [(160, 3, 5, 6)],
)
@pytest.mark.parametrize(
    "transform",
    [None, lambda x: x],
)
@pytest.mark.parametrize(
    "filter",
    [None, ["file0.h5", "file1.h5"]],
)
def test_CalgaryCampinasDataset(num_samples, shape, transform, filter):
    with tempfile.TemporaryDirectory() as tempdir:
        for _ in range(num_samples):
            kspace = np.random.rand(*shape)
            h5file = h5py.File(pathlib.Path(tempdir) / f"file{_}.h5", "w")
            h5file.create_dataset("kspace", data=kspace)
            h5file.close()
        if filter:
            f = open(pathlib.Path(tempdir) / "filter.lst", "w", encoding="utf-8")
            for filename in filter:
                f.write(filename + "\n")
            f.close()
        dataset = CalgaryCampinasDataset(
            pathlib.Path(tempdir),
            crop_outer_slices=True,
            filenames_filter=[pathlib.Path(pathlib.Path(tempdir) / f) for f in filter] if filter else None,
            transform=transform,
        )
        assert len(dataset) == (num_samples if not filter else len(filter)) * (shape[0] - 100)
        assert all("kspace" in _.keys() for _ in dataset)

        # Test with filenames_lists
        if filter:
            dataset = CalgaryCampinasDataset(
                pathlib.Path(tempdir),
                crop_outer_slices=True,
                filenames_filter=None,
                filenames_lists=["filter.lst"],
                filenames_lists_root=pathlib.Path(tempdir),
                transform=transform,
            )
            assert len(dataset) == len(filter) * (shape[0] - 100)
            assert all("kspace" in _.keys() for _ in dataset)


@pytest.mark.parametrize(
    "shape, num_coils",
    [[(6, 20, 10), 8], [(20, 20, 20), 5]],
)
@pytest.mark.parametrize(
    "transform",
    [None, lambda x: x],
)
@pytest.mark.parametrize(
    "T2_star",
    [False, True],
)
def test_shepp_logan_datasets(shape, num_coils, transform, T2_star):
    datasets = [SheppLoganT1Dataset, SheppLoganT2Dataset, SheppLoganProtonDataset]
    args = {"shape": shape, "num_coils": num_coils, "transform": transform, "text_description": "test"}
    for d in datasets:
        dataset = d(**({**args, **{"T2_star": T2_star}} if d == SheppLoganT2Dataset else args))
        assert len(dataset) == shape[-1]
        assert dataset[0]["kspace"].shape == (num_coils,) + shape[:-1]


@pytest.mark.parametrize(
    "num_samples",
    [3],
)
@pytest.mark.parametrize(
    "shape, num_coils",
    [[(6, 20, 10), 5], [(20, 10), 3], [(2, 3, 20, 10), None]],
)
@pytest.mark.parametrize(
    "transform",
    [None, lambda x: x],
)
def test_FakeMRIBlobsDataset(num_samples, num_coils, shape, transform):
    FAKE_KEYS = {
        "kspace",
        "filename",
        "slice_no",
        "scaling_factor",
        "encoding_size",
        "reconstruction_size",
    }
    if len(shape) not in [2, 3]:
        with pytest.raises(NotImplementedError):
            dataset = FakeMRIBlobsDataset(num_samples, num_coils, shape, transform)
    else:
        dataset = FakeMRIBlobsDataset(
            num_samples,
            num_coils,
            shape,
            transform,
            pass_attrs=True,
            text_description="test",
            filenames="file",
            seed=0,
        )
        assert all(FAKE_KEYS.issubset(dataset[_]) for _ in range(len(dataset)))


@pytest.mark.parametrize(
    "num_samples, shapes",
    [
        [[3, 5], [(5, 3), (5, 4)]],
        [[4, 7], [(2, 5, 3), (4, 5, 4)]],
    ],
)
def test_ConcatDataset(num_samples, shapes):
    from torchvision.datasets import FakeData

    datasets = []
    for num, shape in zip(num_samples, shapes):
        datasets.append(FakeData(num, image_size=shape, random_offset=0))
    dataset = ConcatDataset(datasets)

    assert len(dataset) == sum(num_samples)

    for dataset_idx, num in enumerate(num_samples):
        assert np.allclose(datasets[dataset_idx][num - 1][0], dataset[np.cumsum(num_samples)[dataset_idx] - 1][0])

    with pytest.raises(ValueError):
        dataset[-(np.cumsum(num_samples) + 1)]


@pytest.mark.parametrize(
    "num_samples",
    [3],
)
@pytest.mark.parametrize(
    "shape",
    [(3, 9, 10, 100, 130)],
)
@pytest.mark.parametrize(
    "kspace_context",
    [None, "time", "slice"],
)
@pytest.mark.parametrize("extra_keys, compute_mask", [[["mask"], False], [None, True], [None, False]])
@pytest.mark.parametrize(
    "transform",
    [None, lambda x: x],
)
@pytest.mark.parametrize(
    "filter",
    [None, ["file0.mat", "file1.mat"]],
)
def test_CMRxReconDataset(num_samples, shape, kspace_context, compute_mask, extra_keys, transform, filter):
    with tempfile.TemporaryDirectory() as tempdir:
        for _ in range(num_samples):
            kspace = np.random.rand(*shape) + 1j * np.random.rand(*shape)
            ny, nx = shape[-2:]
            if compute_mask or extra_keys is not None:
                mask = np.zeros((ny, nx), dtype=bool)
                mask[:, ny // 2 - 12 : ny // 2 + 12] = True
                mask[:, np.random.randint(0, ny, 30)] = True

            if compute_mask:
                kspace = mask[None, None, None] * kspace

            with h5py.File(pathlib.Path(tempdir) / f"file{_}.mat", "w") as f:
                dtype = np.dtype([("real", np.float32), ("imag", np.float32)])
                # Reshape the complex data into a shape that matches the compound datatype
                compound_data = np.empty(shape, dtype=dtype)
                compound_data["real"] = np.real(kspace)
                compound_data["imag"] = np.imag(kspace)
                f.create_dataset("kspace_full", data=compound_data)
                if extra_keys is not None:
                    f.create_dataset("mask", data=mask, dtype=bool)

        dataset = CMRxReconDataset(
            pathlib.Path(tempdir),
            transform=transform,
            kspace_context=kspace_context,
            compute_mask=compute_mask,
            extra_keys=extra_keys,
            filenames_filter=[pathlib.Path(pathlib.Path(tempdir) / f) for f in filter] if filter else None,
        )
        sample = dataset[0]
        assert "kspace" in sample

        if kspace_context is None:
            assert dataset.ndim == 2
            assert len(dataset) == np.prod(shape[:2]) * (num_samples if not filter else len(filter))
            assert sample["kspace"].shape == (shape[2],) + shape[3:][::-1]
        elif kspace_context == "time":
            assert dataset.ndim == 3
            assert len(dataset) == shape[1] * (num_samples if not filter else len(filter))
            assert sample["kspace"].shape == (shape[2], shape[0]) + shape[3:][::-1]
        else:
            assert dataset.ndim == 3
            assert len(dataset) == shape[0] * (num_samples if not filter else len(filter))
            assert sample["kspace"].shape == (shape[2], shape[1]) + shape[3:][::-1]
        if compute_mask or extra_keys is not None:
            assert "sampling_mask" in sample
            assert "acs_mask" in sample
            np.allclose(sample["sampling_mask"], mask.T[None, ..., None])
