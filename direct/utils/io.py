# coding=utf-8
# Copyright (c) DIRECT Contributors

# Several of the utilities here are copied/modified from torchvision under the BSD License.
# https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/datasets/utils.py

import bz2
import gzip
import hashlib
import json
import logging
import lzma
import os
import os.path
import pathlib
import re
import tarfile
import urllib
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

USER_AGENT = "direct-group/direct"


def read_json(fn: Union[Dict, str, pathlib.Path]) -> Dict:
    """
    Read file and output dict, or take dict and output dict.


    Parameters
    ----------
    fn : Union[Dict, str, pathlib.Path]


    Returns
    -------
    dict

    """
    if isinstance(fn, dict):
        return fn

    with open(fn, "r") as f:
        data = json.load(f)
    return data


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()

        if isinstance(obj, np.ndarray):
            if obj.size > 10e4:
                warnings.warn(
                    "Trying to JSON serialize a very large array of size {obj.size}. "
                    "Reconsider doing this differently"
                )
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_json(fn: Union[str, pathlib.Path], data: Dict, indent=2) -> None:
    """
    Write dict data to fn.

    Parameters
    ----------
    fn : Path or str
    data : dict
    indent: int

    Returns
    -------
    None
    """
    with open(fn, "w") as f:
        json.dump(data, f, indent=indent, cls=ArrayEncoder)


def read_list(fn: Union[List, str, pathlib.Path]) -> List:
    """
    Read file and output list, or take list and output list. Can read data from URLs.

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list, or a URL to a text file.

    Returns
    -------
    list
        Text file read line by line.
    """
    if isinstance(fn, (pathlib.Path, str)):
        if isinstance(fn, str) and check_is_valid_url(fn):
            data = read_text_from_url(fn)
            return [_.strip() for _ in data.split("\n") if not _.startswith("#") and _ != ""]
        else:
            with open(fn) as f:
                data = f.readlines()
            return [_.strip() for _ in data if not _.startswith("#")]
    return fn


def write_list(fn: Union[str, pathlib.Path], data) -> None:
    """
    Write list line by line to file.

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list
    data : list or tuple
    Returns
    -------
    None
    """
    with open(fn, "w") as f:
        for line in data:
            f.write(f"{line}\n")


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


def download_url(
    url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, max_redirect_hops: int = 3
) -> None:
    """Download a file from a url and place it in root.

    Parameters
    ----------
    url : str
        URL to download file from
    root : str
        Directory to place downloaded file in
    filename : str, optional:
        Name to save the file under. If None, use the basename of the URL
    md5 : str, optional
        MD5 checksum of the download. If None, do not check
    max_redirect_hops : int, optional)
        Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        logger.info("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # download the file
    try:
        logger.info(f"Downloading {url} to {fpath}")
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            logger.info(f"Failed download. Trying https -> http instead. Downloading {url} to {fpath}")
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def _extract_tar(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
        tar.extractall(to_path)


_ZIP_COMPRESSION_MAP: Dict[str, int] = {
    ".bz2": zipfile.ZIP_BZIP2,
    ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with zipfile.ZipFile(
        from_path, "r", compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)


_ARCHIVE_EXTRACTORS: Dict[str, Callable[[str, str, Optional[str]], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
}
_COMPRESSED_FILE_OPENERS: Dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}
_FILE_TYPE_ALIASES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}


def _detect_file_type(file: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect the archive type and/or compression of a file.

    Args:
        file (str): the filename

    Returns:
        (tuple): tuple of suffix, archive type, and compression

    Raises:
        RuntimeError: if file has no suffix or suffix is not supported
    """
    suffixes = pathlib.Path(file).suffixes
    if not suffixes:
        raise RuntimeError(
            f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
        )
    suffix = suffixes[-1]

    # check if the suffix is a known alias
    if suffix in _FILE_TYPE_ALIASES:
        return (suffix, *_FILE_TYPE_ALIASES[suffix])

    # check if the suffix is an archive type
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, suffix, None

    # check if the suffix is a compression
    if suffix in _COMPRESSED_FILE_OPENERS:
        # check for suffix hierarchy
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # check if the suffix2 is an archive type
            if suffix2 in _ARCHIVE_EXTRACTORS:
                return suffix2 + suffix, suffix2, suffix

        return suffix, None, suffix

    valid_suffixes = sorted(set(_FILE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
    raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")


def _decompress(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    r"""Decompress a file.

    The compression is automatically detected from the file name.

    Parameters
    ----------
    from_path : str
        Path to the file to be decompressed.
    to_path : str
        Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns
    -------
    str, Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    if to_path is None:
        to_path = from_path.replace(suffix, archive_type if archive_type is not None else "")

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        os.remove(from_path)

    return to_path


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        return _decompress(
            from_path,
            os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, compression)

    return to_path


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    logger.info(f"Extracting {archive} to {extract_root}")
    extract_archive(archive, extract_root, remove_finished)


def read_text_from_url(url, chunk_size: int = 1024):
    """
    Read a text file from a URL, e.g. a config file

    Parameters
    ----------
    url : str
    chunk_size : int

    Returns
    -------
    str
        Data from URL
    """
    if not check_is_valid_url(url):
        raise ValueError(f"{url} is not a valid URL.")

    data = b""

    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    data += chunk
    except urllib.error.HTTPError as e:
        e.msg = f"{e.msg}: {url}"
        raise

    return data.decode()


def check_is_valid_url(path: str) -> bool:
    """
    Check if the given path is a valid url.

    Parameters
    ----------
    path : str

    Returns
    -------
    Bool describing if this is an URL or not.
    """
    # From https://gist.github.com/dokterbob/998722/1c380cb896afa22306218f73384b79d2d4386638
    if not str(path).startswith("http") and not str(path).startswith("s3") and not str(path).startswith("ftp"):
        return False

    regex = re.compile(
        r"^((?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )  # host is optional, allow for relative URLs

    if re.match(regex, path):
        return True
    return False
