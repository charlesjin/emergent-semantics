import pathlib
import datetime

import torch


YEAR = 2024
MONTH = 3
DAY = 7


class Cache:
    def __init__(self, verbose=True):
        self.cache = {}
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _file_recency(self, path):
        """Check if file at path is from this year."""
        fname = pathlib.Path(path)
        try:
            mtime = datetime.datetime.fromtimestamp(
                fname.stat().st_mtime, tz=datetime.timezone.utc
            )
        except FileNotFoundError:
            self._print(f"Didn't find {path}")
            return None
        return mtime

    def check(self, path, check_day=True, check_year=True):
        recency = self._file_recency(path)
        if recency is None:
            self._print(f"Count not find {path}.")
            return False

        if check_day and (
            recency.year < YEAR or recency.month < MONTH or recency.day < DAY
        ):
            self._print(
                f"Found {path} but file is stale: "
                f"{(recency.year, recency.month, recency.day)} vs "
                f"{(YEAR, MONTH, DAY)}."
            )
            return False
        if check_year and recency.year < YEAR:
            self._print(f"Found {path} but file is stale: {recency.year} vs {YEAR}.")
            return False
        return True

    def _load_file(self, path, check_recency=True):
        if not self.check(path, check_day=False, check_year=check_recency):
            return None

        try:
            r = torch.load(path, map_location="cpu")
            self._print(f"Successfully loaded {path} from disk.")
        except:
            self._print(f"Found {path} but failed to load from disk.")
            return None

        return r

    def load(self, path, max_length=None):
        # Check if in cache
        if path in self.cache:
            self._print(f"Cache hit on {path}.")
            return self.cache[path]

        # Try to load all at once
        self._print(f"Cache miss on {path}, loading from disk...")
        ds = self._load_file(path, check_recency=True)
        if ds:
            self.cache[path] = ds
            return ds

        # Try to load pieces
        idx = 0
        length = 0
        ds = []
        while True:
            part_idx = str(idx).zfill(3)
            part_path = f"{path}-{part_idx}"
            ds_part = self._load_file(part_path)
            if ds_part is None:
                break

            ds.extend(ds_part)
            idx += 1
            length += len(ds_part)
            if max_length and length >= max_length:
                break

        if not ds:
            self._print(f"{path} not found on disk.")
            return None

        self._print(f"Successfully loaded {path} parts from disk.")
        self.cache[path] = ds
        return ds

    def insert(self, path, ds):
        if path in self.cache:
            raise ValueError(f"{path} already in cache!")
        self.cache[path] = ds

    def evict(self, path):
        del self.cache[path]


CACHE = Cache()
