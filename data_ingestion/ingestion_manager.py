"""
Production-grade async data ingestion with DVC lineage, aiohttp concurrency,
and per-URL fault isolation.
"""

import os
import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import logging

import aiohttp

from data_ingestion.loader import DataLoader
from data_ingestion.dataset_object import DatasetObject
from data_ingestion.sampling import validate_dataset

logger = logging.getLogger(__name__)


class DataIngestionManager:
    """
    Async, production-grade data ingestion with:
      - Concurrent multi-URL downloads via aiohttp + asyncio.gather
      - Per-URL fault isolation (404 / conn errors caught, rest continue)
      - SHA-256 cache keying to avoid redundant downloads
      - DVC lineage: `dvc add <cache_path>` called after each successful cache write
    """

    def __init__(self, cache_dir: Union[str, Path] = "./data/dataset_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Mapping of str session ID -> list of successfully ingested hashes
        self.session_datasets: Dict[str, List[str]] = {}
        self._loader = DataLoader()
        self.cache_metadata: Dict[str, Any] = self._load_cache_metadata()

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _load_cache_metadata(self) -> Dict[str, Any]:
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache_metadata(self) -> None:
        metadata_file = self.cache_dir / "cache_metadata.json"
        # Atomic write: temp file + os.replace prevents half-written JSON
        # if the process crashes mid-write or concurrent ingestions race.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.cache_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.cache_metadata, f, indent=2)
            os.replace(tmp_path, str(metadata_file))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _normalize_url(source: str) -> str:
        """
        Canonicalise a URL so that cosmetic variants produce the same hash.

        Normalisation steps:
          1. Strip leading/trailing whitespace
          2. Remove trailing slashes
          3. Lowercase the scheme + host (path is case-sensitive)
          4. Remove ``www.`` prefix from host
          5. Strip query params and fragments (they don't affect the dataset)
          6. Kaggle shortcut: ``kaggle.com/datasets/<owner>/<name>`` is the
             canonical form regardless of full-URL embellishments

        Non-URL strings (local paths) are returned as-is after stripping.
        """
        source = source.strip()
        if not source.startswith(("http://", "https://")):
            # Local path — normalise slashes but keep case (Windows paths)
            return source.replace("\\", "/").rstrip("/")
        parsed = urlparse(source)
        host = parsed.hostname or ""
        host = host.lower().removeprefix("www.")
        path = parsed.path.rstrip("/")
        # Kaggle canonical form: just owner/dataset from the path
        if "kaggle.com" in host and "/datasets/" in path:
            parts = path.split("/datasets/", 1)
            if len(parts) == 2:
                dataset_slug = parts[1].strip("/")
                return f"kaggle://datasets/{dataset_slug}"
        return f"{parsed.scheme}://{host}{path}"

    def _generate_hash(self, source: str) -> str:
        """Generate a 16-char SHA-256 hex digest for a normalised source identifier."""
        normalised = self._normalize_url(source)
        return hashlib.sha256(normalised.encode()).hexdigest()[:16]

    def _legacy_hash(self, source: str) -> str:
        """Hash using the raw source string (pre-normalisation era)."""
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    @staticmethod
    def _is_kaggle_url(url: str) -> bool:
        return "kaggle.com/datasets" in url

    # ------------------------------------------------------------------ #
    # DVC integration
    # ------------------------------------------------------------------ #

    def _dvc_add(self, cache_path: Path) -> None:
        """
        Register *cache_path* with DVC for data lineage via `dvc add`.
        Failures are logged (not raised) so they never block ingestion.
        """
        try:
            result = subprocess.run(
                ["dvc", "add", str(cache_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(
                    "dvc add failed for %s: %s", cache_path, result.stderr.strip()
                )
            else:
                logger.info("DVC: tracked %s", cache_path)
        except FileNotFoundError:
            logger.info("DVC not installed – skipping lineage for %s", cache_path)
        except Exception as exc:
            logger.warning("dvc add error: %s", exc)

    # ------------------------------------------------------------------ #
    # Public async API
    # ------------------------------------------------------------------ #

    async def ingest_data(
        self,
        sources: Union[str, List[str]],
        force_download: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[Dict[str, DatasetObject], Dict[str, Any]]:
        """
        Concurrently ingest datasets from multiple sources.

        Args:
            sources:        Single URL/path or list of URLs/paths.
            force_download: Force re-download even when cached.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple:
              - datasets      : {source_hash -> DatasetObject} (standardized, validated)
              - metadata      : ingestion metadata dict (hashes, failures, timing)
        """
        if isinstance(sources, str):
            sources = [sources]

        metadata: Dict[str, Any] = {
            "sources": sources,
            "ingestion_time": datetime.now().isoformat(),
            "cached_hashes": {},
            "cache_status": {},
            "failed": {},
        }

        tasks = [self._ingest_single(source, force_download, progress_callback) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        datasets: Dict[str, DatasetObject] = {}
        for source, result in zip(sources, results):
            source_hash = self._generate_hash(source)
            if isinstance(result, Exception):
                metadata["failed"][source] = str(result)
                logger.error("Ingestion failed for [%s]: %s", source, result)
            elif result is None:
                metadata["failed"][source] = "Ingestion returned no data"
                logger.error("Ingestion returned None for [%s]", source)
            else:
                lazy_ref, cache_path = result
                
                # Wrap in DatasetObject with metadata
                ingestion_meta = self.cache_metadata.get(
                    source_hash,
                    {"source": source, "cache_path": str(cache_path)},
                )
                dataset_obj = DatasetObject(
                    dataset_id=source_hash,
                    lazy_data=lazy_ref,
                    metadata=ingestion_meta,
                )
                
                # Validate before adding to results
                try:
                    validate_dataset(dataset_obj)
                    datasets[source_hash] = dataset_obj
                    metadata["cached_hashes"][source] = source_hash
                    metadata["cache_status"][source] = "ok"
                    logger.info("Dataset [%s] validated and ready", source_hash)
                except ValueError as ve:
                    metadata["failed"][source] = str(ve)
                    logger.error(
                        "Dataset validation failed for [%s]: %s", source_hash, ve
                    )

        return datasets, metadata

    # ------------------------------------------------------------------ #
    # Per-source dispatch
    # ------------------------------------------------------------------ #

    async def _ingest_single(
        self,
        source: str,
        force_download: bool,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Optional[Tuple[Any, Path]]:
        """
        Ingest one source.  Returns (lazy_ref, cache_path) tuple or raises.
        Exceptions propagate to asyncio.gather for per-URL isolation.

        Backward compatibility: if the normalised hash produces a cache miss,
        falls back to the legacy (raw-string) hash.  If found under the legacy
        key the metadata is migrated to the new normalised key in-place.
        
        Note: Returns raw lazy_ref here (not DatasetObject). Wrapping happens
        in ingest_data() after all ingestion is complete.
        """
        source_hash = self._generate_hash(source)
        cache_path = self.cache_dir / source_hash

        # Reload metadata from disk in case another process updated the cache
        self.cache_metadata = self._load_cache_metadata()

        # Cache hit: try normalised hash first, then legacy raw-string hash
        if not force_download:
            # Try normalised hash
            if source_hash in self.cache_metadata:
                lazy_ref = self._loader.load_cached(cache_path)
                if lazy_ref is not None:
                    logger.info("Cache HIT  [%s] -> %s", source_hash, source)
                    return lazy_ref, cache_path

            # Try legacy hash (raw string, pre-normalisation era)
            legacy_hash = self._legacy_hash(source)
            if legacy_hash != source_hash and legacy_hash in self.cache_metadata:
                legacy_path = self.cache_dir / legacy_hash
                lazy_ref = self._loader.load_cached(legacy_path)
                if lazy_ref is not None:
                    logger.info(
                        "Cache HIT  [%s] (legacy hash for %s) — migrating to [%s]",
                        legacy_hash, source, source_hash,
                    )
                    # Migrate metadata to normalised key
                    self.cache_metadata[source_hash] = self.cache_metadata.pop(legacy_hash)
                    self.cache_metadata[source_hash]["source_hash"] = source_hash
                    self._save_cache_metadata()
                    # Rename directory so future lookups use the normalised hash
                    if legacy_path.exists() and not cache_path.exists():
                        legacy_path.rename(cache_path)
                    return lazy_ref, cache_path if cache_path.exists() else legacy_path

        logger.info("Cache MISS [%s] -> downloading %s", source_hash, source)

        # Route to the right downloader
        if self._is_kaggle_url(source):
            cache_path = await self._ingest_kaggle(source, cache_path, progress_callback)
        elif source.startswith(("http://", "https://")):
            cache_path = await self._ingest_remote_url(source, cache_path)
        else:
            cache_path = await self._ingest_local_path(source, cache_path)

        # DVC lineage tracking (offloaded to thread – subprocess.run is blocking)
        await asyncio.to_thread(self._dvc_add, cache_path)

        # Persist cache metadata (offloaded – json.dump is blocking I/O)
        await asyncio.to_thread(
            self._update_cache_metadata, source_hash, source, cache_path
        )

        lazy_ref = self._loader.load_cached(cache_path)
        if lazy_ref is None:
            raise RuntimeError(
                f"Cache directory {cache_path} has no recognised data files "
                "after ingestion."
            )
        return lazy_ref, cache_path

    # ------------------------------------------------------------------ #
    # Remote URL downloader (aiohttp – truly async)
    # ------------------------------------------------------------------ #

    async def _ingest_remote_url(self, url: str, cache_path: Path) -> Path:
        """
        Download a remote URL with aiohttp.
        Raises FileNotFoundError on 404; ConnectionError on other HTTP errors.
        """
        if "mendeley.com/datasets" in url:
            raise ValueError(
                "Mendeley datasets must be downloaded manually. "
                "Use the local-file upload option instead."
            )

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "data.csv"
        if "." not in filename:
            filename = "data.csv"

        cache_path.mkdir(parents=True, exist_ok=True)
        filepath = cache_path / filename

        # total=7200 (2 h) accommodates 50 GB+ downloads on slower links;
        # sock_read=300 still aborts genuinely stalled connections quickly.
        timeout = aiohttp.ClientTimeout(total=7200, sock_read=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 404:
                    raise FileNotFoundError(f"HTTP 404 Not Found: {url}")
                if response.status != 200:
                    raise ConnectionError(
                        f"HTTP {response.status} while downloading {url}"
                    )
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    raise ValueError(
                        f"URL returns an HTML page, not a direct data file: {url}"
                    )
                # Write chunks via asyncio.to_thread so synchronous disk I/O
                # never blocks the event loop (critical for 50 GB+ files).
                fh = await asyncio.to_thread(open, filepath, "wb")
                try:
                    async for chunk in response.content.iter_chunked(65536):
                        await asyncio.to_thread(fh.write, chunk)
                finally:
                    await asyncio.to_thread(fh.close)

        logger.info("Downloaded %s -> %s", url, filepath)
        return cache_path

    # ------------------------------------------------------------------ #
    # Kaggle downloader (sync, run in thread-pool executor)
    # ------------------------------------------------------------------ #

    async def _ingest_kaggle(self, url: str, cache_path: Path, progress_callback: Optional[Callable[[int, str], None]] = None) -> Path:
        """Offload blocking Kaggle CLI call to a thread-pool executor."""
        return await asyncio.to_thread(
            self._ingest_kaggle_sync, url, cache_path, progress_callback
        )

    def _ingest_kaggle_sync(self, url: str, cache_path: Path, progress_callback: Optional[Callable[[int, str], None]] = None) -> Path:
        """
        Synchronous Kaggle ingestion.
        Credentials are read from environment variables first, then from ~/.kaggle/kaggle.json as fallback.
        """
        kaggle_username: Optional[str] = os.getenv("KAGGLE_USERNAME")
        kaggle_key: Optional[str] = os.getenv("KAGGLE_KEY")
        
        # Fallback: read from ~/.kaggle/kaggle.json if env vars not set
        if not kaggle_username or not kaggle_key:
            kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_json_path.exists():
                try:
                    with open(kaggle_json_path, "r") as f:
                        creds = json.load(f)
                        kaggle_username = creds.get("username")
                        kaggle_key = creds.get("key")
                        logger.info("Loaded Kaggle credentials from ~/.kaggle/kaggle.json")
                except Exception as e:
                    logger.warning("Failed to read kaggle.json: %s", e)
        
        if not kaggle_username or not kaggle_key:
            raise EnvironmentError(
                "Kaggle credentials missing. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables or ensure ~/.kaggle/kaggle.json exists."
            )

        parts = url.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid Kaggle URL format: {url}")
        dataset_id = f"{parts[-2]}/{parts[-1]}"

        temp_dir = Path(tempfile.gettempdir()) / f"kaggle_{parts[-1]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            env = {
                **os.environ,
                "KAGGLE_USERNAME": kaggle_username,
                "KAGGLE_KEY": kaggle_key,
            }
            proc = subprocess.Popen(
                [
                    "kaggle", "datasets", "download",
                    "-d", dataset_id,
                    "-p", str(temp_dir),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            
            error_output = ""
            for line in iter(proc.stdout.readline, ''):
                error_output += line
                if "%|" in line and progress_callback is not None:
                    try:
                        chunk = line.split("%")[0]
                        pct_str = chunk.split()[-1].strip()
                        pct = int(pct_str)
                        if pct > 0:
                            progress_callback(5 + int(pct * 0.8), f"Downloading {dataset_id}... {pct}%")
                    except Exception:
                        pass

            proc.stdout.close()
            returncode = proc.wait(timeout=1800)

            if returncode != 0:
                if "401" in error_output or "Unauthorized" in error_output:
                    raise PermissionError(
                        "Kaggle API authentication failed. Check credentials."
                    )
                raise RuntimeError(f"kaggle CLI failed: {error_output}")

            zip_files = list(temp_dir.glob("*.zip"))
            if not zip_files:
                raise RuntimeError(
                    f"Expected a zip from Kaggle for {dataset_id}, got none."
                )
            for zf in zip_files:
                with zipfile.ZipFile(zf, "r") as zref:
                    # ZipSlip protection: reject members that escape temp_dir
                    for member in zref.namelist():
                        member_path = (temp_dir / member).resolve()
                        if not str(member_path).startswith(str(temp_dir.resolve())):
                            raise ValueError(
                                f"ZipSlip detected: '{member}' escapes target directory"
                            )
                    zref.extractall(temp_dir)
            csv_files = list(temp_dir.rglob("*.csv"))
            image_files = list(temp_dir.rglob("*.jpg")) + list(temp_dir.rglob("*.jpeg")) + list(temp_dir.rglob("*.png"))
            
            if not csv_files and not image_files:
                raise RuntimeError(
                    f"No CSV or Images found in Kaggle archive for {dataset_id}."
                )
                
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Universal Multimodal Extraction: move EVERYTHING from the Kaggle Zip directly to the cache folder!
            for item in temp_dir.iterdir():
                dest = cache_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(cache_path))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("Kaggle dataset %s cached at %s", dataset_id, cache_path)
        return cache_path

    # ------------------------------------------------------------------ #
    # Local path (run in executor to avoid blocking the event loop)
    # ------------------------------------------------------------------ #

    async def _ingest_local_path(self, path: str, cache_path: Path) -> Path:
        return await asyncio.to_thread(
            self._copy_local_path, path, cache_path
        )

    def _copy_local_path(self, path: str, cache_path: Path) -> Path:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Local file not found: {path}")
        cache_path.mkdir(parents=True, exist_ok=True)
        dest = cache_path / src.name
        shutil.copy2(src, dest)
        logger.info("Local file copied %s -> %s", src, dest)
        return cache_path

    # ------------------------------------------------------------------ #
    # Cache metadata persistence
    # ------------------------------------------------------------------ #

    def _update_cache_metadata(
        self,
        source_hash: str,
        source: str,
        cache_path: Path,
    ) -> None:
        meta: Dict[str, Any] = {
            "source": source,
            "source_hash": source_hash,
            "timestamp": datetime.now().isoformat(),
            "cache_path": str(cache_path),
        }
        data_files = (
            list(cache_path.glob("*.parquet"))
            + list(cache_path.glob("*.csv"))
            + list(cache_path.glob("*.xlsx"))
            + list(cache_path.glob("*.xls"))
        )
        if data_files:
            meta["size_mb"] = round(os.path.getsize(data_files[0]) / (1024 * 1024), 3)
        self.cache_metadata[source_hash] = meta
        self._save_cache_metadata()

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            "total_cached": len(self.cache_metadata),
            "cache_dir": str(self.cache_dir),
            "cached_items": list(self.cache_metadata.keys()),
            "metadata": self.cache_metadata,
        }

    def clear_cache(self, source_hash: Optional[str] = None) -> None:
        if source_hash:
            cache_path = self.cache_dir / source_hash
            if cache_path.exists():
                shutil.rmtree(cache_path)
            self.cache_metadata.pop(source_hash, None)
            self._save_cache_metadata()
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {}
            self._save_cache_metadata()
