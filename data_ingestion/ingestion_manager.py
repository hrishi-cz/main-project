"""Advanced data ingestion with caching, hashing, and multi-source support."""

import os
import hashlib
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import requests
from urllib.parse import urlparse


class DataIngestionManager:
    """Manages data ingestion with caching, validation, and multi-source support."""
    
    def __init__(self, cache_dir: str = "./data/dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.cache_metadata, f, indent=2)
    
    def _generate_hash(self, source: str) -> str:
        """Generate SHA-256 hash of data source."""
        hash_obj = hashlib.sha256(source.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars
    
    def _is_kaggle_url(self, url: str) -> bool:
        """Check if URL is Kaggle dataset."""
        return "kaggle.com/datasets" in url
    
    def _download_file(self, url: str, output_path: Path, progress_callback=None) -> bool:
        """Download file from URL with progress tracking."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress)
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def ingest_data(
        self,
        sources: Union[str, List[str]],
        progress_callback=None,
        force_download: bool = False
    ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Ingest data from multiple sources with caching.
        
        Args:
            sources: Single URL/path or list of URLs/paths
            progress_callback: Callback function for progress updates
            force_download: Force download even if cached
        
        Returns:
            Tuple of (loaded_data_dict, metadata_dict)
        """
        if isinstance(sources, str):
            sources = [sources]
        
        loaded_data = {}
        metadata = {
            "sources": sources,
            "ingestion_time": datetime.now().isoformat(),
            "cached_hashes": {},
            "modalities": {},
            "cache_status": {}  # Track which datasets were cached vs downloaded
        }
        
        for source in sources:
            source_hash = self._generate_hash(source)
            cache_path = self.cache_dir / source_hash
            
            # Check cache
            if source_hash in self.cache_metadata and not force_download:
                if progress_callback:
                    progress_callback(50, f"Loading cached data from {source_hash}")
                
                cached_data = self._load_cached_data(cache_path, source_hash)
                if cached_data is not None:
                    loaded_data[source_hash] = cached_data
                    metadata["cached_hashes"][source] = source_hash
                    metadata["cache_status"][source] = "cached"  # Track as cached
                    if progress_callback:
                        progress_callback(100, f"Loaded from cache: {source_hash}")
                    continue
            
            # Cache miss or force download
            if progress_callback:
                progress_callback(10, f"Downloading from {source}")
            
            # Handle different source types
            if self._is_kaggle_url(source):
                data = self._ingest_kaggle(source, cache_path, progress_callback)
            elif source.startswith(('http://', 'https://')):
                data = self._ingest_remote_url(source, cache_path, progress_callback)
            else:
                data = self._ingest_local_path(source, cache_path, progress_callback)
            
            if data is not None:
                loaded_data[source_hash] = data
                metadata["cached_hashes"][source] = source_hash
                metadata["cache_status"][source] = "downloaded"  # Track as newly downloaded
                
                # Save to cache
                self._save_to_cache(cache_path, source_hash, data, source)
                
                if progress_callback:
                    progress_callback(100, f"Data ready from {source}")
        
        return loaded_data, metadata
    
    def _ingest_kaggle(self, url: str, cache_path: Path, progress_callback=None) -> Optional[pd.DataFrame]:
        """Ingest Kaggle dataset using kaggle-api."""
        try:
            import subprocess
            import os
            import tempfile
            
            if progress_callback:
                progress_callback(10, "Parsing Kaggle dataset URL...")
            
            # Extract dataset identifier from URL
            # URL format: https://www.kaggle.com/datasets/{owner}/{dataset-name}
            parts = url.strip('/').split('/')
            if len(parts) < 2:
                raise ValueError(f"Invalid Kaggle URL: {url}")
            
            dataset_id = f"{parts[-2]}/{parts[-1]}"
            
            if progress_callback:
                progress_callback(20, f"Downloading: {dataset_id}")
            
            # Create temp directory for download
            temp_dir = Path(tempfile.gettempdir()) / f"kaggle_{parts[-1]}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset using kaggle CLI
            try:
                result = subprocess.run(
                    ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(temp_dir)],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes for large downloads
                )
                
                if result.returncode != 0:
                    if "401" in result.stderr or "Unauthorized" in result.stderr:
                        raise Exception("Kaggle API key not found. Set up credentials at ~/.kaggle/kaggle.json")
                    raise Exception(f"Kaggle download failed: {result.stderr}")
                
            except FileNotFoundError:
                raise Exception("kaggle CLI not found. Install with: pip install kaggle")
            
            if progress_callback:
                progress_callback(50, "Extracting archive...")
            
            # Find any downloaded files
            all_files = list(temp_dir.glob("*"))
            if not all_files:
                raise Exception(f"Kaggle download returned no files. Check dataset ID: {dataset_id}")
            
            # Find and extract any zip files
            import zipfile
            zip_files = list(temp_dir.glob("*.zip"))
            
            if not zip_files:
                # Check what was actually downloaded
                file_list = [f.name for f in all_files]
                raise Exception(f"Expected zip file from Kaggle, but got: {file_list}. Dataset may require login or be restricted.")
            
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    zip_file.unlink()  # Delete zip after extraction
                except zipfile.BadZipFile:
                    raise Exception(f"Downloaded file is not a valid zip: {zip_file.name}")
            
            if progress_callback:
                progress_callback(70, "Loading dataset...")
            
            # Find CSV file
            csv_files = list(temp_dir.glob("*.csv"))
            if not csv_files:
                # Try nested directories
                csv_files = list(temp_dir.rglob("*.csv"))
            
            if not csv_files:
                # List what files we have
                all_extracted = list(temp_dir.rglob("*"))
                file_types = set([f.suffix for f in all_extracted if f.is_file()])
                raise Exception(f"No CSV files found. Available file types: {file_types}")
            
            # Load first CSV (main dataset)
            csv_file = csv_files[0]
            data = pd.read_csv(csv_file, nrows=10000)  # Limit rows for memory
            
            if progress_callback:
                progress_callback(90, "Caching dataset...")
            
            # Copy to cache
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = cache_path / csv_file.name
            data.to_csv(cache_file, index=False)
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if progress_callback:
                progress_callback(100, f"Kaggle dataset loaded: {len(data)} rows")
            
            return data
            
        except Exception as e:
            print(f"Error ingesting Kaggle dataset: {e}")
            return None
    
    def _ingest_remote_url(self, url: str, cache_path: Path, progress_callback=None) -> Optional[pd.DataFrame]:
        """Ingest data from remote URL."""
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Check if Mendeley URL
            if "mendeley.com/datasets" in url:
                raise Exception(
                    "Mendeley datasets require manual download. Please:\n"
                    "1. Visit: " + url + "\n"
                    "2. Click the download button on their website\n"
                    "3. Upload the file using the 'Upload Local File' option instead"
                )
            
            # Determine file type from URL
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or "data.csv"
            
            # If no extension, try to get it from the response
            if '.' not in filename:
                filename = "data.csv"  # default
            
            filepath = cache_path / filename
            
            if progress_callback:
                progress_callback(20, "Attempting download...")
            
            # Try to download
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if it's actually a file (not HTML page)
            if 'text/html' in content_type:
                # This is a web page, not a downloadable file
                if progress_callback:
                    progress_callback(50, "Searching for downloadable data...")
                
                # Try GET to look for download links in HTML
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Check if there are any data files mentioned
                    if '.csv' in response.text.lower() or '.xlsx' in response.text.lower():
                        raise Exception(f"This is a dataset page (not a direct file). Please download the file manually and provide the direct link or file path.")
                    else:
                        raise Exception(f"Invalid URL: No data file found at this location")
                else:
                    raise Exception(f"Failed to access URL: HTTP {response.status_code}")
            
            # Download the actual file
            if not self._download_file(url, filepath, progress_callback):
                raise Exception("Download failed - check URL and internet connection")
            
            if progress_callback:
                progress_callback(80, "Loading data...")
            
            # Load data based on file type
            if filename.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filename.endswith('.parquet'):
                return pd.read_parquet(filepath)
            elif filename.endswith('.json'):
                return pd.read_json(filepath)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                return pd.read_excel(filepath)
            else:
                # Try to detect by trying to load as different formats
                try:
                    return pd.read_csv(filepath)
                except:
                    try:
                        return pd.read_parquet(filepath)
                    except:
                        try:
                            return pd.read_json(filepath)
                        except:
                            raise Exception(f"Unsupported file format: {filename}")
        except Exception as e:
            if progress_callback:
                progress_callback(100, f"Load failed: {str(e)[:50]}")
            print(f"Error ingesting remote URL: {e}")
            return None
    
    def _ingest_local_path(self, path: str, cache_path: Path, progress_callback=None) -> Optional[pd.DataFrame]:
        """Ingest data from local path."""
        try:
            if progress_callback:
                progress_callback(30, f"Loading local file...")
            
            if path.endswith('.csv'):
                data = pd.read_csv(path)
            elif path.endswith('.parquet'):
                data = pd.read_parquet(path)
            elif path.endswith('.json'):
                data = pd.read_json(path)
            else:
                return None
            
            if progress_callback:
                progress_callback(100, "Local data loaded")
            
            return data
        except Exception as e:
            print(f"Error ingesting local path: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, source_hash: str, data: pd.DataFrame, source: str):
        """Save data and metadata to cache."""
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_file = cache_path / "data.parquet"
        data.to_parquet(data_file)
        
        # Save metadata
        cache_meta = {
            "source": source,
            "source_hash": source_hash,
            "timestamp": datetime.now().isoformat(),
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "size_mb": os.path.getsize(data_file) / (1024 * 1024),
        }
        
        meta_file = cache_path / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(cache_meta, f, indent=2)
        
        # Update global cache metadata
        self.cache_metadata[source_hash] = cache_meta
        self._save_cache_metadata()
    
    def _load_cached_data(self, cache_path: Path, source_hash: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        try:
            data_file = cache_path / "data.parquet"
            if data_file.exists():
                return pd.read_parquet(data_file)
        except Exception as e:
            print(f"Error loading cached data: {e}")
        return None
    
    def get_cache_info(self) -> Dict:
        """Get information about cached datasets."""
        return {
            "total_cached": len(self.cache_metadata),
            "cache_dir": str(self.cache_dir),
            "cached_items": list(self.cache_metadata.keys()),
            "metadata": self.cache_metadata
        }
    
    def clear_cache(self, source_hash: Optional[str] = None):
        """Clear cache directory."""
        if source_hash:
            cache_path = self.cache_dir / source_hash
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                if source_hash in self.cache_metadata:
                    del self.cache_metadata[source_hash]
                self._save_cache_metadata()
        else:
            import shutil
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {}
            self._save_cache_metadata()
