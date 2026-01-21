"""Progress display utilities for training monitoring."""

from tqdm import tqdm
from typing import Iterable, Optional


class ProgressDisplay:
    """Display progress for long-running operations."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.pbar = tqdm(total=total, desc=desc)
    
    def update(self, n: int = 1, msg: str = ""):
        """Update progress bar."""
        self.pbar.update(n)
        if msg:
            self.pbar.set_description(f"{self.desc} - {msg}")
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()
    
    @staticmethod
    def wrap_iterable(iterable: Iterable, desc: str = "Progress"):
        """Wrap iterable with progress bar."""
        return tqdm(iterable, desc=desc)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
