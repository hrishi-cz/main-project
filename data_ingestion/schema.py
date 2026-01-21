"""Data schema definitions for datasets."""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ColumnSchema:
    """Schema for a data column."""
    name: str
    dtype: str
    modality: str  # 'image', 'tabular', 'text'
    optional: bool = False


class DataSchema:
    """Schema definition for datasets."""
    
    def __init__(self, columns: List[ColumnSchema]):
        self.columns = columns
        self.column_map = {col.name: col for col in columns}
    
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column schema by name."""
        return self.column_map.get(name)
    
    def get_modalities(self) -> List[str]:
        """Get unique modalities in schema."""
        return list(set(col.modality for col in self.columns))
    
    def validate(self, data: Dict) -> bool:
        """Validate data against schema."""
        for col in self.columns:
            if col.name not in data and not col.optional:
                return False
        return True
