"""Advanced schema detection with column type and problem type inference."""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# Constants
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
TIMESERIES_EXTENSIONS = {'.npy', '.wav', '.csv', '.parquet', '.h5', '.hdf5', '.mat'}
TARGET_KEYWORDS = {'target', 'label', 'diagnosis', 'class', 'category', 'outcome', 'result', 'scp_codes', 'diagnostic'}

def is_json_like(value: str) -> bool:
    """Check if string is JSON/dict-like."""
    try:
        json.loads(value)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def is_multi_label(col_data: pd.Series) -> bool:
    """Check if column contains multi-label values (lists, dicts, comma/pipe separated)."""
    sample = col_data.dropna().head(50).astype(str)
    if len(sample) == 0:
        return False
    
    multi_label_indicators = 0
    for val in sample:
        val_str = str(val).strip()
        # Check for JSON/dict format: {"key": "value"}
        if val_str.startswith('{') or val_str.startswith('['):
            multi_label_indicators += 1
        # Check for comma/pipe separated with multiple items
        elif ',' in val_str or '|' in val_str:
            parts = re.split(r'[,|]', val_str)
            if len(parts) > 1 and any(p.strip() for p in parts):
                multi_label_indicators += 1
    
    return (multi_label_indicators / len(sample)) > 0.3


def simple_similarity(s1: str, s2: str) -> float:
    """Simple string similarity ratio (0-100) using basic matching."""
    s1 = s1.lower()
    s2 = s2.lower()
    
    # Exact match
    if s1 == s2:
        return 100
    
    # Substring match
    if s1 in s2 or s2 in s1:
        return 85
    
    # Character overlap ratio
    s1_set = set(s1)
    s2_set = set(s2)
    overlap = len(s1_set & s2_set)
    union = len(s1_set | s2_set)
    
    if union == 0:
        return 0
    
    return (overlap / union) * 100


@dataclass
class DetectedColumn:
    """Detected column information."""
    name: str
    modality: str  # 'image', 'text', 'tabular', 'timeseries', 'multi-label'
    dtype: str
    cardinality: int
    sample_values: List


@dataclass
class SchemaDetectionResult:
    """Result of schema detection."""
    image_cols: List[str]
    text_cols: List[str]
    tabular_cols: List[str]
    target_col: Optional[str]
    problem_type: str  # 'classification_binary', 'classification_multiclass', 'regression'
    detected_columns: List[Dict]
    detection_confidence: float
    modalities: List[str]


class SchemaDetector:
    """Automatically detect schema and problem type from data."""
    
    # Image file extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    # Target column fuzzy match keywords
    TARGET_KEYWORDS = ['label', 'target', 'class', 'y', 'outcome', 'result', 'prediction']
    
    def __init__(self, text_length_threshold: int = 50, fuzzy_threshold: int = 75):
        self.text_length_threshold = text_length_threshold
        self.fuzzy_threshold = fuzzy_threshold
    
    def detect_schema(
        self,
        data: pd.DataFrame,
        progress_callback=None
    ) -> SchemaDetectionResult:
        """
        Automatically detect schema from data.
        
        Args:
            data: Input DataFrame
            progress_callback: Progress callback function
        
        Returns:
            SchemaDetectionResult with detected columns and problem type
        """
        if progress_callback:
            progress_callback(10, "Starting schema detection...")
        
        # Phase 1: Detect column types
        if progress_callback:
            progress_callback(20, "Detecting column types...")
        
        detected_columns = []
        image_cols = []
        text_cols = []
        tabular_cols = []
        timeseries_cols = []
        multi_label_cols = []
        for col in data.columns:
            detected = self._detect_column_type(col, data[col])
            detected_columns.append(detected)
            if detected.modality == 'image':
                image_cols.append(col)
            elif detected.modality == 'text':
                text_cols.append(col)
            elif detected.modality == 'tabular':
                tabular_cols.append(col)
            elif detected.modality == 'timeseries':
                timeseries_cols.append(col)
            elif detected.modality == 'multi-label':
                multi_label_cols.append(col)
        if progress_callback:
            progress_callback(40, f"Detected {len(image_cols)} image, {len(text_cols)} text, {len(tabular_cols)} tabular, {len(timeseries_cols)} timeseries, {len(multi_label_cols)} multi-label columns")
        
        # Phase 2: Detect target column
        if progress_callback:
            progress_callback(50, "Detecting target column...")
        
        target_col = self._detect_target_column(data, tabular_cols, image_cols, text_cols, timeseries_cols, multi_label_cols)
        
        if progress_callback:
            progress_callback(60, f"Target column: {target_col or 'Not found'}")
        
        # Phase 3: Infer problem type
        if progress_callback:
            progress_callback(70, "Inferring problem type...")
        
        problem_type = "unsupervised"
        if target_col is not None:
            problem_type = self._infer_problem_type(data[target_col])
        
        if progress_callback:
            progress_callback(90, f"Problem type: {problem_type}")
        
        # Phase 4: Calculate confidence
        modalities = list(set([col.modality for col in detected_columns]))
        confidence = self._calculate_confidence(data, target_col, modalities)
        
        if progress_callback:
            progress_callback(100, f"Detection complete (confidence: {confidence:.2%})")
        
        return SchemaDetectionResult(
            image_cols=image_cols,
            text_cols=text_cols,
            tabular_cols=tabular_cols,
            target_col=target_col,
            problem_type=problem_type,
            detected_columns=[asdict(col) for col in detected_columns],
            detection_confidence=confidence,
            modalities=modalities
        )
    
    def _detect_column_type(self, col_name: str, col_data: pd.Series) -> DetectedColumn:
        """Detect column modality (image, text, tabular, timeseries, multi-label)."""
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return DetectedColumn(
                name=col_name,
                modality='tabular',
                dtype='unknown',
                cardinality=0,
                sample_values=[]
            )

        # Image detection: file paths with image extensions
        if col_data.dtype == 'object':
            sample = non_null.head(20).astype(str)
            image_count = sum(1 for val in sample if any(val.lower().endswith(ext) for ext in IMAGE_EXTENSIONS))
            if image_count / len(sample) > 0.5:
                return DetectedColumn(
                    name=col_name,
                    modality='image',
                    dtype='str',
                    cardinality=len(non_null.unique()),
                    sample_values=non_null.head(3).tolist()
                )

        # Timeseries detection: arrays, lists, signal file references
        if col_data.dtype == 'object':
            sample = non_null.head(20).astype(str)
            ts_count = sum(1 for val in sample if val.startswith('[') or val.startswith('{') or any(val.lower().endswith(ext) for ext in TIMESERIES_EXTENSIONS))
            if ts_count / len(sample) > 0.3:
                return DetectedColumn(
                    name=col_name,
                    modality='timeseries',
                    dtype='object',
                    cardinality=len(non_null.unique()),
                    sample_values=non_null.head(3).tolist()
                )

        # Multi-label detection: lists, dicts, comma/pipe separated
        if is_multi_label(non_null):
            return DetectedColumn(
                name=col_name,
                modality='multi-label',
                dtype='object',
                cardinality=len(non_null.unique()),
                sample_values=non_null.head(3).tolist()
            )

        # Text detection: long strings
        if col_data.dtype == 'object':
            sample = non_null.head(50).astype(str)
            avg_length = sample.str.len().mean()
            is_long_text = avg_length > self.text_length_threshold
            is_mostly_strings = sample.str.len().std() > 10
            if is_long_text and is_mostly_strings:
                return DetectedColumn(
                    name=col_name,
                    modality='text',
                    dtype='str',
                    cardinality=len(non_null.unique()),
                    sample_values=non_null.head(3).tolist()
                )

        # Default: tabular (numeric or categorical)
        return DetectedColumn(
            name=col_name,
            modality='tabular',
            dtype=str(col_data.dtype),
            cardinality=len(non_null.unique()),
            sample_values=non_null.head(3).tolist()
        )
    
    def _is_image_column(self, col_data: pd.Series) -> bool:
        """Check if column contains image paths."""
        if col_data.dtype != 'object':
            return False
        
        # Check sample values for image extensions
        sample = col_data.head(20).astype(str)
        image_count = sum(1 for val in sample if any(val.lower().endswith(ext) for ext in self.IMAGE_EXTENSIONS))
        
        return image_count / len(sample) > 0.5
    
    def _is_text_column(self, col_data: pd.Series) -> bool:
        """Check if column contains text data."""
        if col_data.dtype != 'object':
            return False
        
        # Get non-null values for checking
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return False
        
        # Check average string length of actual values
        sample = non_null.head(50).astype(str)
        avg_length = sample.str.len().mean()
        
        # Must have average length > threshold AND be mostly strings (not short categories)
        is_long_text = avg_length > self.text_length_threshold
        is_mostly_strings = sample.str.len().std() > 10  # Varied length indicates text
        
        return is_long_text and is_mostly_strings
    
    def _detect_target_column(
        self,
        data: pd.DataFrame,
        tabular_cols: List[str],
        image_cols: List[str],
        text_cols: List[str],
        timeseries_cols: List[str],
        multi_label_cols: List[str]
    ) -> Optional[str]:
        """Detect target column using fuzzy name, cardinality, multi-label, and heuristics."""
        candidates = multi_label_cols + tabular_cols + timeseries_cols + text_cols + image_cols
        if not candidates:
            return None
        best_match = None
        best_score = 0
        for col in candidates:
            col_name_lower = col.lower()
            keyword_score = max((simple_similarity(col_name_lower, keyword), keyword) for keyword in TARGET_KEYWORDS)
            keyword_score = keyword_score[0] if keyword_score else 0
            unique_count = len(data[col].unique())
            cardinality_ratio = unique_count / len(data)
            # Multi-label bonus
            multi_label_bonus = 20 if is_multi_label(data[col]) else 0
            # Cardinality scoring
            if cardinality_ratio < 0.05:
                cardinality_score = 0.3
            elif cardinality_ratio < 0.5:
                cardinality_score = 1.0
            elif cardinality_ratio < 0.8:
                cardinality_score = 0.7
            else:
                cardinality_score = 0.2
            is_categorical = not pd.api.types.is_numeric_dtype(data[col])
            categorical_score = 0.8 if is_categorical else 0.3
            combined_score = (keyword_score * 0.3) + (cardinality_score * 0.3) + (categorical_score * 0.2) + multi_label_bonus
            if combined_score > best_score:
                best_score = combined_score
                best_match = col
        if best_score < 40 and candidates:
            best_match = candidates[-1]
        return best_match
    
    def _infer_problem_type(self, target_col: pd.Series) -> str:
        """Infer problem type: binary, multi-class, multi-label, regression, timeseries."""
        unique_values = target_col.nunique()
        if is_multi_label(target_col):
            return "classification_multilabel"
        if pd.api.types.is_numeric_dtype(target_col):
            if unique_values <= 20:
                if unique_values == 2:
                    return "classification_binary"
                else:
                    return "classification_multiclass"
            else:
                return "regression"
        else:
            if unique_values == 2:
                return "classification_binary"
            elif unique_values <= 20:
                return "classification_multiclass"
            else:
                return "classification_multilabel"
    
    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        target_col: Optional[str],
        modalities: List[str]
    ) -> float:
        """Calculate overall detection confidence."""
        confidence = 0.6  # Base confidence (higher for real data)
        
        # Sufficient columns indicate real data
        if len(data.columns) >= 3:
            confidence += 0.15
        
        # Sufficient samples indicate real data
        if len(data) >= 50:
            confidence += 0.15
        
        # Target found = higher confidence
        if target_col is not None:
            confidence += 0.1
        
        return min(confidence, 1.0)


class MultiDatasetSchemaDetector:
    """Detect and merge schemas from multiple datasets."""
    
    def __init__(self):
        self.detector = SchemaDetector()
    
    def detect_schema(
        self,
        datasets: Dict[str, pd.DataFrame],
        progress_callback=None
    ) -> Dict:
        """
        Detect schemas from multiple datasets.
        
        Args:
            datasets: Dictionary of {dataset_hash: DataFrame}
            progress_callback: Progress callback
        
        Returns:
            Combined schema information
        """
        # Use the existing detect_and_merge_schemas method
        merged_result = self.detect_and_merge_schemas(datasets, progress_callback)
        
        # Extract key information for API response
        detected_schemas = merged_result.get("detected_schemas", {})
        merged_info = merged_result.get("merged", {})
        
        # Get first schema's details for summary
        first_schema = next(iter(detected_schemas.values())) if detected_schemas else None
        
        return {
            "status": "success",
            "datasets_analyzed": len(datasets),
            "target_column": first_schema.target_col if first_schema else None,
            "problem_type": first_schema.problem_type if first_schema else "unsupervised",
            "modalities": merged_info.get("modalities", []),
            "detection_confidence": first_schema.detection_confidence if first_schema else 0.0,
            "detected_schemas": {
                hash_id: asdict(schema) 
                for hash_id, schema in detected_schemas.items()
            }
        }
    
    def detect_and_merge_schemas(
        self,
        datasets: Dict[str, pd.DataFrame],
        progress_callback=None
    ) -> Dict:
        """
        Detect schemas from multiple datasets and merge them.
        
        Args:
            datasets: Dictionary of {dataset_hash: DataFrame}
            progress_callback: Progress callback
        
        Returns:
            Merged schema information
        """
        detected_schemas = {}
        all_modalities = set()
        all_image_cols = set()
        all_text_cols = set()
        all_tabular_cols = set()
        
        for i, (hash_id, data) in enumerate(datasets.items()):
            if progress_callback:
                progress = (i / len(datasets)) * 100
                progress_callback(progress, f"Detecting schema for dataset {i+1}/{len(datasets)}")
            
            schema = self.detector.detect_schema(data)
            detected_schemas[hash_id] = schema
            
            all_modalities.update(schema.modalities)
            all_image_cols.update(schema.image_cols)
            all_text_cols.update(schema.text_cols)
            all_tabular_cols.update(schema.tabular_cols)
        
        if progress_callback:
            progress_callback(100, "Schema detection complete")
        
        return {
            "detected_schemas": detected_schemas,
            "merged": {
                "modalities": list(all_modalities),
                "image_cols_count": len(all_image_cols),
                "text_cols_count": len(all_text_cols),
                "tabular_cols_count": len(all_tabular_cols),
                "problem_types": [s.problem_type for s in detected_schemas.values()],
            }
        }
