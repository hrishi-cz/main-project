"""
INTEGRATION GUIDE: How to wire FIX-4 into apex2-worktree

==================================================================
OBJECTIVE
==================================================================
Replace SIFT/OCAM/TF-IDF heuristics with unified modality pipeline.
Expected improvement: image/text accuracy 60% → 85-95%.

# COMPONENTS CREATED

✓ A.3a: ModalityEncoder (modality_encoder.py)
Detect + encode any modality → embeddings

✓ A.3b: UniversalTargetValidator (target_validator.py)
Learn predictability metric for any modality

✓ A.3c: Integrator (integrator.py)
Orchestrate full pipeline: detect → encode → validate

✓ A.3d: THIS GUIDE (integration_guide.md)
How to wire into existing code

==================================================================
STEP 1: Update schema_detector.py (Entry Point)
==================================================================

FILE: data_ingestion/schema_detector.py

BEFORE (heuristic-based):
def detect_schema(...):
for col in columns:
if is_image_column(col): # Use SIFT heuristics
...
elif is_text_column(col): # Use TF-IDF heuristics
...

AFTER (ML-based):
from .integrator import Integrator, ModalityMetadata

def detect_schema(...):
integrator = Integrator(min_predictability=0.3)

      # Process all modalities
      modality_results = integrator.process_multimodal(
          raw_data_dict=column_data,  # {"image_col": [...], "text_col": [...]}
          y=target_variable,          # For validation
          task_type="regression",     # Or "classification"
      )

      # Extract embeddings + metadata
      for field_name, metadata in modality_results.items():
          embeddings = metadata.embeddings  # (N, D)
          score = metadata.final_score()

          schema[field_name] = {
              "modality": metadata.modality_name,
              "predictability": metadata.predictability_score,
              "complementarity": metadata.complementarity_score,
              "encoder": metadata.encoder_name,
              "is_valid": metadata.is_valid,
          }

          # Store embeddings for inference
          embedding_cache[field_name] = embeddings

CHANGES TO MAKE:

1. Add import: from .integrator import Integrator
2. Instantiate: integrator = Integrator()
3. Replace heuristics with: integrator.process_multimodal(...)
4. Store embeddings in pipeline state for inference

==================================================================
STEP 2: Update inference_engine.py (Use Encoded Features)
==================================================================

FILE: pipeline/inference_engine.py

BEFORE (raw features or simple heuristics):
def predict(input_data): # Manually concat raw features
features = []
for col in input_data:
features.append(raw_values[col])
X = np.concatenate(features, axis=1)
return model.predict(X)

AFTER (use embeddings):
from data_ingestion.integrator import Integrator

class InferenceEngine:
def **init**(self):
self.integrator = Integrator()
self.embedding_cache = {} # From schema_detector

      def predict(self, input_data):
          # Encode input using same integrator
          embeddings_dict = {}
          for field_name, raw_data in input_data.items():
              metadata = self.integrator.process_single_modality(
                  raw_data=raw_data,
              )
              embeddings_dict[field_name] = metadata.embeddings

          # Concat all embeddings
          all_embeddings = np.concatenate(
              list(embeddings_dict.values()), axis=1
          )

          # Predict
          return self.model.predict(all_embeddings)

==================================================================
STEP 3: Update training_orchestrator.py (Store Metadata)
==================================================================

FILE: pipeline/training_orchestrator.py

ADD:
def train_with_modality_analysis(self, X_raw, y): # Phase 1: Modality analysis
from data_ingestion.integrator import Integrator

      integrator = Integrator()
      modality_results = integrator.process_multimodal(
          raw_data_dict=X_raw,
          y=y,
          task_type=self.task_type,
      )

      # Log scores
      logger.info("Modality analysis:")
      for field_name, metadata in modality_results.items():
          logger.info(f"  {field_name}: score={metadata.final_score():.3f}")

      # Save metadata
      self.modality_metadata = modality_results

      # Phase 2: Train on embeddings
      embeddings_list = [
          metadata.embeddings
          for metadata in modality_results.values()
      ]
      X_embeddings = np.concatenate(embeddings_list, axis=1)

      # Train model on X_embeddings
      self.model.fit(X_embeddings, y)

==================================================================
STEP 4: Update orchestrator.py (High-level coordinator)
==================================================================

FILE: pipeline/orchestrator.py

ADD METHOD:
def analyze_data_modalities(self, dataset):
"""Analyze and validate all modalities in dataset."""
from data_ingestion.integrator import process_dataset

      modality_results = process_dataset(
          raw_data_dict=dataset.X_raw,
          y=dataset.y,
          task_type=self.task_type,
      )

      # Decide which modalities to use
      valid_modalities = {
          field: meta for field, meta in modality_results.items()
          if meta.is_valid
      }

      logger.info(
          "Using %d/%d modalities: %s",
          len(valid_modalities),
          len(modality_results),
          list(valid_modalities.keys()),
      )

      return modality_results, valid_modalities

==================================================================
STEP 5: Update embedding_cache.py (Pre-compute & cache)
==================================================================

FILE: pipeline/embedding_cache.py

ADD:
from data_ingestion.integrator import Integrator

class ModalityEmbeddingCache:
def **init**(self):
self.cache = {} # field_name → embeddings
self.metadata = {} # field_name → ModalityMetadata

      def compute_and_cache(self, raw_data_dict, y=None):
          """Pre-compute all embeddings once at setup."""
          integrator = Integrator()
          results = integrator.process_multimodal(raw_data_dict, y=y)

          for field_name, metadata in results.items():
              self.cache[field_name] = metadata.embeddings
              self.metadata[field_name] = metadata

          return self.cache, self.metadata

      def get_embeddings(self, field_name):
          """Retrieve cached embeddings."""
          return self.cache.get(field_name)

==================================================================
DEPENDENCIES & IMPORTS
==================================================================

Add to requirements.txt:
scikit-learn>=1.0.0 (RandomForest, cross_val_score)
sentence-transformers>=2.2.0 (TextEncoder)
opencv-python>=4.5.0 (ImageEncoder - if using SIFT)
pillow>=8.0.0 (Image processing)

In each file that uses integrator:
from data_ingestion.integrator import Integrator, ModalityMetadata, process_dataset
from data_ingestion.modality_encoder import ModalityEncoder
from data_ingestion.target_validator import UniversalTargetValidator

==================================================================
CONFIGURATION
==================================================================

In config/hyperparameters.py, ADD:

# Modality encoding/validation config

MODALITY_CONFIG = {
"min_predictability": 0.3, # Min score to include modality
"cv_folds": 3, # CV folds for validation
"text_encoder": "sentence-transformers/all-mpnet-base-v2",
"image_encoder": "SIFT", # Or "ResNet50", "ViT-B16"
"max_features": 50, # Max RF features
}

In code:
from config.hyperparameters import MODALITY_CONFIG

integrator = Integrator(
min_predictability=MODALITY_CONFIG["min_predictability"],
)

==================================================================
TESTING & VALIDATION
==================================================================

Write tests in tests/test_modality_pipeline.py:

import pytest
from data_ingestion.integrator import Integrator, process_dataset

def test_auto_detect():
integrator = Integrator()

      # Test image detection
      images = [PIL.Image.new('RGB', (64, 64))] * 10
      mod = integrator.detect_modality(images)
      assert mod == "image"

      # Test text detection
      texts = ["hello world", "foo bar"] * 5
      mod = integrator.detect_modality(texts)
      assert mod == "text"

def test_encode_and_validate():
integrator = Integrator()
y = np.random.rand(10)

      # Create test data
      texts = ["hello"] * 10

      # Process
      meta = integrator.process_single_modality(texts, y=y)

      # Check outputs
      assert meta.embeddings.shape[0] == 10
      assert 0 <= meta.final_score() <= 1
      assert meta.is_valid == (meta.final_score() >= 0.3)

def test_multimodal():
raw_data = {
"text": ["hello"] \* 10,
"tabular": np.random.rand(10, 5),
}
y = np.random.rand(10)

      results = process_dataset(raw_data, y=y)

      assert "text" in results
      assert "tabular" in results
      for meta in results.values():
          assert meta.embeddings.shape[0] == 10

RUN TESTS:
pytest tests/test_modality_pipeline.py -v

==================================================================
EXPECTED IMPROVEMENTS & METRICS
==================================================================

BEFORE (Heuristic):
Image accuracy: 60-70% (SIFT + SVM)
Text accuracy: 60-70% (TF-IDF + LR)
Tabular: 85-90%

AFTER (ML-based):
Image accuracy: 85-95% (ResNet50/ViT embeddings + RF validation)
Text accuracy: 85-95% (sentence-transformers + RF validation)
Tabular: 85-90% (unchanged)

METRICS TO TRACK:

1. Modality predictability scores (before integration)
2. Final model accuracy on holdout (after training on embeddings)
3. Cross-validation stability (variance of CV scores)
4. Feature importance (concentration of signal)

LOG OUTPUT EXAMPLE:
INFO: Integrator initialized: min_predictability=0.300
INFO: Auto-detected modality: text
INFO: Encoded text: embeddings shape = (1000, 768), encoder = sentence-transformers/all-mpnet-base-v2
INFO: Modality 'text' scores: pred=0.850 comp=0.600 degen=0.950 noise=0.820 feat=0.780 → final=0.822
INFO: Processed field text_col → modality text, score 0.822
INFO: Using 3/4 modalities: ['image_col', 'text_col', 'tabular_col']

==================================================================
ROLLBACK PLAN (if needed)
==================================================================

If issues arise, revert to heuristic path:

1. Keep old code in data_ingestion/schema_detector_old.py
2. In schema_detector.py, add flag:

   USE_MODALITY_PIPELINE = True # Set to False to use old code

   if USE_MODALITY_PIPELINE: # Use Integrator (A.3c)
   results = integrator.process_multimodal(...)
   else: # Fall back to old SIFT/TF-IDF
   results = old_detect_schema(...)

3. If unstable, gradually switch on per-modality:

   if modality == "image" and USE_IMAGE_PIPELINE: # Use new pipeline
   else: # Use old pipeline

==================================================================
TIMELINE
==================================================================

Week 1:

- Implement A.3a, A.3b, A.3c, integrate into schema_detector.py
- Run unit tests
- Benchmark on dev dataset

Week 2:

- Integrate into inference_engine.py
- Test end-to-end predictions
- Update monitoring to track modality scores

Week 3:

- A/B test new vs old on holdout set
- Tune MODALITY_CONFIG thresholds
- Document results

==================================================================
SUCCESS CRITERIA
==================================================================

✓ All 3 components (encoder, validator, integrator) working
✓ Auto-detect >= 95% accurate on common modalities
✓ Embedding shape correct for each modality
✓ Validation scores in [0, 1], sensible values
✓ Integration into schema_detector.py complete
✓ Image/text accuracy improves ≥ 15 percentage points
✓ No errors in production inference
✓ Modality scores logged and monitored

==================================================================
"""

# This is a markdown/doc file. Run through the integration checklist before deploying.
