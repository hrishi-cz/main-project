"""
Production-grade drift detector using rigorous statistical tests.

Metrics implemented
-------------------
KS  (Kolmogorov-Smirnov)       – ``scipy.stats.ks_2samp`` per numerical
                                  feature; report the *maximum* statistic
                                  across all features.
PSI (Population Stability Index)– Binning-based stability measure computed
                                  per feature and averaged across features.
                                  Formula: Σ (prod_% − ref_%) × ln(prod_%/ref_%)
FDD (Feature Drift Distance)   – Maximum Mean Discrepancy (MMD) with an
                                  RBF kernel evaluated on a random subsample
                                  of up to 500 rows.  Captures multivariate
                                  distributional shift in the full feature
                                  space.

PDF Thresholds (hardcoded)
--------------------------
PSI  > 0.25  →  drift
KS   > 0.30  →  drift
FDD  > 0.50  →  drift

If **any** threshold is breached, ``DriftReport.drift_detected`` is ``True``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF thresholds (fixed per spec)
# ---------------------------------------------------------------------------

PSI_THRESHOLD: float = 0.25
KS_THRESHOLD:  float = 0.30
FDD_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """
    Complete output of a single drift detection run.

    Attributes
    ----------
    psi            : Mean PSI across all features (scalar).
    ks_statistic   : Maximum KS statistic across all features (scalar).
    fdd            : Feature Drift Distance (MMD, scalar).
    drift_detected : ``True`` when **any** metric breaches its threshold.
    status         : Per-metric breach flags:
                     ``{"psi": bool, "ks_statistic": bool, "fdd": bool}``.
    per_feature_ks  : KS statistic for each individual feature.
    per_feature_psi : PSI value for each individual feature.
    n_features      : Number of features analysed.
    n_reference     : Sample count in the reference (older) split.
    n_production    : Sample count in the production (recent) split.
    """

    psi: float
    ks_statistic: float
    fdd: float
    drift_detected: bool
    status: Dict[str, bool] = field(default_factory=dict)
    per_feature_ks: Dict[str, float] = field(default_factory=dict)
    per_feature_psi: Dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    n_reference: int = 0
    n_production: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Stateless drift detector.  All state lives in the returned
    :class:`DriftReport`; the instance can be reused across multiple calls.

    Usage
    -----
    >>> dd = DriftDetector()
    >>> report = dd.detect(reference_array, production_array, feature_names)
    >>> report.drift_detected
    False
    """

    PSI_BINS: int = 10
    MMD_SUBSAMPLE: int = 500  # subsample cap for O(n²) MMD kernel computation

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Run KS, PSI, and FDD tests between *reference* and *production*.

        Parameters
        ----------
        reference    : 2-D float array, shape ``(n_ref, n_features)``.
        production   : 2-D float array, shape ``(n_prod, n_features)``.
        feature_names: Optional list of feature names for labelling per-feature
                       results.  Auto-generated as ``feature_0 …`` when absent.

        Returns
        -------
        :class:`DriftReport`
        """
        reference  = np.asarray(reference,  dtype=np.float64)
        production = np.asarray(production, dtype=np.float64)

        # Force 2-D
        if reference.ndim == 1:
            reference  = reference.reshape(-1, 1)
        if production.ndim == 1:
            production = production.reshape(-1, 1)

        n_ref,  n_feat = reference.shape
        n_prod, _      = production.shape

        if feature_names is None or len(feature_names) != n_feat:
            feature_names = [f"feature_{i}" for i in range(n_feat)]

        # ── Per-feature KS and PSI ────────────────────────────────────────
        per_ks:  Dict[str, float] = {}
        per_psi: Dict[str, float] = {}

        for i, fname in enumerate(feature_names):
            ref_col  = reference[:, i]
            prod_col = production[:, i]
            per_ks[fname]  = self._compute_ks(ref_col, prod_col)
            per_psi[fname] = self._compute_psi(ref_col, prod_col)

        ks_statistic: float = (
            float(max(per_ks.values())) if per_ks else 0.0
        )
        psi: float = (
            float(np.mean(list(per_psi.values()))) if per_psi else 0.0
        )

        # ── Multivariate FDD (MMD with RBF kernel) ────────────────────────
        fdd: float = self._compute_mmd(reference, production)

        # ── Threshold checks (PDF spec) ───────────────────────────────────
        status: Dict[str, bool] = {
            "psi":          psi          > PSI_THRESHOLD,
            "ks_statistic": ks_statistic > KS_THRESHOLD,
            "fdd":          fdd          > FDD_THRESHOLD,
        }
        drift_detected: bool = any(status.values())

        logger.info(
            "DriftDetector: psi=%.4f (>%.2f? %s)  ks=%.4f (>%.2f? %s)  "
            "fdd=%.4f (>%.2f? %s)  drift=%s",
            psi,          PSI_THRESHOLD, status["psi"],
            ks_statistic, KS_THRESHOLD,  status["ks_statistic"],
            fdd,          FDD_THRESHOLD, status["fdd"],
            drift_detected,
        )

        return DriftReport(
            psi=psi,
            ks_statistic=ks_statistic,
            fdd=fdd,
            drift_detected=drift_detected,
            status=status,
            per_feature_ks=per_ks,
            per_feature_psi=per_psi,
            n_features=n_feat,
            n_reference=n_ref,
            n_production=n_prod,
        )

    # ------------------------------------------------------------------ #
    # KS test (per feature)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_ks(ref: np.ndarray, prod: np.ndarray) -> float:
        """
        Two-sample Kolmogorov-Smirnov statistic for one feature column.

        Uses ``scipy.stats.ks_2samp`` (exact two-sample variant).
        Returns the test *statistic* (0–1), not the p-value.
        """
        if len(ref) < 2 or len(prod) < 2:
            return 0.0
        result = stats.ks_2samp(ref, prod)
        return float(result.statistic)

    # ------------------------------------------------------------------ #
    # PSI (per feature)
    # ------------------------------------------------------------------ #

    def _compute_psi(
        self,
        ref: np.ndarray,
        prod: np.ndarray,
    ) -> float:
        """
        Population Stability Index for a single feature.

        Algorithm
        ---------
        1. Bin edges are derived from the combined min/max of both arrays so
           every observed value falls within a bin.
        2. Reference  → expected proportions (``ref_%``).
           Production → actual   proportions (``prod_%``).
        3. PSI formula::

               PSI = Σ (prod_% − ref_%) × ln(prod_% / ref_%)

        Epsilon 1e-8 is added to every bin count before normalising to avoid
        ``log(0)`` and division-by-zero on empty bins.

        PSI interpretation (conventional)
        -----------------------------------
        < 0.10  : negligible change
        0.10–0.25 : moderate change
        > 0.25  : significant drift (PDF threshold)
        """
        if len(ref) < 2 or len(prod) < 2:
            return 0.0

        combined_min = float(min(ref.min(), prod.min()))
        combined_max = float(max(ref.max(), prod.max()))

        if combined_min == combined_max:
            # Constant feature – no distributional information
            return 0.0

        bin_edges = np.linspace(combined_min, combined_max, self.PSI_BINS + 1)

        ref_counts,  _ = np.histogram(ref,  bins=bin_edges)
        prod_counts, _ = np.histogram(prod, bins=bin_edges)

        eps: float = 1e-8
        ref_pct  = (ref_counts  + eps) / (len(ref)  + eps * self.PSI_BINS)
        prod_pct = (prod_counts + eps) / (len(prod)  + eps * self.PSI_BINS)

        # PSI is mathematically non-negative; clamp any floating-point noise
        psi: float = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return max(psi, 0.0)

    # ------------------------------------------------------------------ #
    # FDD – Maximum Mean Discrepancy with RBF kernel
    # ------------------------------------------------------------------ #

    def _compute_mmd(
        self,
        ref: np.ndarray,
        prod: np.ndarray,
    ) -> float:
        """
        Multivariate Maximum Mean Discrepancy with a Gaussian (RBF) kernel.

        A random subsample of up to ``MMD_SUBSAMPLE`` rows is drawn from each
        split before computing the O(n²) kernel matrices.

        Bandwidth
        ---------
        Median heuristic: ``σ² = median(positive pairwise squared distances) / 2``.
        This is scale-invariant and avoids manual bandwidth selection.

        MMD² (unbiased estimate)
        ------------------------
        ::

            MMD²(X, Y) =  [Σᵢ≠ⱼ k(xᵢ,xⱼ)] / [n(n-1)]
                        + [Σᵢ≠ⱼ k(yᵢ,yⱼ)] / [m(m-1)]
                        − 2 × mean_ij k(xᵢ,yⱼ)

        Returns
        -------
        ``sqrt(max(MMD², 0))`` — always non-negative.
        """
        rng = np.random.default_rng(seed=42)

        n_ref_sub  = min(self.MMD_SUBSAMPLE, len(ref))
        n_prod_sub = min(self.MMD_SUBSAMPLE, len(prod))
        idx_r = rng.choice(len(ref),  n_ref_sub,  replace=False)
        idx_p = rng.choice(len(prod), n_prod_sub, replace=False)
        X: np.ndarray = ref[idx_r].astype(np.float64)
        Y: np.ndarray = prod[idx_p].astype(np.float64)

        # Median-heuristic bandwidth
        all_pts  = np.vstack([X, Y])
        sq_dists = np.sum(
            (all_pts[:, None, :] - all_pts[None, :, :]) ** 2,
            axis=-1,
        )
        positive = sq_dists[sq_dists > 0]
        sigma_sq: float = float(np.median(positive)) / 2.0 if positive.size > 0 else 1.0

        def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return np.exp(-d / (2.0 * sigma_sq))

        Kxx = _rbf(X, X)
        Kyy = _rbf(Y, Y)
        Kxy = _rbf(X, Y)

        n, m = len(X), len(Y)

        term_xx: float = (
            (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
            if n > 1 else 0.0
        )
        term_yy: float = (
            (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
            if m > 1 else 0.0
        )
        term_xy: float = 2.0 * float(np.mean(Kxy))

        mmd_sq: float = term_xx + term_yy - term_xy
        return float(np.sqrt(max(mmd_sq, 0.0)))
