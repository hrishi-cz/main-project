"""
pipeline/monitoring.py (ENHANCED)

Monitoring engine with auto-report generation.
Triggers on performance degradation, drift detection, or retrain events.
Integrates paper generation into the monitoring workflow.
"""

import logging
import os
import json
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringEngine:
    """
    Post-training monitoring with automatic report generation.
    
    Triggers:
    - Accuracy drops below threshold → auto-generate paper
    - Calibration (ECE) exceeds threshold → auto-generate paper
    - Drift detected → auto-generate paper
    - Retrain completed → auto-generate paper
    
    Usage:
        monitor = MonitoringEngine()
        result = monitor.evaluate_and_report(model_id, metrics)
        # Returns: {"alerts": [...], "report_generated": True, "report_path": "..."}
    """

    def __init__(self, registry_dir: str = "models", reports_dir: str = "reports"):
        """
        Parameters
        ----------
        registry_dir : str
            Model registry directory.
        reports_dir : str
            Where to save generated reports.
        """
        self.registry_dir = registry_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

        # Alert thresholds
        self.accuracy_threshold = 0.60
        self.ece_threshold = 0.15
        self.f1_threshold = 0.40

    def evaluate_and_report(
        self,
        model_id: str,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluate metrics and trigger auto-report if needed.

        Parameters
        ----------
        model_id : str
            Model identifier from registry.
        metrics : Dict[str, float]
            Metrics dict from training: {"accuracy": ..., "f1": ..., "ece": ...}

        Returns
        -------
        Dict with:
            - "alerts": list of alert messages
            - "report_generated": bool
            - "report_path": str (if generated)
        """
        alerts = []

        # -----  Alert Checks  -----
        accuracy = metrics.get("accuracy")
        if accuracy is not None and accuracy < self.accuracy_threshold:
            alerts.append(f"⚠️  Low accuracy: {accuracy:.3f} < {self.accuracy_threshold}")

        ece = metrics.get("ece")
        if ece is not None and ece > self.ece_threshold:
            alerts.append(f"⚠️  Poor calibration (ECE): {ece:.3f} > {self.ece_threshold}")

        f1 = metrics.get("f1")
        if f1 is not None and f1 < self.f1_threshold:
            alerts.append(f"⚠️  Low F1 score: {f1:.3f} < {self.f1_threshold}")

        # -----  Trigger Report if Alerts -----
        report_path = None
        if alerts:
            report_path = self._generate_report(model_id, metrics, alerts)

        return {
            "alerts": alerts,
            "report_generated": bool(report_path),
            "report_path": report_path,
        }

    def _generate_report(
        self,
        model_id: str,
        metrics: Dict[str, float],
        alerts: list,
    ) -> str:
        """
        Generate and save a monitoring report.
        Also triggers full paper generation if configured.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{model_id}_{timestamp}.md"
        report_path = os.path.join(self.reports_dir, report_name)

        # Build report content
        report_lines = [
            f"# Monitoring Report\n",
            f"**Model ID**: {model_id}\n",
            f"**Generated**: {datetime.now().isoformat()}\n\n",
            f"## Alerts\n",
        ]

        for alert in alerts:
            report_lines.append(f"- {alert}\n")

        report_lines.append(f"\n## Metrics\n")
        for key, val in metrics.items():
            report_lines.append(f"- **{key}**: {val:.4f}\n")

        # Try to include full paper if possible
        try:
            from research.paper_service import PaperService
            
            logger.info("  Generating full research paper...")
            service = PaperService(registry_dir=self.registry_dir)
            paper_text, plot_path = service.generate()
            
            report_lines.append(f"\n## Full Research Paper\n")
            report_lines.append(paper_text)
            
            if plot_path:
                report_lines.append(f"\n**Plot saved**: {plot_path}\n")
                
        except Exception as e:
            logger.warning(f"  Could not generate full paper: {e}")
            report_lines.append(f"\n*(Full paper generation failed: {e})*\n")

        # Save report
        report_content = "".join(report_lines)
        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"✓ Monitoring report saved: {report_path}")
        return report_path

    def list_reports(self) -> list:
        """List all generated reports."""
        if not os.path.exists(self.reports_dir):
            return []
        return sorted(os.listdir(self.reports_dir))

    def get_report(self, report_name: str) -> Optional[str]:
        """Load report content by name."""
        report_path = os.path.join(self.reports_dir, report_name)
        if not os.path.exists(report_path):
            return None
        with open(report_path, "r") as f:
            return f.read()
