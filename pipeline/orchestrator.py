"""Pipeline orchestrator for coordinating ML workflows."""

from typing import Dict, List, Optional, Callable


class Orchestrator:
    """Orchestrates the entire ML pipeline."""
    
    def __init__(self):
        self.tasks = {}
        self.results = {}
    
    def register_task(self, name: str, task_func: Callable):
        """Register a pipeline task."""
        self.tasks[name] = task_func
    
    def run_task(self, name: str, **kwargs):
        """Run a registered task."""
        if name not in self.tasks:
            return None
        
        result = self.tasks[name](**kwargs)
        self.results[name] = result
        return result
    
    def run_pipeline(self, task_order: List[str], **kwargs):
        """Run pipeline in specified task order."""
        results = {}
        for task_name in task_order:
            result = self.run_task(task_name, **kwargs)
            results[task_name] = result
        return results
    
    def get_result(self, task_name: str) -> Optional:
        """Get result from a completed task."""
        return self.results.get(task_name)
