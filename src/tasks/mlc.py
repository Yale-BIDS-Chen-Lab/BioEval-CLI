# Multi-Label Classification (MLC) task implementation

from typing import Dict, List

class MLCTask:
    """Handler for Multi-Label Classification tasks"""
    
    def postprocess(self, output: str, label_string: str) -> str:
        """
        Process model output for MLC tasks.
        
        Note: This method should not be used. Use postprocessing functions 
        specified in the YAML config instead (e.g., process_mlc_custom).
        
        Args:
            output (str): The model output string
            label_string (str): Not used
            
        Returns:
            str: The raw output (no processing)
        """
        return output if output else ""
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics for MLC tasks"""
        return ["accuracy", "macro_f1", "weighted_f1"]
    
    def add_example_metadata(self, record: Dict) -> Dict:
        """Add example-level metadata for MLC tasks"""
        # No additional metadata fields are added to records for MLC tasks
        return record 