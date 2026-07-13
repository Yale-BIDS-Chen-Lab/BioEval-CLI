# Single Label Classification (SLC) task implementation

from typing import Dict, List

class SLCTask:
    """Handler for Single Label Classification tasks like chemprot and ddi"""
    
    def postprocess(self, output: str, label_string: str = None) -> str:
        """
        Process model output for SLC tasks.
        
        Note: This method should not be used. Use postprocessing functions 
        specified in the YAML config instead (e.g., load_normalized_data).
        
        Args:
            output (str): The model output string
            label_string (str): Not used for SLC tasks
            
        Returns:
            str: The raw output (no processing)
        """
        return output if output else ""
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics for SLC tasks"""
        return ["macro_f1", "weighted_f1"]
    
    def add_example_metadata(self, record: Dict) -> Dict:
        """
        Add example-level metadata for SLC tasks.
        
        Args:
            record (Dict): The record containing prediction data
            
        Returns:
            Dict: The record with added metadata
        """
        # No additional metadata fields are added to records for SLC tasks
        return record 