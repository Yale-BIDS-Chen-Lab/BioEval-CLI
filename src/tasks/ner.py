# Named Entity Recognition (NER) task implementation

import re
from typing import Dict, List, Tuple, Union

class NERTask:
    """Handler for Named Entity Recognition tasks"""
    
    def postprocess(self, output: str, label_string: str = "disease") -> str:
        """
        Process model output for NER tasks.
        
        Note: This method should not be used. Use postprocessing functions 
        specified in the YAML config instead (e.g., process_ner_custom).
        
        Args:
            output (str): The model output string
            label_string (str): Not used
            
        Returns:
            str: The raw output (no processing)
        """
        return output if output else ""
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics for NER tasks"""
        return ["exact_match_precision", "exact_match_recall", "exact_match_f1"]
    
    def add_example_metadata(self, record: Dict) -> Dict:
        """Add example-level metadata for NER tasks"""
        return record 