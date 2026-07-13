# Text Generation task implementation (summarization, simplification)

from typing import Dict, List

class GenerationTask:
    """Handler for text generation tasks like summarization and simplification"""
    
    def postprocess(self, output: str, label_string: str = None) -> str:
        """
        Process model output for generation tasks.
        
        Note: This method should not be used. Use postprocessing functions 
        specified in the YAML config instead (e.g., load_normalized_data).
        
        Args:
            output (str): The model output string
            label_string (str): Not used for generation tasks
            
        Returns:
            str: The raw output (no processing)
        """
        return output if output else ""
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics for generation tasks"""
        return ["rouge1", "rouge2", "rougeL", "bertscore", "bartscore", "meteor"]
    
    def add_example_metadata(self, record: Dict) -> Dict:
        """Add example-level metadata for generation tasks"""
        # No additional metadata fields are added to records for generation tasks
        return record 