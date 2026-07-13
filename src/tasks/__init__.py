# Task-specific implementations for biomedical NLP evaluation

def get_task_handler(task_type: str):
    """Factory function to get appropriate task handler"""
    if task_type == "mcq":
        from .mcq import MCQTask
        return MCQTask()
    elif task_type == "ner":
        from .ner import NERTask
        return NERTask()
    elif task_type == "mlc":
        from .mlc import MLCTask
        return MLCTask()
    elif task_type == "slc":
        from .slc import SLCTask
        return SLCTask()
    elif task_type == "generation":
        from .generation import GenerationTask
        return GenerationTask()
    else:
        raise ValueError(f"Unknown task type: {task_type}") 