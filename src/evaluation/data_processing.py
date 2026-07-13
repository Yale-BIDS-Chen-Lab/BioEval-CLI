"""
Data Processing Module for BioEval

This module contains functions for processing different types of biomedical NLP datasets.
Functions are organized into two categories:

1. File-based functions (for evaluation): process entire JSON files and return (golds, preds) tuples
2. Single-string functions (for postprocessing): process individual model outputs for postprocessed_output field

Supported tasks:
- NER: Named Entity Recognition (ncbi_disease, bc5cdr_chemical)
- SLC: Single Label Classification (chemprot, ddi)
- MCQ: Multiple Choice Questions (medqa, pubmedqa, custom datasets)
- MLC: Multi-Label Classification (hoc, litcovid, custom datasets)
- Generation: Text summarization or simplification (plos, cochrane, ms2, pubmed_summarization)

Unified Processing Approach:
- NER: Uses process_ner_custom/process_ner_token_indices with entity_types parameter (token-based)
        OR process_ner_char_offsets with entity_types parameter (character offset-based)
- MCQ: Uses process_mcq_custom with option_type parameter ("A-E" or "Yes/No/Maybe")
- MLC: Uses process_mlc_custom with label_string parameter
- SLC/Generation: Uses load_normalized_data (basic text normalization for consistent comparison)
"""

import json
import re
import os
from typing import List, Tuple, Union

# =============================================================================
# NER (Named Entity Recognition) Functions - TOKEN INDICES
# =============================================================================

def process_ner_token_indices_single(html: str, entity_types: str) -> Tuple[List[str], List[List[Union[int, str]]]]:
    """
    Converts an HTML-annotated sentence string with <span class="entity_type">...</span> 
    into token indices for entities. Uses INCLUSIVE end indexing.
    
    Args:
        html (str): Sentence with entities wrapped in <span class="entity_type">...</span>.
        entity_types (str): Comma-delimited string of entity types (e.g., "disease" or "disease, drug, gene").
        
    Returns:
        tokens (List[str]): tokenized sentence
        entities (List[List[int, int, str]]): List of [start, end_inclusive, label] for each entity
        
    Raises:
        TypeError: If entity_types is not a string.
    """
    # Validate input type
    if not isinstance(entity_types, str):
        raise TypeError(f"entity_types must be a string, got {type(entity_types).__name__}")
    
    # Parse comma-delimited string into list
    entity_types = [et.strip() for et in entity_types.split(",")]
    
    # Create a combined pattern for all entity types
    patterns = []
    for entity_type in entity_types:
        # Handle both single and double quotes: class="disease" or class='disease'
        pattern = rf'<\s*span\s+class\s*=\s*["\']{re.escape(entity_type)}["\']\s*>'
        patterns.append(pattern)
    
    # Replace all entity spans with special markers
    clean = html
    for i, pattern in enumerate(patterns):
        clean = re.sub(pattern, f' <START_{i}> ', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<\s*/\s*span\s*>', ' <END> ', clean, flags=re.IGNORECASE)

    tokens = []
    entities = []
    
    in_span = False
    current_entity_start = None
    current_entity_type = None

    for token in clean.strip().split():
        if token.startswith('<START_'):
            in_span = True
            current_entity_start = len(tokens)
            try:
                entity_index = int(token.replace('<START_', '').replace('>', ''))
                current_entity_type = entity_types[entity_index]
            except (ValueError, IndexError):
                # Handle malformed markers
                continue
        elif token == '<END>':
            if in_span and current_entity_start is not None and current_entity_type is not None:
                # Add entity [start, end_inclusive, label] - INCLUSIVE end indexing
                entities.append([current_entity_start, len(tokens) - 1, current_entity_type.title()])
            in_span = False
            current_entity_start = None
            current_entity_type = None
        else:
            tokens.append(token)

    return tokens, entities

def process_ner_token_indices(file_path: str, entity_types: str) -> Tuple[List[List[List[Union[int, str]]]], List[List[List[Union[int, str]]]]]:
    """
    Process JSON data for NER tasks by parsing HTML predictions and extracting entities (TOKEN INDICES).
    
    Args:
        file_path (str): Path to the JSON file containing HTML predictions and reference entities.
        entity_types (str): Comma-delimited string of entity types (e.g., "disease" or "disease, drug, gene").
        
    Returns:
        tuple: Two lists containing:
            - golds: List of entity lists, each entity as [start_token, end_token_inclusive, label]
            - preds: List of entity lists parsed from HTML predictions
            
    Raises:
        TypeError: If entity_types is not a string.
    """
    # Validate input type
    if not isinstance(entity_types, str):
        raise TypeError(f"entity_types must be a string, got {type(entity_types).__name__}")
        
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    golds = []
    preds = []
    
    for entry in data:
        # Get reference entities (already in [start, end_inclusive, label] format)
        gold_entities = entry['reference']
        golds.append(gold_entities)
        
        # Parse HTML prediction to extract entities
        html_pred = entry['output']
        tokens, pred_entities = process_ner_token_indices_single(html_pred, entity_types)
        preds.append(pred_entities)
    
    return golds, preds

# Backward compatibility: keep process_ner_custom as alias
def process_ner_custom_single(html: str, entity_types: str) -> Tuple[List[str], List[List[Union[int, str]]]]:
    """Legacy alias for process_ner_token_indices_single. Use process_ner_token_indices_single instead."""
    return process_ner_token_indices_single(html, entity_types)

def process_ner_custom(file_path: str, entity_types: str) -> Tuple[List[List[List[Union[int, str]]]], List[List[List[Union[int, str]]]]]:
    """Legacy alias for process_ner_token_indices. Use process_ner_token_indices instead."""
    return process_ner_token_indices(file_path, entity_types)

# =============================================================================
# NER (Named Entity Recognition) Functions - CHARACTER OFFSETS
# =============================================================================

def char_offsets_to_token_indices(text: str, char_start: int, char_end: int) -> Tuple[int, int]:
    """
    Convert character offsets to token indices using whitespace tokenization.
    
    Args:
        text (str): The original text
        char_start (int): Character start position (inclusive)
        char_end (int): Character end position (inclusive)
        
    Returns:
        tuple: (token_start, token_end) where both are inclusive token indices
    """
    tokens = text.split()
    token_start_idx = None
    token_end_idx = None
    
    char_pos = 0
    for token_idx, token in enumerate(tokens):
        token_start_char = char_pos
        token_end_char = char_pos + len(token) - 1
        
        # Check if this token overlaps with the target character range
        if token_start_idx is None and token_end_char >= char_start and token_start_char <= char_end:
            token_start_idx = token_idx
        
        if token_start_char <= char_end and token_end_char >= char_start:
            token_end_idx = token_idx
        
        # Move to next token (add token length + 1 for space)
        char_pos += len(token) + 1
    
    # If we didn't find tokens, return (0, 0) as fallback
    if token_start_idx is None:
        token_start_idx = 0
    if token_end_idx is None:
        token_end_idx = 0
    
    return token_start_idx, token_end_idx

def process_ner_char_offsets_single(html: str, entity_types: str, input: str) -> Tuple[List[str], List[List[Union[int, str]]]]:
    """
    Converts an HTML-annotated sentence string with <span class="entity_type">...</span> 
    into character offsets for entities. Uses INCLUSIVE end indexing.
    
    This function extracts entities from HTML and returns their character positions in the original text.
    
    Args:
        html (str): Sentence with entities wrapped in <span class="entity_type">...</span>.
        entity_types (str): Comma-delimited string of entity types (e.g., "disease" or "disease, drug, gene").
        input (str): The original input text (used for character offset calculation).
        
    Returns:
        tokens (List[str]): tokenized sentence
        entities (List[List[int, int, str]]): List of [start_char, end_char_inclusive, label] for each entity
        
    Raises:
        TypeError: If entity_types is not a string.
    """
    # Validate input type
    if not isinstance(entity_types, str):
        raise TypeError(f"entity_types must be a string, got {type(entity_types).__name__}")
    
    # Parse comma-delimited string into list
    entity_types_list = [et.strip() for et in entity_types.split(",")]
    
    # Create a combined pattern for all entity types
    patterns = []
    for entity_type in entity_types_list:
        pattern = rf'<\s*span\s+class\s*=\s*["\']{re.escape(entity_type)}["\']\s*>'
        patterns.append(pattern)
    
    # Extract entity texts and types from HTML
    entity_texts = []
    clean = html
    for i, pattern in enumerate(patterns):
        # Find all matches with this entity type
        for match in re.finditer(pattern + r'(.*?)<\s*/\s*span\s*>', html, flags=re.IGNORECASE):
            entity_text = match.group(1).strip()
            entity_texts.append((entity_text, entity_types_list[i]))
    
    # Find character offsets in the original text
    entities = []
    for entity_text, entity_type in entity_texts:
        # Find the entity in the input text
        start_pos = input.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text) - 1  # Inclusive end
            entities.append([start_pos, end_pos, entity_type.title()])
    
    # Tokenize for compatibility (though not used for offsets)
    tokens = input.split()
    
    return tokens, entities

def process_ner_char_offsets(file_path: str, entity_types: str) -> Tuple[List[List[List[Union[int, str]]]], List[List[List[Union[int, str]]]]]:
    """
    Process JSON data for NER tasks by parsing HTML predictions and extracting entities (CHARACTER OFFSETS).
    
    Args:
        file_path (str): Path to the JSON file containing HTML predictions and reference entities.
        entity_types (str): Comma-delimited string of entity types (e.g., "disease" or "disease, drug, gene").
        
    Returns:
        tuple: Two lists containing:
            - golds: List of entity lists, each entity as [start_char, end_char_inclusive, label]
            - preds: List of entity lists parsed from HTML predictions
            
    Raises:
        TypeError: If entity_types is not a string.
    """
    # Validate input type
    if not isinstance(entity_types, str):
        raise TypeError(f"entity_types must be a string, got {type(entity_types).__name__}")
        
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    golds = []
    preds = []
    
    for entry in data:
        # Get reference entities (already in [start_char, end_char_inclusive, label] format)
        gold_entities = entry['reference']
        golds.append(gold_entities)
        
        # Parse HTML prediction to extract entities with character offsets
        html_pred = entry['output']
        input = entry['input']
        tokens, pred_entities = process_ner_char_offsets_single(html_pred, entity_types, input)
        preds.append(pred_entities)
    
    return golds, preds

# =============================================================================
# Data Loading Functions (normalized)
# =============================================================================

def load_normalized_data(file_path: str) -> Tuple[List, List]:
    """
    Load JSON data with basic text normalization (lowercase + strip whitespace).
    Used for all tasks to ensure consistent text comparison.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple: Two lists containing normalized gold (reference) values and predicted values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
        # Handle both list (inference files) and dict with 'records' key (eval files)
        if isinstance(data, list):
            records = data
        else:
            records = data.get('records', data)
        golds = [entry['reference'].lower().strip() for entry in records]
        preds = [entry['postprocessed_output'].lower().strip() for entry in records]
    return golds, preds

# =============================================================================
# MCQ (Multiple Choice Questions) Functions
# =============================================================================

def process_mcq_custom_single(output: str, option_type: str) -> str:
    """
    Unified MCQ processor that handles different option types.
    
    Args:
        output (str): The model output string.
        option_type (str): Type of options - "A-E" for a,b,c,d,e, "A-D" for a,b,c,d, or "Yes/No/Maybe" for yes,no,maybe.
        
    Returns:
        str: The matched option or "missing" if no valid option found.
    """
    if not output or not output.strip():
        return "missing"
    
    output = output.lower().strip()
    
    if option_type == "A-E":
        # Look for first character a, b, c, d, e
        return output[0] if output[0] in {'a', 'b', 'c', 'd', 'e'} else "missing"
    elif option_type == "A-D":
        # Look for first character a, b, c, d
        return output[0] if output[0] in {'a', 'b', 'c', 'd'} else "missing"
    elif option_type == "Yes/No/Maybe":
        # Look for substring yes, no, maybe anywhere in text
        for label in ['yes', 'no', 'maybe']:
            if label in output:
                return label
        return "missing"
    else:
        raise ValueError(f"Unsupported option_type: {option_type}. Use 'A-E', 'A-D', or 'Yes/No/Maybe'")

def process_mcq_custom(file_path: str, option_type: str) -> Tuple[List[str], List[str]]:
    """
    Process JSON data for custom MCQ datasets with specified option type.

    Args:
        file_path (str): Path to the JSON file.
        option_type (str): Type of options - "A-E" for a,b,c,d,e, "A-D" for a,b,c,d, or "Yes/No/Maybe" for yes,no,maybe.

    Returns:
        tuple: Two lists containing processed gold (reference) values and predicted values.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    golds = [process_mcq_custom_single(entry['reference'], option_type) for entry in data]
    preds = [process_mcq_custom_single(entry['output'], option_type) for entry in data]
    return golds, preds





# =============================================================================
# Classification (Multi-Label Classification) Functions
# =============================================================================

def process_mlc_custom(file_path: str, label_string: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generalized classification parser for custom label sets. Accepts a comma-separated label string.
    Args:
        file_path (str): Path to the JSON file.
        label_string (str): Comma-separated string of labels.
    Returns:
        tuple: Two lists of binary label vectors for golds and preds.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    golds = [process_mlc_custom_single(entry['reference'], label_string) for entry in data]
    preds = [process_mlc_custom_single(entry['output'], label_string) for entry in data]
    return golds, preds

def process_mlc_custom_single(output: str, label_string: str) -> List[int]:
    """
    Process a single output string for custom classification using the same logic as process_mlc_custom.
    Converts text to binary vector based on comma-separated label list.
    
    Args:
        output (str): The model output string.
        label_string (str): Comma-separated string of labels.
        
    Returns:
        list: Binary vector representing which labels are present in the output.
    """
    label_list = [label.strip().lower() for label in label_string.split(",")]
    output = output.lower()
    result = [0] * len(label_list)
    for index, choice in enumerate(label_list):
        if choice in output:
            result[index] = 1
    return result