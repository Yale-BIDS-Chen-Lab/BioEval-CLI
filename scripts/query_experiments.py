#!/usr/bin/env python3
"""
Query Experiments Script for BioEval

This script allows users to search for experiment configurations and results
by task, dataset, and/or model.

Usage:
    python3 scripts/query_experiments.py --task mcq
    python3 scripts/query_experiments.py --dataset medqa
    python3 scripts/query_experiments.py --model meta-llama/Llama-3.2-1B-Instruct
    python3 scripts/query_experiments.py --task mcq --dataset medqa --model meta-llama/Llama-3.2-1B-Instruct
    python3 scripts/query_experiments.py --task mcq --dataset medqa --model meta-llama/Llama-3.2-1B-Instruct --config-dir config/model_cards --outputs-dir data/outputs
    
    python3 scripts/query_experiments.py --list-available
    python3 scripts/query_experiments.py --list-available --config-dir config/model_cards --outputs-dir data/outputs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ExperimentQueryer:
    def __init__(self, model_cards_dir: str = "config/model_cards", outputs_dir: str = "data/outputs"):
        # Handle both relative and absolute paths
        if Path(model_cards_dir).is_absolute():
            self.model_cards_dir = Path(model_cards_dir)
        else:
            # Get project root directory (parent of scripts directory)
            project_root = Path(__file__).parent.parent
            self.model_cards_dir = project_root / model_cards_dir
            
        if Path(outputs_dir).is_absolute():
            self.outputs_dir = Path(outputs_dir)
        else:
            project_root = Path(__file__).parent.parent
            self.outputs_dir = project_root / outputs_dir
        
    def search_configs(self, dataset: Optional[str] = None, 
                      model: Optional[str] = None, 
                      task: Optional[str] = None) -> List[Dict]:
        """Search for YAML configurations matching criteria."""
        configs = []
        
        if not self.model_cards_dir.exists():
            print(f"Warning: Config directory does not exist: {self.model_cards_dir}")
            return configs
        
        # Find all YAML files
        yaml_files = list(self.model_cards_dir.rglob("*.yaml"))
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    content = f.read()
                    
                # Parse basic info from YAML content
                config_info = self._parse_yaml_content(content, yaml_file)
                
                # Apply filters
                if dataset and dataset.lower() != config_info.get('dataset', '').lower():
                    continue
                if model and model.lower() not in config_info.get('model_name', '').lower():
                    continue
                if task and task.lower() != config_info.get('task', '').lower():
                    continue
                    
                configs.append(config_info)
                
            except Exception as e:
                print(f"Warning: Could not parse {yaml_file}: {e}")
                
        return configs
    
    def _parse_yaml_content(self, content: str, file_path: Path) -> Dict:
        """Parse YAML content to extract key information."""
        lines = content.split('\n')
        config_info = {
            'file_path': str(file_path),
            'task': None,
            'dataset': None,
            'model_name': None,
            'provider': None,
            'temperature': None,
            'max_new_tokens': None
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('task:'):
                value = line.split(':', 1)[1].strip()
                # Remove comment if present
                if '#' in value:
                    value = value.split('#')[0].strip()
                config_info['task'] = value
            elif line.startswith('dataset:'):
                value = line.split(':', 1)[1].strip()
                # Remove comment if present
                if '#' in value:
                    value = value.split('#')[0].strip()
                config_info['dataset'] = value
            elif line.startswith('name:'):
                value = line.split(':', 1)[1].strip().strip('"')
                # Remove comment if present
                if '#' in value:
                    value = value.split('#')[0].strip().strip('"')
                config_info['model_name'] = value
            elif line.startswith('provider:'):
                value = line.split(':', 1)[1].strip()
                # Remove comment if present
                if '#' in value:
                    value = value.split('#')[0].strip()
                config_info['provider'] = value
            elif line.startswith('temperature:'):
                value = line.split(':', 1)[1].strip()
                # Remove comment if present
                if '#' in value:
                    value = value.split('#')[0].strip()
                config_info['temperature'] = value
            elif line.startswith('max_new_tokens:'):
                value = line.split(':', 1)[1].strip()
                # Remove comment if present
                if '#' in value:
                    value = value.split('#')[0].strip()
                config_info['max_new_tokens'] = value
                
        return config_info
    
    def search_results(self, dataset: Optional[str] = None, 
                      model: Optional[str] = None,
                      task: Optional[str] = None) -> List[Dict]:
        """Search for evaluation results matching criteria."""
        results = []
        
        if not self.outputs_dir.exists():
            print(f"Warning: Results directory does not exist: {self.outputs_dir}")
            return results
            
        # Find all evaluation files (pattern: {dataset}_{hash}.json)
        eval_files = list(self.outputs_dir.glob("*.json"))
        
        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                
                # Extract dataset name from filename (everything before the hash)
                # Pattern: {dataset}_{hash}.json where hash is 8 hex characters
                # For "bc5cdr_chemical_char_offsets_62d814ce.json" -> "bc5cdr_chemical_char_offsets"
                filename = eval_file.stem
                parts = filename.split('_')
                # Check if last part is 8-character hex (the hash)
                if len(parts) > 1 and len(parts[-1]) == 8 and all(c in '0123456789abcdef' for c in parts[-1]):
                    dataset_name = '_'.join(parts[:-1])
                else:
                    # If doesn't match pattern, use the whole filename
                    dataset_name = filename
                
                # Apply filters
                if dataset and dataset.lower() != dataset_name.lower():
                    continue
                    
                # Extract model info from metadata
                model_name = data.get('inference_metadata', {}).get('model_name', 'Unknown')
                if model and model.lower() not in model_name.lower():
                    continue
                    
                # Extract task info
                task_name = data.get('dataset_info', {}).get('task', 'Unknown')
                if task and task.lower() != task_name.lower():
                    continue
                
                result_info = {
                    'file_path': str(eval_file),
                    'dataset': dataset_name,
                    'task': task_name,
                    'model_name': model_name,
                    'provider': data.get('inference_metadata', {}).get('provider', 'Unknown'),
                    'timestamp': data.get('evaluation_metadata', {}).get('evaluation_timestamp', 'Unknown'),
                    'metrics': data.get('evaluation_metadata', {}).get('metrics', {})
                }
                
                results.append(result_info)
                
            except Exception as e:
                print(f"Warning: Could not parse {eval_file}: {e}")
                
        return results

    def list_available(self) -> None:
        """List all available datasets, tasks, and models."""
        print("📋 Available Experiments")
        print("=" * 60)
        print(f"📁 Config directory: {self.model_cards_dir}")
        print(f"📁 Results directory: {self.outputs_dir}")
        print()
        
        # Get all configurations
        configs = self.search_configs()
        
        # Extract unique values
        datasets = set()
        tasks = set()
        models = set()
        
        for config in configs:
            if config['dataset']:
                datasets.add(config['dataset'])
            if config['task']:
                tasks.add(config['task'])
            if config['model_name']:
                models.add(config['model_name'])
        
        print(f"📊 Datasets ({len(datasets)}):")
        for dataset in sorted(datasets):
            print(f"   • {dataset}")
            
        print(f"\n🎯 Tasks ({len(tasks)}):")
        for task in sorted(tasks):
            print(f"   • {task}")
            
        print(f"\n🤖 Models ({len(models)}):")
        for model in sorted(models):
            print(f"   • {model}")


def main():
    parser = argparse.ArgumentParser(
        description="Query BioEval experiments by task, dataset, and/or model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_experiments.py --dataset MedQA
  python query_experiments.py --model gpt-4o
  python query_experiments.py --task NER
  python query_experiments.py --dataset MedQA --model gpt-4o
  python query_experiments.py --task mlc --dataset hoc
  python query_experiments.py --list-available
  python query_experiments.py --list-available --outputs-dir /path/to/results
  python query_experiments.py --dataset hoc --config-dir /path/to/configs --outputs-dir /path/to/results
        """
    )
    
    # Search criteria
    parser.add_argument('--dataset', help='Filter by dataset name')
    parser.add_argument('--model', help='Filter by model name')  
    parser.add_argument('--task', help='Filter by task type')
    
    # Directory options
    parser.add_argument('--config-dir', default='config/model_cards',
                       help='Directory containing model card YAML files (default: config/model_cards)')
    parser.add_argument('--outputs-dir', default='data/outputs',
                       help='Directory containing result JSON files (default: data/outputs)')
    
    # List available options
    parser.add_argument('--list-available', action='store_true',
                       help='List all available datasets, tasks, and models')
    
    args = parser.parse_args()
    
    # Initialize queryer with custom directories
    queryer = ExperimentQueryer(
        model_cards_dir=args.config_dir,
        outputs_dir=args.outputs_dir
    )
    
    # Handle list available action
    if args.list_available:
        queryer.list_available()
        return
    
    # Require at least one filter for search
    if not any([args.dataset, args.model, args.task]):
        print("❌ Please specify at least one filter: --dataset, --model, or --task")
        print("💡 Or use --list-available to see all available options")
        parser.print_help()
        return
    
    # Build search description
    filters = []
    if args.task:
        filters.append(f"task='{args.task}'")
    if args.dataset:
        filters.append(f"dataset='{args.dataset}'")
    if args.model:
        filters.append(f"model='{args.model}'")
    
    print(f"🔍 Searching for experiments with {', '.join(filters)}")
    print("=" * 60)
    print(f"📁 Config directory: {queryer.model_cards_dir}")
    print(f"📁 Results directory: {queryer.outputs_dir}")
    print()
    
    # Search configurations
    configs = queryer.search_configs(
        dataset=args.dataset,
        model=args.model,
        task=args.task
    )
    
    # Search results
    results = queryer.search_results(
        dataset=args.dataset,
        model=args.model,
        task=args.task
    )
    
    # Display configurations
    if configs:
        print(f"📋 Found {len(configs)} Configuration(s):")
        print("-" * 40)
        for config in configs:
            print(f"📁 {config['file_path']}")
            print(f"   Task: {config['task'] or 'N/A'}")
            print(f"   Dataset: {config['dataset'] or 'N/A'}")
            print(f"   Model: {config['model_name'] or 'N/A'}")
            print(f"   Provider: {config['provider'] or 'N/A'}")
            print()
    
    # Display results
    if results:
        print(f"📊 Found {len(results)} Result(s):")
        print("-" * 40)
        for result in results:
            print(f"📁 {result['file_path']}")
            print(f"   Task: {result['task']}")
            print(f"   Dataset: {result['dataset']}")
            print(f"   Model: {result['model_name']}")
            print(f"   Provider: {result['provider']}")
            print(f"   Timestamp: {result['timestamp']}")
            
            # Show metrics
            metrics = result['metrics']
            if metrics:
                print("   📈 Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"      {metric}: {value:.4f}")
                    else:
                        print(f"      {metric}: {value}")
            print()
    
    # Summary
    if not configs and not results:
        print("❌ No experiments found matching the criteria.")
        print("💡 Try different filter values or use --list-available to see all options.")
    else:
        print(f"✅ Summary: {len(configs)} configuration(s), {len(results)} result file(s).")


if __name__ == "__main__":
    main() 