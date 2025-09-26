#!/usr/bin/env python3
import os
import sys
import tempfile
import json
import click
import subprocess

# Get the absolute path of the MusiteDeep directory
MUSITEDEEP_DIR = os.path.dirname(os.path.abspath(__file__))

# Available PTM models
AVAILABLE_MODELS = [
    'N6-acetyllysine',
    'O-linked_glycosylation', 
    'S-palmitoyl_cysteine',
    'Hydroxyproline',
    'Pyrrolidone_carboxylic_acid',
    'N-linked_glycosylation',
    'Ubiquitination',
    'SUMOylation',
    'Phosphotyrosine',
    'Methyllysine',
    'Methylarginine',
    'Hydroxylysine',
    'Phosphoserine_Phosphothreonine'
]

# Model aliases for easier input
MODEL_ALIASES = {
    # Numbers
    '1': 'N6-acetyllysine', '2': 'O-linked_glycosylation', '3': 'S-palmitoyl_cysteine',
    '4': 'Hydroxyproline', '5': 'Pyrrolidone_carboxylic_acid', '6': 'N-linked_glycosylation',
    '7': 'Ubiquitination', '8': 'SUMOylation', '9': 'Phosphotyrosine',
    '10': 'Methyllysine', '11': 'Methylarginine', '12': 'Hydroxylysine', '13': 'Phosphoserine_Phosphothreonine',
    
    # Short names
    'acetyl': 'N6-acetyllysine', 'o-glyc': 'O-linked_glycosylation', 'palmitoyl': 'S-palmitoyl_cysteine',
    'hydroxyp': 'Hydroxyproline', 'pyrrol': 'Pyrrolidone_carboxylic_acid', 'n-glyc': 'N-linked_glycosylation',
    'ub': 'Ubiquitination', 'sumo': 'SUMOylation', 'py': 'Phosphotyrosine',
    'methyl': 'Methyllysine', 'methylr': 'Methylarginine', 'hydroxyl': 'Hydroxylysine', 'pst': 'Phosphoserine_Phosphothreonine',
    
    # Phosphorylation shortcuts
    'phos': 'Phosphotyrosine,Phosphoserine_Phosphothreonine'
}

def check_model_data():
    """Check if model data is available"""
    models_dir = os.path.join(MUSITEDEEP_DIR, 'models')
    if not os.path.exists(models_dir):
        click.echo("❌ Error: Model data not found!", err=True)
        click.echo("\n📥 Please download model data from the original repository:", err=True)
        click.echo("   https://github.com/duolinwang/MusiteDeep_web/tree/master/MusiteDeep/models", err=True)
        click.echo("\n📁 Place all model folders in: {}".format(models_dir), err=True)
        click.echo("\n💡 See INSTALL.md for detailed instructions.", err=True)
        sys.exit(1)
    
    # Check if at least one model exists
    model_found = False
    for model_name in AVAILABLE_MODELS:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            model_found = True
            break
    
    if not model_found:
        click.echo("❌ Error: No valid model data found in models/ directory!", err=True)
        click.echo("\n📥 Please download model data from:", err=True)
        click.echo("   https://github.com/duolinwang/MusiteDeep_web/tree/master/MusiteDeep/models", err=True)
        click.echo("\n📁 Expected models directory: {}".format(models_dir), err=True)
        sys.exit(1)

def resolve_models(models_input):
    """Resolve model input to actual model names"""
    if models_input.lower() == 'all':
        return AVAILABLE_MODELS, None
    
    model_list = []
    for item in models_input.split(','):
        item = item.strip()
        if item in MODEL_ALIASES:
            # Handle multi-model aliases
            if ',' in MODEL_ALIASES[item]:
                model_list.extend(MODEL_ALIASES[item].split(','))
            else:
                model_list.append(MODEL_ALIASES[item])
        elif item in AVAILABLE_MODELS:
            model_list.append(item)
        else:
            return None, item  # Return error
    
    return list(set(model_list)), None  # Remove duplicates

def parse_results(results_file):
    """Parse MusiteDeep results file and return structured data"""
    results = {}
    current_protein = None
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Position'):
            continue
            
        if line.startswith('>'):
            current_protein = line[1:]  # Remove '>'
            results[current_protein] = []
        else:
            parts = line.split('\t')
            if len(parts) >= 4:
                position = int(parts[0])
                residue = parts[1]
                ptm_scores = parts[2]
                cutoff_result = parts[3]
                
                # Parse PTM scores
                ptm_data = {}
                for score_item in ptm_scores.split(';'):
                    if ':' in score_item:
                        ptm_type, score = score_item.split(':', 1)
                        ptm_data[ptm_type] = float(score)
                
                # Parse cutoff results
                significant_ptms = []
                if cutoff_result != 'None':
                    for sig_item in cutoff_result.split(';'):
                        if ':' in sig_item:
                            ptm_type, score = sig_item.split(':', 1)
                            significant_ptms.append({
                                'ptm_type': ptm_type,
                                'score': float(score)
                            })
                
                results[current_protein].append({
                    'position': position,
                    'residue': residue,
                    'ptm_scores': ptm_data,
                    'significant_ptms': significant_ptms
                })
    
    return results

def run_single_model_prediction(temp_fasta_path, model_name, temp_dir):
    """Run prediction for a single model"""
    temp_output_prefix = os.path.join(temp_dir, 'musitedeep_output_{}'.format(model_name))
    
    # Run MusiteDeep prediction for single model
    cmd = [
        sys.executable, 
        os.path.join(MUSITEDEEP_DIR, 'predict_multi_batch.py'),
        '-input', temp_fasta_path,
        '-output', temp_output_prefix,
        '-model-prefix', "models/{}".format(model_name)
    ]
    
    # Change to MusiteDeep directory to ensure relative paths work
    result = subprocess.Popen(cmd, cwd=MUSITEDEEP_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()
    
    if result.returncode != 0:
        click.echo("Error running prediction for model {}: {}".format(model_name, stderr.decode()), err=True)
        return None
    
    # Parse results
    results_file = temp_output_prefix + '_results.txt'
    if not os.path.exists(results_file):
        click.echo("Error: No results file generated for model {}".format(model_name), err=True)
        return None
    
    return parse_results(results_file)

def merge_results(all_results, protein_name, cutoff):
    """Merge results from multiple models"""
    predictions = []
    
    for model_results in all_results:
        if protein_name in model_results:
            for prediction in model_results[protein_name]:
                position = prediction['position']
                residue = prediction['residue']
                
                # Add each PTM prediction as separate entry
                for ptm_type, score in prediction['ptm_scores'].items():
                    predictions.append({
                        'position': position,
                        'residue': residue,
                        'ptm_type': ptm_type,
                        'score': score,
                        'risk': score > cutoff
                    })
    
    # Sort by position, then by ptm_type
    return sorted(predictions, key=lambda x: (x['position'], x['ptm_type']))

@click.command()
@click.option('--sequence', '-s', help='Protein sequence to predict')
@click.option('--models', '-m', default='all', help='PTM models to use (numbers, short names, or "all")')
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--cutoff', '-c', default=0.5, type=float, help='Score cutoff for significant predictions')
@click.option('--list', 'show_list', is_flag=True, help='Show available models and exit')
def predict(sequence, models, output, cutoff, show_list):
    """MusiteDeep PTM prediction tool"""
    
    # Check model data availability first
    check_model_data()
    
    if show_list:
        click.echo("Available PTM Models:")
        click.echo("=" * 120)
        click.echo("{:<4} {:<15} {:<30} {:<50}".format("No.", "Short Name", "Full Name", "Biological Description"))
        click.echo("-" * 120)
        
        short_names = ['acetyl', 'o-glyc', 'palmitoyl', 'hydroxyp', 'pyrrol', 'n-glyc', 'ub', 'sumo', 'py', 'methyl', 'methylr', 'hydroxyl', 'pst']
        descriptions = [
            'Gene expression regulation, chromatin remodeling',
            'Protein folding, cell adhesion, immune response',
            'Membrane association, protein trafficking',
            'Collagen stability, extracellular matrix formation',
            'Protein degradation signal, N-terminal processing',
            'Protein folding, quality control, cell recognition',
            'Protein degradation, cell cycle, DNA repair',
            'Nuclear transport, transcriptional regulation',
            'Signal transduction, cell growth, differentiation',
            'Gene expression, chromatin structure regulation',
            'Gene expression, RNA processing, DNA repair',
            'Collagen cross-linking, connective tissue stability',
            'Signal transduction, metabolic regulation'
        ]
        
        for i, model in enumerate(AVAILABLE_MODELS, 1):
            short_name = short_names[i-1] if i <= len(short_names) else ""
            description = descriptions[i-1] if i <= len(descriptions) else ""
            click.echo("{:<4} {:<15} {:<30} {:<50}".format(i, short_name, model, description))
        
        click.echo("-" * 120)
        click.echo("\nSpecial options:")
        click.echo("  phos    - Phosphorylation (py + pst)")
        click.echo("  all     - All models")
        click.echo("\nUsage examples:")
        click.echo("  --models 1,2,3      (use numbers)")
        click.echo("  --models py,methyl  (use short names)")
        click.echo("  --models phos       (phosphorylation)")
        return
    
    if not sequence:
        click.echo("Error: --sequence is required", err=True)
        sys.exit(1)
    
    # Validate sequence
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(aa.upper() in valid_amino_acids for aa in sequence):
        click.echo("Error: Invalid amino acid characters in sequence", err=True)
        sys.exit(1)
    
    # Determine models to use
    model_list, error_item = resolve_models(models)
    if model_list is None:
        click.echo("Error: Invalid model name or number: '{}'".format(error_item), err=True)
        click.echo("\nAvailable options:", err=True)
        click.echo("Numbers: 1-13", err=True)
        click.echo("Short names: acetyl, o-glyc, palmitoyl, hydroxyp, pyrrol, n-glyc, ub, sumo, py, methyl, methylr, hydroxyl, pst", err=True)
        click.echo("Special: phos (phosphorylation), all", err=True)
        click.echo("Full names: {}".format(', '.join(AVAILABLE_MODELS)), err=True)
        sys.exit(1)
    
    # Create temporary directory and files
    temp_dir = tempfile.mkdtemp()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_fasta:
        temp_fasta.write(">input_protein\n{}\n".format(sequence.upper()))
        temp_fasta_path = temp_fasta.name
    
    try:
        all_results = []
        
        # Run prediction for each model individually
        for model_name in model_list:
            click.echo("Running prediction for model: {}".format(model_name))
            model_results = run_single_model_prediction(temp_fasta_path, model_name, temp_dir)
            if model_results:
                all_results.append(model_results)
        
        if not all_results:
            click.echo("Error: No successful predictions", err=True)
            sys.exit(1)
        
        # Merge results from all models
        protein_name = "input_protein"
        merged_predictions = merge_results(all_results, protein_name, cutoff)
        
        # Format output
        output_data = {
            'sequence': sequence.upper(),
            'models_used': model_list,
            'cutoff': cutoff,
            'predictions': merged_predictions
        }
        
        # Output results
        # Always show table in console
        click.echo("\nPrediction Results:")
        click.echo("-" * 60)
        click.echo("{:<8} {:<8} {:<25} {:<8} {:<6}".format("Position", "Residue", "PTM Type", "Score", "Risk"))
        click.echo("-" * 60)
        
        for pred in merged_predictions:
            risk_str = "HIGH" if pred['risk'] else "LOW"
            click.echo("{:<8} {:<8} {:<25} {:<8.3f} {:<6}".format(
                pred['position'], pred['residue'], pred['ptm_type'], pred['score'], risk_str
            ))
        
        click.echo("-" * 60)
        click.echo("Total predictions: {}".format(len(merged_predictions)))
        high_risk_count = sum(1 for p in merged_predictions if p['risk'])
        click.echo("High risk predictions: {}".format(high_risk_count))
        
        if output:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo("\nResults also saved to {}".format(output))
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_fasta_path):
            os.unlink(temp_fasta_path)
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    predict()
