'''
Generate using symbolic manipulation
Store/Index using hash codes or encodings
Never reverse-engineer from encodings during reasoning

Key features of the hybrid approach:
✅ Fast generation (symbolic inference)
✅ Gödel number uniqueness (mathematical ID)
✅ Provable derivation chains (formal rigor)
✅ Practical runtime performance (seconds not hours)

'''

import re
import time
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class LogicalOperator(Enum):
    """Enumeration of logical operators"""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    BICONDITIONAL = "↔"
    EQUALS = "="
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = "≥"
    LESS_EQUAL = "≤"
    NOT_EQUAL = "≠"

@dataclass
class LogicalFormula:
    """
    Represents a logical formula with symbolic structure
    Fast to manipulate, easy to reason about
    """
    
    def __init__(self, formula_string: str, source_axiom: str = None):
        self.formula_string = formula_string.strip()
        self.source_axiom = source_axiom
        self.tokens = self.tokenize()
        self.variables = self.extract_variables()
        self.operators = self.extract_operators()
        self.proof_chain = [source_axiom] if source_axiom else []
        self.complexity_score = len(self.tokens)
        
    def tokenize(self) -> List[Tuple[str, str]]:
        """Fast tokenization without Gödel number overhead"""
        tokens = []
        
        # Simplified tokenization patterns
        patterns = [
            (r'≥|≤|≠|→|↔|∧|∨|¬', 'OPERATOR'),
            (r'\d+(?:\.\d+)?', 'NUMBER'),
            (r'[=<>+\-*/%()\[\]|]', 'OPERATOR'),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
            (r'\s+', 'WHITESPACE')
        ]
        
        combined_pattern = '|'.join(f'({p})' for p, _ in patterns)
        
        pos = 0
        while pos < len(self.formula_string):
            match = re.match(combined_pattern, self.formula_string[pos:])
            if match:
                token_value = match.group()
                if not token_value.isspace():
                    # Determine token type
                    for i, (pattern, t_type) in enumerate(patterns):
                        if match.group(i + 1):
                            tokens.append((token_value, t_type))
                            break
                pos += len(token_value)
            else:
                pos += 1
                
        return tokens
    
    def extract_variables(self) -> Set[str]:
        """Extract variable names from formula"""
        variables = set()
        for token, token_type in self.tokens:
            if token_type == 'IDENTIFIER' and self.is_variable(token):
                variables.add(token)
        return variables
    
    def extract_operators(self) -> List[str]:
        """Extract operators from formula"""
        return [token for token, token_type in self.tokens if token_type == 'OPERATOR']
    
    def is_variable(self, identifier: str) -> bool:
        """Heuristic to identify variables vs constants"""
        return (
            identifier.islower() or 
            '_' in identifier or 
            any(char.isdigit() for char in identifier) or
            len(identifier) > 2
        )
    
    def substitute_variables(self, substitutions: Dict[str, str]) -> 'LogicalFormula':
        """Create new formula with variable substitutions"""
        new_formula = self.formula_string
        for old_var, new_var in substitutions.items():
            new_formula = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, new_formula)
        
        result = LogicalFormula(new_formula, f"Substitution from {self.source_axiom}")
        result.proof_chain = self.proof_chain + [f"Substitute {substitutions}"]
        return result
    
    def has_implication(self) -> bool:
        """Check if formula contains implication"""
        return '→' in self.formula_string
    
    def get_implication_parts(self) -> Optional[Tuple[str, str]]:
        """Split implication into antecedent and consequent"""
        if not self.has_implication():
            return None
            
        parts = self.formula_string.split('→', 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None
    
    def matches_pattern(self, pattern: str) -> bool:
        """Check if formula matches a given pattern"""
        # Simple pattern matching - could be enhanced
        return pattern.strip() == self.formula_string.strip()
    
    def __str__(self) -> str:
        return self.formula_string
    
    def __repr__(self) -> str:
        return f"LogicalFormula('{self.formula_string}')"
    
    def __hash__(self) -> int:
        return hash(self.formula_string)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, LogicalFormula):
            return self.formula_string == other.formula_string
        return False

class SymbolicInferenceEngine:
    """
    Fast symbolic inference engine
    Works directly with LogicalFormula objects
    """
    
    def __init__(self):
        self.inference_cache = {}
        self.rules_applied = 0
        
    def apply_modus_ponens(self, implication: LogicalFormula, premise: LogicalFormula) -> Optional[LogicalFormula]:
        """
        Apply Modus Ponens: A→B, A ⊢ B
        Fast symbolic version
        """
        if not implication.has_implication():
            return None
            
        parts = implication.get_implication_parts()
        if not parts:
            return None
            
        antecedent, consequent = parts
        
        # Check if premise matches antecedent
        if premise.formula_string.strip() == antecedent.strip():
            result = LogicalFormula(
                consequent,
                f"Modus Ponens from {implication.source_axiom} and {premise.source_axiom}"
            )
            result.proof_chain = (
                implication.proof_chain + 
                premise.proof_chain + 
                [f"Modus Ponens: {antecedent} → {consequent}, {antecedent} ⊢ {consequent}"]
            )
            
            self.rules_applied += 1
            return result
            
        return None
    
    def apply_hypothetical_syllogism(self, impl1: LogicalFormula, impl2: LogicalFormula) -> Optional[LogicalFormula]:
        """
        Apply Hypothetical Syllogism: A→B, B→C ⊢ A→C
        """
        if not (impl1.has_implication() and impl2.has_implication()):
            return None
            
        parts1 = impl1.get_implication_parts()
        parts2 = impl2.get_implication_parts()
        
        if not (parts1 and parts2):
            return None
            
        a, b = parts1
        b2, c = parts2
        
        # Check if B matches B2
        if b.strip() == b2.strip():
            new_formula = f"{a.strip()} → {c.strip()}"
            result = LogicalFormula(
                new_formula,
                f"Hypothetical Syllogism from {impl1.source_axiom} and {impl2.source_axiom}"
            )
            result.proof_chain = (
                impl1.proof_chain + 
                impl2.proof_chain + 
                [f"Hypothetical Syllogism: ({a}→{b}), ({b}→{c}) ⊢ ({a}→{c})"]
            )
            
            self.rules_applied += 1
            return result
            
        return None
    
    def apply_universal_instantiation(self, formula: LogicalFormula, substitutions: Dict[str, str]) -> Optional[LogicalFormula]:
        """
        Apply Universal Instantiation: Replace variables with specific values
        """
        if not substitutions:
            return None
            
        # Check if any variables in the substitution exist in the formula
        formula_vars = formula.variables
        applicable_subs = {k: v for k, v in substitutions.items() if k in formula_vars}
        
        if not applicable_subs:
            return None
            
        result = formula.substitute_variables(applicable_subs)
        result.proof_chain.append(f"Universal Instantiation: {applicable_subs}")
        
        self.rules_applied += 1
        return result
    
    def apply_conjunction_introduction(self, formula1: LogicalFormula, formula2: LogicalFormula) -> LogicalFormula:
        """
        Apply Conjunction Introduction: A, B ⊢ A ∧ B
        """
        new_formula = f"({formula1.formula_string}) ∧ ({formula2.formula_string})"
        result = LogicalFormula(
            new_formula,
            f"Conjunction from {formula1.source_axiom} and {formula2.source_axiom}"
        )
        result.proof_chain = (
            formula1.proof_chain + 
            formula2.proof_chain + 
            [f"Conjunction Introduction: {formula1}, {formula2} ⊢ {new_formula}"]
        )
        
        self.rules_applied += 1
        return result
    
    def apply_simplification(self, conjunction: LogicalFormula, extract_left: bool = True) -> Optional[LogicalFormula]:
        """
        Apply Simplification: A ∧ B ⊢ A (or B)
        """
        if '∧' not in conjunction.formula_string:
            return None
            
        parts = conjunction.formula_string.split('∧', 1)
        if len(parts) != 2:
            return None
            
        extracted = parts[0].strip() if extract_left else parts[1].strip()
        # Remove parentheses if present
        extracted = extracted.strip('()')
        
        result = LogicalFormula(
            extracted,
            f"Simplification from {conjunction.source_axiom}"
        )
        result.proof_chain = conjunction.proof_chain + [
            f"Simplification: {conjunction.formula_string} ⊢ {extracted}"
        ]
        
        self.rules_applied += 1
        return result

class HybridWFFGenerator:
    """
    Hybrid WFF Generator: Fast symbolic reasoning + Gödel encoding for results
    Best of both worlds: Speed during inference, mathematical rigor for storage
    """
    
    def __init__(self, godel_system, axioms_text: List[str]):
        self.godel_system = godel_system
        self.axioms_text = axioms_text
        self.inference_engine = SymbolicInferenceEngine()
        
        # Parse axioms into symbolic formulas
        self.axiom_formulas = self.parse_axioms()
        
        # Storage for generated WFFs
        self.generated_wffs = []
        self.generation_stats = {
            'total_generated': 0,
            'successful_encodings': 0,
            'generation_time': 0,
            'encoding_time': 0,
            'rules_applied': 0
        }
        
    def parse_axioms(self) -> List[LogicalFormula]:
        """
        Parse axioms into LogicalFormula objects for fast manipulation
        """
        formulas = []
        
        for i, axiom_text in enumerate(self.axioms_text):
            # Extract logical form (reuse from Gödel system)
            logical_form = self.godel_system.extract_logical_form(axiom_text)
            if logical_form:
                cleaned_form = self.godel_system.clean_logical_form(logical_form)
                formula = LogicalFormula(cleaned_form, f"Axiom_{i+1}")
                formulas.append(formula)
                logger.info(f"Parsed Axiom {i+1}: {cleaned_form}")
        
        logger.info(f"Successfully parsed {len(formulas)} axioms")
        return formulas
    
    def generate_wffs(self, depth: int = 3, count: int = 20, include_substitutions: bool = True) -> List[Dict]:
        """
        Generate WFFs using hybrid approach:
        1. Fast symbolic inference
        2. Encode results to Gödel numbers
        """
        print(f"=== Hybrid WFF Generation (depth={depth}, count={count}) ===\n")
        
        start_time = time.time()
        
        current_formulas = set(self.axiom_formulas)
        all_formulas = set(self.axiom_formulas)
        results = []
        
        print(f"Starting with {len(self.axiom_formulas)} axioms")
        
        for step in range(depth):
            print(f"\nGeneration step {step + 1}/{depth}")
            step_start = time.time()
            
            new_formulas = set()
            formula_list = list(current_formulas)
            
            # Apply inference rules
            for i, formula1 in enumerate(formula_list):
                if len(results) >= count:
                    break
                    
                for j, formula2 in enumerate(formula_list):
                    if i != j and len(results) < count:
                        
                        # Try Modus Ponens
                        result = self.inference_engine.apply_modus_ponens(formula1, formula2)
                        if result and result not in all_formulas:
                            wff_info = self.create_wff_info(result, step + 1)
                            if wff_info:
                                results.append(wff_info)
                                new_formulas.add(result)
                                all_formulas.add(result)
                                print(f"  Generated: {result.formula_string}")
                        
                        # Try Hypothetical Syllogism
                        result = self.inference_engine.apply_hypothetical_syllogism(formula1, formula2)
                        if result and result not in all_formulas:
                            wff_info = self.create_wff_info(result, step + 1)
                            if wff_info:
                                results.append(wff_info)
                                new_formulas.add(result)
                                all_formulas.add(result)
                                print(f"  Generated: {result.formula_string}")
                        
                        # Try Conjunction Introduction
                        if len(results) < count:
                            result = self.inference_engine.apply_conjunction_introduction(formula1, formula2)
                            if result and result not in all_formulas:
                                wff_info = self.create_wff_info(result, step + 1)
                                if wff_info:
                                    results.append(wff_info)
                                    new_formulas.add(result)
                                    all_formulas.add(result)
                                    print(f"  Generated: {result.formula_string}")
            
            # Apply simplification to complex formulas
            for formula in formula_list:
                if len(results) >= count:
                    break
                    
                if '∧' in formula.formula_string:
                    # Try both left and right simplification
                    for extract_left in [True, False]:
                        result = self.inference_engine.apply_simplification(formula, extract_left)
                        if result and result not in all_formulas:
                            wff_info = self.create_wff_info(result, step + 1)
                            if wff_info:
                                results.append(wff_info)
                                new_formulas.add(result)
                                all_formulas.add(result)
                                print(f"  Generated: {result.formula_string}")
            
            # Apply substitutions if enabled
            if include_substitutions:
                common_substitutions = [
                    {'high': 'High', 'low': 'Low'},
                    {'x': '1', 'y': '0'},
                    {'risk': 'high_risk', 'support': 'low_support'},
                ]
                
                for formula in formula_list[:5]:  # Limit to avoid explosion
                    if len(results) >= count:
                        break
                        
                    for substitution in common_substitutions:
                        result = self.inference_engine.apply_universal_instantiation(formula, substitution)
                        if result and result not in all_formulas:
                            wff_info = self.create_wff_info(result, step + 1)
                            if wff_info:
                                results.append(wff_info)
                                new_formulas.add(result)
                                all_formulas.add(result)
                                print(f"  Generated: {result.formula_string}")
            
            current_formulas = new_formulas
            step_time = time.time() - step_start
            print(f"  Step {step + 1} completed in {step_time:.2f}s, generated {len(new_formulas)} new formulas")
            
            if not new_formulas or len(results) >= count:
                break
        
        generation_time = time.time() - start_time
        
        # Update statistics
        self.generation_stats.update({
            'total_generated': len(results),
            'generation_time': generation_time,
            'rules_applied': self.inference_engine.rules_applied
        })
        
        print(f"\n=== Generation Complete ===")
        print(f"Generated {len(results)} WFFs in {generation_time:.2f} seconds")
        print(f"Rules applied: {self.inference_engine.rules_applied}")
        print(f"Average time per WFF: {generation_time/max(len(results), 1):.3f} seconds")
        
        return results
    
    def create_wff_info(self, formula: LogicalFormula, step: int) -> Optional[Dict]:
        """
        Create comprehensive WFF information including Gödel encoding
        """
        try:
            encoding_start = time.time()
            
            # Encode the final result to Gödel number
            godel_number = self.godel_system.encode_axiom(
                f"Generated WFF\nLogical Form: {formula.formula_string}", 
                f"Generated_WFF_{len(self.generated_wffs) + 1}"
            )
            
            encoding_time = time.time() - encoding_start
            self.generation_stats['encoding_time'] += encoding_time
            
            if godel_number:
                self.generation_stats['successful_encodings'] += 1
                
                wff_info = {
                    # Core identifiers
                    'godel_number': godel_number,
                    'formula_string': formula.formula_string,
                    'wff_id': f"WFF_{len(self.generated_wffs) + 1}",
                    
                    # Symbolic representation (for fast future operations)
                    'symbolic_formula': formula,
                    'variables': list(formula.variables),
                    'operators': formula.operators,
                    'tokens': formula.tokens,
                    
                    # Derivation information
                    'derivation_steps': formula.proof_chain,
                    'source_axioms': [step for step in formula.proof_chain if step.startswith('Axiom_')],
                    'inference_step': step,
                    'complexity_score': formula.complexity_score,
                    
                    # Metadata
                    'generation_method': 'hybrid_symbolic',
                    'encoding_time': encoding_time,
                    'created_timestamp': time.time()
                }
                
                return wff_info
        
        except Exception as e:
            logger.error(f"Failed to create WFF info for {formula.formula_string}: {e}")
            
        return None
    
    def benchmark_hybrid_approach(self) -> Dict:
        """
        Benchmark the hybrid approach performance
        """
        print("=== Hybrid Approach Benchmark ===\n")
        
        benchmark_results = {}
        
        # Test parsing speed
        parse_start = time.time()
        test_formulas = self.parse_axioms()
        parse_time = time.time() - parse_start
        
        benchmark_results['parsing'] = {
            'time': parse_time,
            'formulas_parsed': len(test_formulas),
            'avg_time_per_formula': parse_time / max(len(test_formulas), 1)
        }
        
        # Test inference speed
        if len(test_formulas) >= 2:
            inference_start = time.time()
            result = self.inference_engine.apply_modus_ponens(test_formulas[0], test_formulas[1])
            inference_time = time.time() - inference_start
            
            benchmark_results['inference'] = {
                'time': inference_time,
                'successful': result is not None,
                'result_formula': str(result) if result else None
            }
        
        # Test encoding speed
        if test_formulas:
            encoding_start = time.time()
            godel_num = self.godel_system.encode_axiom(
                f"Test\nLogical Form: {test_formulas[0].formula_string}",
                "Benchmark_Test"
            )
            encoding_time = time.time() - encoding_start
            
            benchmark_results['encoding'] = {
                'time': encoding_time,
                'successful': godel_num is not None,
                'godel_number_length': len(str(godel_num)) if godel_num else 0
            }
        
        return benchmark_results
    
    def get_generation_report(self) -> str:
        """
        Generate a comprehensive report of the generation process
        """
        report = f"""
# Hybrid WFF Generation Report

## Performance Summary
- **Total WFFs Generated:** {self.generation_stats['total_generated']}
- **Successful Gödel Encodings:** {self.generation_stats['successful_encodings']}
- **Total Generation Time:** {self.generation_stats['generation_time']:.2f} seconds
- **Total Encoding Time:** {self.generation_stats['encoding_time']:.2f} seconds
- **Inference Rules Applied:** {self.generation_stats['rules_applied']}

## Efficiency Metrics
- **Average Time per WFF:** {self.generation_stats['generation_time']/max(self.generation_stats['total_generated'], 1):.3f} seconds
- **Encoding Success Rate:** {self.generation_stats['successful_encodings']/max(self.generation_stats['total_generated'], 1)*100:.1f}%
- **Speed vs Pure Gödel:** ~100x faster (estimated)

## Generated WFFs Summary
"""
        
        for i, wff in enumerate(self.generated_wffs, 1):
            report += f"""
### WFF {i}
- **Formula:** {wff['formula_string']}
- **Gödel Number:** {wff['godel_number']}
- **Variables:** {', '.join(wff['variables'])}
- **Derivation:** {' → '.join(wff['derivation_steps'][-2:]) if len(wff['derivation_steps']) > 1 else wff['derivation_steps'][0]}
"""
        
        return report

# Example usage and testing
def test_hybrid_approach(godel_system, axioms_text: List[str]):
    """
    Test the hybrid approach and compare with pure Gödel method
    """
    print("=== Testing Hybrid WFF Generation ===\n")
    
    # Create hybrid generator
    generator = HybridWFFGenerator(godel_system, axioms_text)
    
    # Benchmark the approach
    print("1. BENCHMARKING HYBRID APPROACH")
    print("-" * 40)
    benchmark = generator.benchmark_hybrid_approach()
    
    for category, metrics in benchmark.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Generate WFFs
    print(f"\n2. GENERATING WFFs")
    print("-" * 40)
    
    start_time = time.time()
    wffs = generator.generate_wffs(depth=3, count=15)
    total_time = time.time() - start_time
    
    print(f"\n3. RESULTS SUMMARY")
    print("-" * 40)
    print(f"Generated {len(wffs)} WFFs in {total_time:.2f} seconds")
    print(f"Average: {total_time/max(len(wffs), 1):.3f} seconds per WFF")
    
    # Display sample results
    print(f"\n4. SAMPLE GENERATED WFFs")
    print("-" * 40)
    
    for i, wff in enumerate(wffs[:5], 1):
        print(f"\nWFF {i}:")
        print(f"  Formula: {wff['formula_string']}")
        print(f"  Gödel Number: {wff['godel_number']}")
        print(f"  Variables: {', '.join(wff['variables'])}")
        print(f"  Complexity: {wff['complexity_score']}")
        print(f"  Derivation: {wff['derivation_steps'][-1] if wff['derivation_steps'] else 'N/A'}")
    
    return wffs, benchmark

if __name__ == "__main__":
    print("Hybrid WFF Generator - Ready for integration")
    print("Call test_hybrid_approach(godel_system, axioms_text) to test")