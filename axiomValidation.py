import re
from typing import Dict, List, Tuple, Any, Set, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AxiomValidator:
    """
    Enhanced axiom validator with comprehensive analysis capabilities
    """
    
    def __init__(self, godel_system):
        self.godel_system = godel_system
        self.validation_cache = {}
        
    def validate_axiom_set(self, axioms: List[str], axiom_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of axiom set
        """
        if axiom_names is None:
            axiom_names = [f"Axiom_{i+1}" for i in range(len(axioms))]
        
        logger.info(f"Validating set of {len(axioms)} axioms")
        
        # Individual axiom validation
        individual_results = {}
        successfully_parsed = []
        
        for axiom, name in zip(axioms, axiom_names):
            result = self.validate_individual_axiom(axiom, name)
            individual_results[name] = result
            
            if result['successfully_encoded']:
                successfully_parsed.append((axiom, name))
        
        # Set-level validation (only on successfully parsed axioms)
        set_results = {
            'individual_axiom_results': individual_results,
            'set_level_analysis': {},
            'summary': {
                'total_axioms': len(axioms),
                'successfully_parsed': len(successfully_parsed),
                'parsing_success_rate': len(successfully_parsed) / len(axioms) if axioms else 0
            }
        }
        
        if successfully_parsed:
            set_results['set_level_analysis'] = {
                'consistency_analysis': self.analyze_consistency([a[0] for a in successfully_parsed]),
                'completeness_analysis': self.analyze_completeness([a[0] for a in successfully_parsed]),
                'variable_analysis': self.analyze_variables([a[0] for a in successfully_parsed]),
                'complexity_analysis': self.analyze_complexity([a[0] for a in successfully_parsed])
            }
        
        return set_results
    
    def validate_individual_axiom(self, axiom: str, axiom_name: str = None) -> Dict[str, Any]:
        """
        Detailed validation of individual axiom
        """
        result = {
            'axiom_name': axiom_name or 'Unnamed',
            'successfully_encoded': False,
            'godel_number': None,
            'issues': [],
            'quality_metrics': {},
            'structural_analysis': {}
        }
        
        try:
            # Attempt encoding
            godel_number = self.godel_system.encode_axiom(axiom, axiom_name)
            
            if godel_number is not None:
                result['successfully_encoded'] = True
                result['godel_number'] = godel_number
                
                # Get metadata for analysis
                metadata = self.godel_system.axiom_metadata.get(axiom_name, {})
                
                # Quality metrics
                result['quality_metrics'] = self.compute_quality_metrics(metadata)
                
                # Structural analysis
                result['structural_analysis'] = self.analyze_structure(metadata)
                
            else:
                result['issues'].append("Failed to encode axiom")
                
        except Exception as e:
            result['issues'].append(f"Validation error: {str(e)}")
        
        return result
    
    def compute_quality_metrics(self, metadata: Dict) -> Dict[str, Any]:
        """
        Compute quality metrics for an axiom
        """
        tokens = metadata.get('tokens', [])
        logical_form = metadata.get('logical_form', '')
        
        if not tokens:
            return {'error': 'No tokens available'}
        
        # Count different token types
        operators = [t for t in tokens if t[1] in ['OPERATOR', 'KEYWORD']]
        identifiers = [t for t in tokens if t[1] == 'IDENTIFIER']
        numbers = [t for t in tokens if t[1] == 'NUMBER']
        
        # Complexity metrics
        complexity_score = len(operators) * 2 + len(identifiers) + len(numbers) * 0.5
        
        # Logical depth (nesting level)
        logical_depth = logical_form.count('(') + logical_form.count('[')
        
        return {
            'total_tokens': len(tokens),
            'operator_count': len(operators),
            'identifier_count': len(identifiers),
            'number_count': len(numbers),
            'complexity_score': complexity_score,
            'logical_depth': logical_depth,
            'has_implication': '→' in logical_form,
            'has_biconditional': '↔' in logical_form,
            'has_conjunction': '∧' in logical_form,
            'has_disjunction': '∨' in logical_form,
            'has_negation': '¬' in logical_form
        }
    
    def analyze_structure(self, metadata: Dict) -> Dict[str, Any]:
        """
        Analyze logical structure of axiom
        """
        logical_form = metadata.get('logical_form', '')
        tokens = metadata.get('tokens', [])
        
        # Identify structure patterns
        structure_patterns = {
            'simple_conditional': r'^[^→]*→[^→]*$',
            'biconditional': r'↔',
            'complex_conditional': r'→.*→',
            'conjunction_chain': r'∧.*∧',
            'disjunction_chain': r'∨.*∨',
            'nested_formula': r'\([^)]*[→↔∧∨][^)]*\)'
        }
        
        structure_analysis = {}
        for pattern_name, pattern in structure_patterns.items():
            structure_analysis[pattern_name] = bool(re.search(pattern, logical_form))
        
        # Analyze variable relationships
        identifiers = [t[0] for t in tokens if t[1] == 'IDENTIFIER']
        unique_identifiers = list(set(identifiers))
        
        structure_analysis.update({
            'unique_variables': len(unique_identifiers),
            'variable_reuse': len(identifiers) - len(unique_identifiers),
            'variable_list': unique_identifiers
        })
        
        return structure_analysis
    
    def analyze_consistency(self, axioms: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency across axioms
        """
        # Extract all logical forms
        logical_forms = []
        for axiom in axioms:
            logical_form = self.godel_system.extract_logical_form(axiom)
            if logical_form:
                logical_forms.append(self.godel_system.clean_logical_form(logical_form))
        
        # Look for potential contradictions
        contradictions = []
        implications = {}
        
        for i, form in enumerate(logical_forms):
            # Extract simple implications A → B
            impl_matches = re.findall(r'([^→]+)→([^→]+)', form)
            
            for premise, conclusion in impl_matches:
                premise = premise.strip()
                conclusion = conclusion.strip()
                
                key = premise
                if key in implications:
                    if implications[key] != conclusion:
                        contradictions.append({
                            'type': 'conflicting_implications',
                            'premise': premise,
                            'conclusion1': implications[key],
                            'conclusion2': conclusion,
                            'axiom_indices': [implications[key + '_index'], i]
                        })
                else:
                    implications[key] = conclusion
                    implications[key + '_index'] = i
        
        return {
            'potential_contradictions': contradictions,
            'total_implications_found': len([k for k in implications.keys() if not k.endswith('_index')]),
            'consistency_score': 1.0 if not contradictions else max(0.0, 1.0 - len(contradictions) * 0.2)
        }
    
    def analyze_completeness(self, axioms: List[str]) -> Dict[str, Any]:
        """
        Analyze domain completeness
        """
        # Define expected domain concepts
        domain_concepts = {
            'student_attributes': ['age', 'gender', 'international', 'scholarship'],
            'academic_performance': ['grade', 'gpa', 'credits', 'semester'],
            'financial_factors': ['debt', 'tuition', 'scholarship', 'loan'],
            'outcomes': ['dropout', 'graduate', 'retention', 'success'],
            'risk_factors': ['risk', 'probability', 'likelihood', 'chance']
        }
        
        # Check coverage
        concept_coverage = {}
        all_axiom_text = ' '.join(axioms).lower()
        
        for concept_group, keywords in domain_concepts.items():
            covered_keywords = []
            for keyword in keywords:
                if keyword in all_axiom_text:
                    covered_keywords.append(keyword)
            
            concept_coverage[concept_group] = {
                'covered_keywords': covered_keywords,
                'coverage_ratio': len(covered_keywords) / len(keywords),
                'total_keywords': len(keywords)
            }
        
        # Overall completeness score
        total_coverage = sum(data['coverage_ratio'] for data in concept_coverage.values())
        completeness_score = total_coverage / len(domain_concepts)
        
        return {
            'concept_coverage': concept_coverage,
            'completeness_score': completeness_score,
            'well_covered_concepts': [k for k, v in concept_coverage.items() if v['coverage_ratio'] > 0.5],
            'poorly_covered_concepts': [k for k, v in concept_coverage.items() if v['coverage_ratio'] < 0.3]
        }
    
    def analyze_variables(self, axioms: List[str]) -> Dict[str, Any]:
        """
        Analyze variable usage across axioms
        """
        all_variables = set()
        axiom_variables = {}
        
        for i, axiom in enumerate(axioms):
            logical_form = self.godel_system.extract_logical_form(axiom)
            if logical_form:
                cleaned_form = self.godel_system.clean_logical_form(logical_form)
                tokens = self.godel_system.tokenize(cleaned_form)
                
                variables = [t[0] for t in tokens if t[1] == 'IDENTIFIER' and self.godel_system.is_likely_variable(t[0])]
                axiom_variables[f"Axiom_{i+1}"] = list(set(variables))
                all_variables.update(variables)
        
        # Find shared variables
        variable_usage = {}
        for var in all_variables:
            usage = []
            for axiom_name, vars_in_axiom in axiom_variables.items():
                if var in vars_in_axiom:
                    usage.append(axiom_name)
            variable_usage[var] = usage
        
        return {
            'total_unique_variables': len(all_variables),
            'variables_by_axiom': axiom_variables,
            'variable_usage_frequency': variable_usage,
            'shared_variables': [var for var, usage in variable_usage.items() if len(usage) > 1],
            'isolated_variables': [var for var, usage in variable_usage.items() if len(usage) == 1]
        }
    
    def analyze_complexity(self, axioms: List[str]) -> Dict[str, Any]:
        """
        Analyze complexity distribution across axioms
        """
        complexity_scores = []
        structure_distribution = {
            'simple_implications': 0,
            'complex_conditionals': 0,
            'biconditionals': 0,
            'conjunctions': 0,
            'disjunctions': 0
        }
        
        for axiom in axioms:
            logical_form = self.godel_system.extract_logical_form(axiom)
            if logical_form:
                cleaned_form = self.godel_system.clean_logical_form(logical_form)
                tokens = self.godel_system.tokenize(cleaned_form)
                
                # Compute complexity score
                operators = [t for t in tokens if t[1] in ['OPERATOR', 'KEYWORD']]
                identifiers = [t for t in tokens if t[1] == 'IDENTIFIER']
                
                complexity_score = len(operators) * 2 + len(identifiers)
                complexity_scores.append(complexity_score)
                
                # Analyze structure
                if '→' in cleaned_form and cleaned_form.count('→') == 1:
                    structure_distribution['simple_implications'] += 1
                elif '→' in cleaned_form and cleaned_form.count('→') > 1:
                    structure_distribution['complex_conditionals'] += 1
                
                if '↔' in cleaned_form:
                    structure_distribution['biconditionals'] += 1
                if '∧' in cleaned_form:
                    structure_distribution['conjunctions'] += 1
                if '∨' in cleaned_form:
                    structure_distribution['disjunctions'] += 1
        
        # Compute statistics
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            max_complexity = max(complexity_scores)
            min_complexity = min(complexity_scores)
        else:
            avg_complexity = max_complexity = min_complexity = 0
        
        return {
            'complexity_scores': complexity_scores,
            'average_complexity': avg_complexity,
            'max_complexity': max_complexity,
            'min_complexity': min_complexity,
            'structure_distribution': structure_distribution,
            'total_analyzed': len(complexity_scores)
        }


# Advanced analysis functions
def analyze_axiom_relationships(godel_system) -> Dict[str, Any]:
    """
    Analyze relationships between encoded axioms
    """
    if len(godel_system.axiom_encodings) < 2:
        return {"error": "Need at least 2 axioms to analyze relationships"}
    
    relationships = {}
    axiom_items = list(godel_system.axiom_encodings.items())
    
    for i in range(len(axiom_items)):
        for j in range(i + 1, len(axiom_items)):
            name1, godel1 = axiom_items[i]
            name2, godel2 = axiom_items[j]
            
            # Compute relationship metrics
            gcd_val = compute_gcd(godel1, godel2)
            ratio = godel1 / godel2 if godel2 != 0 else float('inf')
            
            # Get shared variables
            metadata1 = godel_system.axiom_metadata.get(name1, {})
            metadata2 = godel_system.axiom_metadata.get(name2, {})
            
            tokens1 = metadata1.get('tokens', [])
            tokens2 = metadata2.get('tokens', [])
            
            vars1 = set(t[0] for t in tokens1 if t[1] == 'IDENTIFIER')
            vars2 = set(t[0] for t in tokens2 if t[1] == 'IDENTIFIER')
            
            shared_vars = vars1.intersection(vars2)
            
            relationships[f"{name1}_{name2}"] = {
                'gcd': gcd_val,
                'ratio': ratio,
                'shared_variables': list(shared_vars),
                'shared_variable_count': len(shared_vars),
                'total_variables': len(vars1.union(vars2))
            }
    
    return relationships


def compute_gcd(a: int, b: int) -> int:
    """Compute greatest common divisor"""
    while b:
        a, b = b, a % b
    return a