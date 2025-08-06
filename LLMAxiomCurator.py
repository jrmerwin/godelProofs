import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class AxiomCandidate:
    """Represents a WFF candidate for addition to axiom set"""
    wff_id: str
    formula: str
    godel_number: int
    derivation_steps: List[str]
    source_axioms: List[str]
    complexity_score: float
    llm_utility_score: int
    llm_reasoning: str
    category: str  # 'high_value_derived', 'simple_conclusion', 'rejected'
    confidence: float

class LLMAxiomCurator:
    """
    Uses LLM to intelligently select WFFs for addition to axiom set
    Balances mathematical rigor with practical utility
    """
    
    def __init__(self, api_provider='anthropic', api_key=None):
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.curation_cache = {}
        
        # Selection criteria weights
        self.selection_criteria = {
            'utility_threshold': 3,        # Minimum utility score
            'complexity_min': 2,           # Minimum complexity for derived rules
            'complexity_max': 15,          # Maximum complexity (avoid over-complex)
            'max_high_value': 12,          # Max high-value derived axioms
            'max_simple': 8,               # Max simple conclusions
            'diversity_bonus': 0.5         # Bonus for covering new variables
        }
    
    def curate_axioms(self, analyzed_wffs: List[Dict], original_axioms: List[str]) -> Dict:
        """
        Main method: Curate new axioms from generated WFFs
        
        Returns:
            Dictionary with selected axioms and analysis
        """
        print("=== LLM-Powered Axiom Curation ===\n")
        
        if not analyzed_wffs:
            return {'error': 'No WFFs provided for curation'}
        
        # Step 1: Pre-filter WFFs for efficiency
        candidates = self._pre_filter_candidates(analyzed_wffs)
        print(f"Pre-filtered to {len(candidates)} candidates from {len(analyzed_wffs)} WFFs")
        
        # Step 2: LLM evaluation of candidates
        evaluated_candidates = self._evaluate_candidates_with_llm(candidates, original_axioms)
        
        # Step 3: Final selection algorithm
        selected_axioms = self._select_final_axioms(evaluated_candidates)
        
        # Step 4: Generate comprehensive report
        curation_report = self._generate_curation_report(selected_axioms, evaluated_candidates, original_axioms)
        
        return curation_report
    
    def _pre_filter_candidates(self, analyzed_wffs: List[Dict]) -> List[Dict]:
        """
        Pre-filter WFFs to reduce LLM API calls
        """
        candidates = []
        
        for wff in analyzed_wffs:
            # Basic quality filters
            utility_score = wff.get('utility_score', 0)
            complexity = wff.get('complexity_score', 0)
            formula_length = len(wff.get('formula_string', ''))
            
            # Skip obviously bad candidates
            if (utility_score < 2 or 
                complexity < 1 or 
                complexity > 20 or
                formula_length < 5 or
                formula_length > 200):
                continue
            
            # Skip if no meaningful derivation
            if len(wff.get('derivation_steps', [])) < 1:
                continue
            
            # Skip if contains obvious errors or placeholders
            formula = wff.get('formula_string', '').lower()
            if any(word in formula for word in ['error', 'unknown', 'placeholder', 'test']):
                continue
            
            candidates.append(wff)
        
        # Sort by utility score for priority processing
        candidates.sort(key=lambda x: x.get('utility_score', 0), reverse=True)
        
        # Limit to top 40 candidates to manage API costs
        return candidates[:40]
    
    def _evaluate_candidates_with_llm(self, candidates: List[Dict], original_axioms: List[str]) -> List[AxiomCandidate]:
        """
        Use LLM to evaluate each candidate for axiom addition
        """
        print(f"Evaluating {len(candidates)} candidates with LLM...")
        
        evaluated = []
        
        # Process in batches to manage API rate limits
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(candidates)-1)//batch_size + 1}")
            
            batch_results = self._evaluate_batch(batch, original_axioms)
            evaluated.extend(batch_results)
            
            # Rate limiting
            time.sleep(2)
        
        return evaluated
    
    def _evaluate_batch(self, wff_batch: List[Dict], original_axioms: List[str]) -> List[AxiomCandidate]:
        """
        Evaluate a batch of WFFs with a single LLM call
        """
        # Create comprehensive evaluation prompt
        prompt = self._create_evaluation_prompt(wff_batch, original_axioms)
        
        try:
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response into candidates
            candidates = self._parse_evaluation_response(response, wff_batch)
            
            return candidates
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for batch: {e}")
            # Return default evaluations on failure
            return [self._create_default_candidate(wff) for wff in wff_batch]
    
    def _create_evaluation_prompt(self, wff_batch: List[Dict], original_axioms: List[str]) -> str:
        """
        Create detailed evaluation prompt for LLM
        """
        # Summarize original axioms
        axiom_summary = "\n".join([f"  {i+1}. {axiom[:100]}..." for i, axiom in enumerate(original_axioms[:10])])
        if len(original_axioms) > 10:
            axiom_summary += f"\n  ... and {len(original_axioms)-10} more axioms"
        
        # Format WFF candidates
        candidate_section = ""
        for i, wff in enumerate(wff_batch, 1):
            candidate_section += f"""
CANDIDATE {i}:
Formula: {wff.get('formula_string', 'N/A')}
Derivation: {' → '.join(wff.get('derivation_steps', [])[-2:]) if len(wff.get('derivation_steps', [])) > 0 else 'Direct'}
Source Axioms: {', '.join(wff.get('source_axioms', []))}
Complexity: {wff.get('complexity_score', 0)}
Current Utility Score: {wff.get('utility_score', 0)}/5
"""
        
        prompt = f"""You are an expert in formal logic and student retention modeling. I need you to evaluate generated logical formulas (WFFs) to determine which ones should be added as new axioms to expand our reasoning capabilities.

CONTEXT:
We have a student retention prediction system with {len(original_axioms)} original axioms covering factors like:
- Financial risk (debt, scholarships, tuition)
- Academic performance (grades, credits, GPA)
- Demographics and support systems
- Intervention strategies

ORIGINAL AXIOMS (first 10):
{axiom_summary}

EVALUATION TASK:
For each candidate formula below, provide a structured evaluation to determine if it should be added as a new axiom.

{candidate_section}

For EACH candidate, provide your evaluation in this exact JSON format:

{{
  "candidate_1": {{
    "recommended_action": "high_value_derived" | "simple_conclusion" | "rejected",
    "utility_score": 1-5,
    "reasoning": "Brief explanation of decision",
    "category_justification": "Why this category?",
    "practical_value": "How would this help in practice?",
    "confidence": 0.0-1.0
  }},
  "candidate_2": {{ ... }},
  ...
}}

EVALUATION CRITERIA:
- **high_value_derived**: Complex, non-obvious insights that combine multiple factors. Novel relationships that would be valuable for reasoning chains. Score 4-5.
- **simple_conclusion**: Straightforward but useful facts. Direct implications that save computation time. Score 3-4.
- **rejected**: Trivial, obvious, redundant with existing axioms, or not practically useful. Score 1-2.

Consider:
1. Does this add NEW reasoning capability not easily derivable?
2. Would this be useful for practical student retention decisions?
3. Is it non-trivial but not overly complex?
4. Does it capture a meaningful relationship between factors?
5. Would other rules benefit from having this as a starting point?

Aim to recommend roughly 60% high_value_derived, 30% simple_conclusion, 10% rejected to get a balanced expansion.

Please respond with ONLY the JSON evaluation."""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (Anthropic Claude)
        """
        try:
            import anthropic
            import requests
            
            # Try anthropic library first
            try:
                client = anthropic.Anthropic(api_key=self.api_key)
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
            except ImportError:
                # Fallback to requests
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                
                data = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4000,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()["content"][0]["text"]
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_evaluation_response(self, response: str, wff_batch: List[Dict]) -> List[AxiomCandidate]:
        """
        Parse LLM response into AxiomCandidate objects
        """
        candidates = []
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end] if json_start != -1 else response
            
            evaluations = json.loads(json_str)
            
            for i, wff in enumerate(wff_batch, 1):
                eval_key = f"candidate_{i}"
                
                if eval_key in evaluations:
                    eval_data = evaluations[eval_key]
                    
                    candidate = AxiomCandidate(
                        wff_id=wff.get('wff_id', f"WFF_{i}"),
                        formula=wff.get('formula_string', ''),
                        godel_number=wff.get('godel_number', 0),
                        derivation_steps=wff.get('derivation_steps', []),
                        source_axioms=wff.get('source_axioms', []),
                        complexity_score=wff.get('complexity_score', 0),
                        llm_utility_score=eval_data.get('utility_score', 0),
                        llm_reasoning=eval_data.get('reasoning', ''),
                        category=eval_data.get('recommended_action', 'rejected'),
                        confidence=eval_data.get('confidence', 0.5)
                    )
                    
                    candidates.append(candidate)
                else:
                    # Fallback for missing evaluation
                    candidates.append(self._create_default_candidate(wff))
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return default candidates on parse failure
            candidates = [self._create_default_candidate(wff) for wff in wff_batch]
        
        return candidates
    
    def _create_default_candidate(self, wff: Dict) -> AxiomCandidate:
        """
        Create default candidate when LLM evaluation fails
        """
        return AxiomCandidate(
            wff_id=wff.get('wff_id', 'Unknown'),
            formula=wff.get('formula_string', ''),
            godel_number=wff.get('godel_number', 0),
            derivation_steps=wff.get('derivation_steps', []),
            source_axioms=wff.get('source_axioms', []),
            complexity_score=wff.get('complexity_score', 0),
            llm_utility_score=2,  # Default neutral score
            llm_reasoning="LLM evaluation failed - using default assessment",
            category='rejected',
            confidence=0.3
        )
    
    def _select_final_axioms(self, evaluated_candidates: List[AxiomCandidate]) -> Dict:
        """
        Final selection algorithm combining LLM recommendations with diversity
        """
        # Separate by category
        high_value = [c for c in evaluated_candidates if c.category == 'high_value_derived']
        simple = [c for c in evaluated_candidates if c.category == 'simple_conclusion']
        
        # Sort by combined score (utility * confidence)
        high_value.sort(key=lambda x: x.llm_utility_score * x.confidence, reverse=True)
        simple.sort(key=lambda x: x.llm_utility_score * x.confidence, reverse=True)
        
        # Select top candidates with diversity consideration
        selected_high_value = self._select_diverse_candidates(
            high_value, 
            self.selection_criteria['max_high_value']
        )
        
        selected_simple = self._select_diverse_candidates(
            simple,
            self.selection_criteria['max_simple']
        )
        
        return {
            'high_value_derived': selected_high_value,
            'simple_conclusions': selected_simple,
            'total_selected': len(selected_high_value) + len(selected_simple)
        }
    
    def _select_diverse_candidates(self, candidates: List[AxiomCandidate], max_count: int) -> List[AxiomCandidate]:
        """
        Select candidates ensuring diversity in variables/concepts
        """
        if len(candidates) <= max_count:
            return candidates
        
        selected = []
        used_variables = set()
        
        # First pass: select highest-scoring candidates with new variables
        for candidate in candidates:
            if len(selected) >= max_count:
                break
            
            # Extract variables from formula
            candidate_vars = self._extract_variables(candidate.formula)
            
            # Check for new variables
            new_vars = candidate_vars - used_variables
            
            # Select if high quality or brings new variables
            if (candidate.llm_utility_score >= 4 or 
                (len(new_vars) > 0 and candidate.llm_utility_score >= 3)):
                selected.append(candidate)
                used_variables.update(candidate_vars)
        
        # Second pass: fill remaining slots with highest scoring
        remaining_candidates = [c for c in candidates if c not in selected]
        remaining_count = max_count - len(selected)
        
        selected.extend(remaining_candidates[:remaining_count])
        
        return selected
    
    def _extract_variables(self, formula: str) -> Set[str]:
        """
        Extract variable names from formula string
        """
        import re
        # Simple variable extraction - alphanumeric identifiers
        variables = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula))
        # Remove common operators/keywords
        operators = {'AND', 'OR', 'NOT', 'IF', 'THEN', 'High', 'Low', 'True', 'False'}
        return variables - operators
    
    def _generate_curation_report(self, selected_axioms: Dict, all_evaluated: List[AxiomCandidate], original_axioms: List[str]) -> Dict:
        """
        Generate comprehensive curation report
        """
        high_value = selected_axioms['high_value_derived']
        simple = selected_axioms['simple_conclusions']
        
        # Create new axiom list
        new_axiom_formulas = []
        for candidate in high_value + simple:
            new_axiom_formulas.append(candidate.formula)
        
        extended_axiom_set = original_axioms + new_axiom_formulas
        
        # Generate report
        report = {
            'curation_summary': {
                'original_axiom_count': len(original_axioms),
                'candidates_evaluated': len(all_evaluated),
                'high_value_selected': len(high_value),
                'simple_conclusions_selected': len(simple),
                'total_new_axioms': len(new_axiom_formulas),
                'final_axiom_count': len(extended_axiom_set)
            },
            
            'selected_high_value_axioms': [
                {
                    'formula': c.formula,
                    'godel_number': c.godel_number,
                    'utility_score': c.llm_utility_score,
                    'reasoning': c.llm_reasoning,
                    'confidence': c.confidence,
                    'source_axioms': c.source_axioms
                } for c in high_value
            ],
            
            'selected_simple_conclusions': [
                {
                    'formula': c.formula,
                    'godel_number': c.godel_number,
                    'utility_score': c.llm_utility_score,
                    'reasoning': c.llm_reasoning,
                    'confidence': c.confidence
                } for c in simple
            ],
            
            'extended_axiom_set': extended_axiom_set,
            'new_axiom_formulas': new_axiom_formulas,
            
            'quality_metrics': {
                'average_utility_score': sum(c.llm_utility_score for c in high_value + simple) / len(high_value + simple) if (high_value + simple) else 0,
                'average_confidence': sum(c.confidence for c in high_value + simple) / len(high_value + simple) if (high_value + simple) else 0,
                'variable_diversity': len(self._get_all_variables_in_selected(high_value + simple))
            }
        }
        
        return report
    
    def _get_all_variables_in_selected(self, candidates: List[AxiomCandidate]) -> Set[str]:
        """
        Get all unique variables in selected candidates
        """
        all_vars = set()
        for candidate in candidates:
            all_vars.update(self._extract_variables(candidate.formula))
        return all_vars
    
    def print_curation_summary(self, report: Dict):
        """
        Print human-readable curation summary
        """
        summary = report['curation_summary']
        
        print("=== AXIOM CURATION SUMMARY ===")
        print(f"📊 Original axioms: {summary['original_axiom_count']}")
        print(f"🔍 Candidates evaluated: {summary['candidates_evaluated']}")
        print(f"🎯 High-value axioms selected: {summary['high_value_selected']}")
        print(f"📝 Simple conclusions selected: {summary['simple_conclusions_selected']}")
        print(f"📈 Total new axioms: {summary['total_new_axioms']}")
        print(f"🏁 Final axiom count: {summary['final_axiom_count']}")
        
        print(f"\n=== QUALITY METRICS ===")
        metrics = report['quality_metrics']
        print(f"Average utility score: {metrics['average_utility_score']:.1f}/5")
        print(f"Average confidence: {metrics['average_confidence']:.2f}")
        print(f"Variable diversity: {metrics['variable_diversity']} unique variables")
        
        print(f"\n=== SELECTED HIGH-VALUE AXIOMS ===")
        for i, axiom in enumerate(report['selected_high_value_axioms'], 1):
            print(f"{i}. {axiom['formula']}")
            print(f"   Utility: {axiom['utility_score']}/5 | Confidence: {axiom['confidence']:.2f}")
            print(f"   Reason: {axiom['reasoning']}")
            print()
        
        print(f"=== SELECTED SIMPLE CONCLUSIONS ===")
        for i, axiom in enumerate(report['selected_simple_conclusions'], 1):
            print(f"{i}. {axiom['formula']}")
            print(f"   Utility: {axiom['utility_score']}/5 | Confidence: {axiom['confidence']:.2f}")
            print()

# Usage example
def curate_axioms_from_wffs(analyzed_wffs: List[Dict], original_axioms: List[str]) -> Dict:
    """
    Main function to curate new axioms from generated WFFs
    """
    print("=== Starting LLM-Powered Axiom Curation ===\n")
    
    curator = LLMAxiomCurator()
    
    if not curator.api_key:
        print("⚠️ No Anthropic API key found!")
        return {'error': 'No API key'}
    
    # Perform curation
    curation_report = curator.curate_axioms(analyzed_wffs, original_axioms)
    
    # Print summary
    curator.print_curation_summary(curation_report)
    
    return curation_report

if __name__ == "__main__":
    print("LLM Axiom Curator - Ready for integration")
    print("Call curate_axioms_from_wffs(analyzed_wffs, original_axioms) to curate")