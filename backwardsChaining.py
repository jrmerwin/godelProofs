import time
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ProofGoal:
    """Represents a goal we're trying to prove"""
    formula: str
    depth: int = 0
    parent_goal: Optional['ProofGoal'] = None
    subgoals: List['ProofGoal'] = None
    
    def __post_init__(self):
        if self.subgoals is None:
            self.subgoals = []

@dataclass
class ProofStep:
    """Represents a single step in a proof"""
    rule_name: str
    premises: List[str]
    conclusion: str
    axioms_used: List[str]
    justification: str

class BackwardChainingProver:
    """
    Backward chaining theorem prover
    Starts with a goal and works backward to find axioms that prove it
    """
    
    def __init__(self, hybrid_generator):
        self.hybrid_generator = hybrid_generator
        self.axiom_formulas = hybrid_generator.axiom_formulas
        self.inference_engine = hybrid_generator.inference_engine
        
        # Create lookup structures for efficient backward search
        self.implications = self._index_implications()
        self.facts = self._index_facts()
        
        # Search state
        self.proof_cache = {}
        self.search_depth_limit = 10
        self.max_proof_attempts = 1000
        
    def _index_implications(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Index implications by their conclusions for backward chaining
        Returns: {conclusion: [(antecedent, full_implication, axiom_name), ...]}
        """
        implications = {}
        
        for formula in self.axiom_formulas:
            if formula.has_implication():
                parts = formula.get_implication_parts()
                if parts:
                    antecedent, consequent = parts
                    antecedent = antecedent.strip()
                    consequent = consequent.strip()
                    
                    if consequent not in implications:
                        implications[consequent] = []
                    
                    implications[consequent].append((
                        antecedent, 
                        formula.formula_string, 
                        formula.source_axiom
                    ))
        
        return implications
    
    def _index_facts(self) -> Set[str]:
        """
        Index direct facts (non-implication axioms) for lookup
        """
        facts = set()
        
        for formula in self.axiom_formulas:
            if not formula.has_implication():
                facts.add(formula.formula_string.strip())
        
        return facts
    
    def prove_goal(self, target_formula: str, max_depth: int = 5) -> Optional[Dict]:
        """
        Main method: Try to prove a target formula from axioms
        
        Args:
            target_formula: The formula we want to prove (e.g., "student_at_risk = 1")
            max_depth: Maximum search depth to prevent infinite loops
            
        Returns:
            Proof dictionary with complete derivation, or None if unprovable
        """
        print(f"=== Attempting to Prove: {target_formula} ===\n")
        
        start_time = time.time()
        self.search_depth_limit = max_depth
        
        # Check cache first
        if target_formula in self.proof_cache:
            print("Found cached proof!")
            return self.proof_cache[target_formula]
        
        # Initialize proof search
        initial_goal = ProofGoal(target_formula.strip(), 0)
        proof_steps = []
        
        # Try to prove the goal
        success, proof_tree = self._prove_goal_recursive(initial_goal, set(), proof_steps)
        
        search_time = time.time() - start_time
        
        if success:
            # Construct complete proof
            proof = self._construct_proof(proof_tree, proof_steps, target_formula)
            proof['search_time'] = search_time
            
            # Cache successful proof
            self.proof_cache[target_formula] = proof
            
            print(f"✅ PROOF FOUND in {search_time:.3f} seconds!")
            self._print_proof_summary(proof)
            
            return proof
        else:
            print(f"❌ No proof found in {search_time:.3f} seconds")
            print(f"Explored {len(proof_steps)} potential proof steps")
            return None
    
    def _prove_goal_recursive(self, goal: ProofGoal, visited: Set[str], proof_steps: List[ProofStep]) -> Tuple[bool, Optional[ProofGoal]]:
        """
        Recursive backward chaining proof search
        """
        # Prevent infinite loops
        if goal.formula in visited or goal.depth > self.search_depth_limit:
            return False, None
        
        visited.add(goal.formula)
        
        # Base case: Check if goal is a direct fact (axiom)
        if goal.formula in self.facts:
            print(f"  {'  ' * goal.depth}✓ Direct fact: {goal.formula}")
            return True, goal
        
        print(f"  {'  ' * goal.depth}🎯 Trying to prove: {goal.formula}")
        
        # Strategy 1: Find implications that conclude with our goal
        if goal.formula in self.implications:
            for antecedent, full_implication, axiom_name in self.implications[goal.formula]:
                print(f"  {'  ' * goal.depth}📋 Found rule: {full_implication}")
                
                # Create subgoal to prove the antecedent
                subgoal = ProofGoal(antecedent, goal.depth + 1, goal)
                goal.subgoals.append(subgoal)
                
                # Recursively try to prove antecedent
                sub_success, sub_proof = self._prove_goal_recursive(subgoal, visited.copy(), proof_steps)
                
                if sub_success:
                    # Record proof step
                    proof_step = ProofStep(
                        rule_name="Modus Ponens",
                        premises=[antecedent, full_implication],
                        conclusion=goal.formula,
                        axioms_used=[axiom_name],
                        justification=f"From {axiom_name}: {antecedent} → {goal.formula}"
                    )
                    proof_steps.append(proof_step)
                    
                    print(f"  {'  ' * goal.depth}✅ Proved via: {axiom_name}")
                    return True, goal
        
        # Strategy 2: Try to construct goal through conjunction breakdown
        if '∧' in goal.formula:
            conjuncts = [c.strip().strip('()') for c in goal.formula.split('∧')]
            
            print(f"  {'  ' * goal.depth}🔗 Breaking down conjunction: {conjuncts}")
            
            # Try to prove each conjunct
            all_proved = True
            conjunct_proofs = []
            
            for conjunct in conjuncts:
                subgoal = ProofGoal(conjunct, goal.depth + 1, goal)
                goal.subgoals.append(subgoal)
                
                sub_success, sub_proof = self._prove_goal_recursive(subgoal, visited.copy(), proof_steps)
                
                if sub_success:
                    conjunct_proofs.append(sub_proof)
                else:
                    all_proved = False
                    break
            
            if all_proved:
                proof_step = ProofStep(
                    rule_name="Conjunction Introduction",
                    premises=conjuncts,
                    conclusion=goal.formula,
                    axioms_used=[],
                    justification=f"Proved all conjuncts: {' ∧ '.join(conjuncts)}"
                )
                proof_steps.append(proof_step)
                
                print(f"  {'  ' * goal.depth}✅ Proved conjunction")
                return True, goal
        
        # Strategy 3: Try substitution/unification with existing facts
        for fact in self.facts:
            if self._formulas_unifiable(goal.formula, fact):
                print(f"  {'  ' * goal.depth}🔄 Unifiable with fact: {fact}")
                
                proof_step = ProofStep(
                    rule_name="Unification",
                    premises=[fact],
                    conclusion=goal.formula,
                    axioms_used=[f"Fact_{fact}"],
                    justification=f"Unifies with: {fact}"
                )
                proof_steps.append(proof_step)
                
                return True, goal
        
        print(f"  {'  ' * goal.depth}❌ Cannot prove: {goal.formula}")
        visited.remove(goal.formula)
        return False, None
    
    def _formulas_unifiable(self, formula1: str, formula2: str) -> bool:
        """
        Simple unification check - can be enhanced with proper unification algorithm
        """
        # Simple pattern matching for now
        if formula1 == formula2:
            return True
        
        # Check if they match with simple variable substitution
        # This is a simplified version - proper unification is more complex
        tokens1 = formula1.split()
        tokens2 = formula2.split()
        
        if len(tokens1) != len(tokens2):
            return False
        
        substitutions = {}
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2:
                continue
            elif t1.isalpha() and t1.islower():  # t1 is a variable
                if t1 in substitutions:
                    if substitutions[t1] != t2:
                        return False
                else:
                    substitutions[t1] = t2
            elif t2.isalpha() and t2.islower():  # t2 is a variable
                if t2 in substitutions:
                    if substitutions[t2] != t1:
                        return False
                else:
                    substitutions[t2] = t1
            else:
                return False  # Different constants
        
        return True
    
    def _construct_proof(self, proof_tree: ProofGoal, proof_steps: List[ProofStep], target: str) -> Dict:
        """
        Construct final proof object with complete derivation
        """
        # Extract axioms used
        axioms_used = set()
        for step in proof_steps:
            axioms_used.update(step.axioms_used)
        
        # Build derivation chain
        derivation_steps = []
        for step in reversed(proof_steps):  # Work forward from axioms
            derivation_steps.append(f"{step.rule_name}: {step.justification}")
        
        # Encode final result to Gödel number
        godel_number = self.hybrid_generator.godel_system.encode_axiom(
            f"Proved Formula\nLogical Form: {target}",
            f"Backward_Proof_{int(time.time())}"
        )
        
        proof = {
            'target_formula': target,
            'proved': True,
            'godel_number': godel_number,
            'axioms_used': list(axioms_used),
            'proof_steps': proof_steps,
            'derivation_steps': derivation_steps,
            'proof_depth': proof_tree.depth if proof_tree else 0,
            'total_steps': len(proof_steps),
            'proof_method': 'backward_chaining'
        }
        
        return proof
    
    def _print_proof_summary(self, proof: Dict):
        """
        Print a human-readable proof summary
        """
        print(f"\n📋 PROOF SUMMARY")
        print(f"Target: {proof['target_formula']}")
        print(f"Axioms used: {', '.join(proof['axioms_used'])}")
        print(f"Proof depth: {proof['proof_depth']}")
        print(f"Total steps: {proof['total_steps']}")
        print(f"Gödel number: {proof['godel_number']}")
        
        print(f"\n🔗 DERIVATION CHAIN:")
        for i, step in enumerate(proof['derivation_steps'], 1):
            print(f"  {i}. {step}")
    
    def prove_multiple_goals(self, target_formulas: List[str], max_depth: int = 5) -> Dict[str, Optional[Dict]]:
        """
        Prove multiple goals and return results
        """
        results = {}
        
        print(f"=== Proving {len(target_formulas)} Goals ===\n")
        
        for i, formula in enumerate(target_formulas, 1):
            print(f"\n--- Goal {i}/{len(target_formulas)} ---")
            result = self.prove_goal(formula, max_depth)
            results[formula] = result
        
        # Summary
        successful_proofs = len([r for r in results.values() if r is not None])
        print(f"\n=== SUMMARY ===")
        print(f"Successfully proved: {successful_proofs}/{len(target_formulas)} goals")
        
        return results
    
    def explain_why_unprovable(self, target_formula: str) -> Dict:
        """
        Analyze why a formula couldn't be proved
        """
        print(f"=== Analyzing Why '{target_formula}' Is Unprovable ===\n")
        
        analysis = {
            'target': target_formula,
            'missing_implications': [],
            'missing_facts': [],
            'suggestions': []
        }
        
        # Check if we have any rules that could conclude this
        if target_formula not in self.implications:
            analysis['missing_implications'].append(f"No axiom concludes with '{target_formula}'")
            analysis['suggestions'].append(f"Add axiom: [condition] → {target_formula}")
        
        # Check if it's a direct fact
        if target_formula not in self.facts:
            analysis['missing_facts'].append(f"'{target_formula}' is not a direct fact")
        
        # Look for partial matches
        partial_matches = []
        for fact in self.facts:
            if any(word in fact.lower() for word in target_formula.lower().split()):
                partial_matches.append(fact)
        
        if partial_matches:
            analysis['suggestions'].append(f"Similar facts exist: {partial_matches}")
        
        return analysis

# Usage examples and integration
def test_backward_chaining(hybrid_generator):
    """
    Test backward chaining with student retention scenarios
    """
    print("=== Testing Backward Chaining Prover ===\n")
    
    # Create prover
    prover = BackwardChainingProver(hybrid_generator)
    
    # Test goals - things we want to prove about student retention
    test_goals = [
        "dropout_risk_high = 1",
        "intervention_needed = 1", 
        "student_at_risk = 1",
        "financial_support_needed = 1",
        "academic_support_needed = 1"
    ]
    
    print("Available axioms:")
    for i, axiom in enumerate(hybrid_generator.axiom_formulas[:5], 1):
        print(f"  {i}. {axiom.formula_string}")
    print(f"  ... and {len(hybrid_generator.axiom_formulas)-5} more\n")
    
    # Prove each goal
    results = prover.prove_multiple_goals(test_goals, max_depth=4)
    
    # Analyze unprovable goals
    print(f"\n=== ANALYSIS OF UNPROVABLE GOALS ===")
    for goal, proof in results.items():
        if proof is None:
            analysis = prover.explain_why_unprovable(goal)
            print(f"\n🔍 {goal}:")
            for suggestion in analysis['suggestions']:
                print(f"  💡 {suggestion}")
    
    return results

if __name__ == "__main__":
    print("Backward Chaining Prover - Ready for integration")
    print("Call test_backward_chaining(hybrid_generator) to test")