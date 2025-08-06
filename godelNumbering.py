import re
from typing import Dict, List, Tuple, Any, Set, Optional
from sympy import isprime, nextprime, factorint
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GodelNumberingSystem:
    """
    Robust Gödel numbering system for logical axioms with improved parsing and encoding
    """
    
    def __init__(self):
        # Core logical operators with assigned prime numbers
        self.operator_primes = {
            # Logical connectives
            '∧': 2,    # AND
            '∨': 3,    # OR
            '¬': 5,    # NOT
            '→': 7,    # IMPLIES
            '↔': 11,   # BICONDITIONAL
            '↑': 103,  # NAND
            '↓': 107,  # NOR
            
            # Comparison operators
            '=': 13,   # EQUALS
            '≠': 17,   # NOT EQUALS
            '>': 19,   # GREATER THAN
            '<': 23,   # LESS THAN
            '≥': 29,   # GREATER EQUAL
            '≤': 31,   # LESS EQUAL
            
            # Arithmetic operators
            '+': 37,   # PLUS
            '-': 41,   # MINUS
            '*': 43,   # MULTIPLY
            '/': 47,   # DIVIDE
            '%': 53,   # MODULO
            
            # Structural symbols
            '(': 59,   # LEFT PARENTHESIS
            ')': 61,   # RIGHT PARENTHESIS
            '[': 67,   # LEFT BRACKET
            ']': 71,   # RIGHT BRACKET
            '|': 117,  # PIPE (for absolute values or OR)
            
            # Quote characters (all variations)
            ''': 109,  # FANCY LEFT QUOTE
            ''': 113,  # FANCY RIGHT QUOTE
            '"': 119,  # LEFT DOUBLE QUOTE
            '"': 121,  # RIGHT DOUBLE QUOTE
            "'": 123,  # REGULAR APOSTROPHE
            '"': 127,  # REGULAR DOUBLE QUOTE
            '′': 131,  # Alternative quote character
            '′': 137,  # Another quote variant
            
            # Keywords
            'IF': 73,
            'THEN': 79,
            'ELSE': 83,
            'AND': 89,
            'OR': 97,
            'NOT': 101
        }
        
        # Dynamic prime assignment
        self.variable_primes: Dict[str, int] = {}
        self.constant_primes: Dict[str, int] = {}
        self.next_prime = 131  # Start assigning after reserved primes
        
        # Storage
        self.axiom_encodings: Dict[str, int] = {}
        self.axiom_metadata: Dict[str, Dict] = {}
        
        # Parsing statistics
        self.parsing_stats = {
            'successful_parses': 0,
            'failed_parses': 0,
            'total_variables_discovered': 0,
            'total_constants_discovered': 0
        }
    
    def get_next_prime(self) -> int:
        """Get the next available prime number"""
        prime = self.next_prime
        self.next_prime = nextprime(self.next_prime)
        return prime
    
    def assign_variable_prime(self, variable: str) -> int:
        """Assign a prime number to a variable"""
        if variable not in self.variable_primes:
            self.variable_primes[variable] = self.get_next_prime()
            self.parsing_stats['total_variables_discovered'] += 1
            logger.debug(f"Assigned prime {self.variable_primes[variable]} to variable '{variable}'")
        return self.variable_primes[variable]
    
    def assign_constant_prime(self, constant: str) -> int:
        """Assign a prime number to a constant"""
        if constant not in self.constant_primes:
            self.constant_primes[constant] = self.get_next_prime()
            self.parsing_stats['total_constants_discovered'] += 1
            logger.debug(f"Assigned prime {self.constant_primes[constant]} to constant '{constant}'")
        return self.constant_primes[constant]
    
    def extract_logical_form(self, axiom_text: str) -> Optional[str]:
        """
        Extract logical form from axiom text with multiple fallback strategies
        """
        # Strategy 1: Look for explicit "Logical Form:" marker
        logical_form_patterns = [
            r'Logical\s+Form:\s*(.+?)(?:\n|$)',
            r'Logic:\s*(.+?)(?:\n|$)',
            r'Formula:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in logical_form_patterns:
            match = re.search(pattern, axiom_text, re.MULTILINE | re.IGNORECASE)
            if match:
                logical_form = match.group(1).strip()
                logger.debug(f"Extracted logical form: {logical_form}")
                return logical_form
        
        # Strategy 2: Look for mathematical expressions in the text
        math_pattern = r'[A-Za-z_][A-Za-z0-9_]*\s*[=<>≤≥≠]\s*[A-Za-z0-9_.]+|[A-Za-z_][A-Za-z0-9_]*\s*[→↔∧∨]'
        math_expressions = re.findall(math_pattern, axiom_text)
        
        if math_expressions:
            # Join expressions with AND if multiple found
            logical_form = ' ∧ '.join(math_expressions)
            logger.debug(f"Constructed logical form from expressions: {logical_form}")
            return logical_form
        
        # Strategy 3: Try to extract from natural language patterns
        nl_patterns = [
            (r'if\s+(.+?)\s+then\s+(.+?)(?:\.|$)', r'\1 → \2'),
            (r'(.+?)\s+implies\s+(.+?)(?:\.|$)', r'\1 → \2'),
            (r'when\s+(.+?),?\s+(.+?)(?:\.|$)', r'\1 → \2'),
        ]
        
        for pattern, replacement in nl_patterns:
            match = re.search(pattern, axiom_text, re.IGNORECASE)
            if match:
                logical_form = re.sub(pattern, replacement, axiom_text, flags=re.IGNORECASE)
                logical_form = self.clean_logical_form(logical_form)
                logger.debug(f"Extracted from natural language: {logical_form}")
                return logical_form
        
        logger.warning(f"Could not extract logical form from: {axiom_text[:100]}...")
        return None
    
    def clean_logical_form(self, logical_form: str) -> str:
        """
        Clean and standardize logical form
        """
        # Remove extra whitespace
        logical_form = re.sub(r'\s+', ' ', logical_form.strip())
        
        # Standardize operators
        replacements = {
            ' and ': ' ∧ ',
            ' AND ': ' ∧ ',
            ' or ': ' ∨ ',
            ' OR ': ' ∨ ',
            ' not ': ' ¬',
            ' NOT ': ' ¬',
            ' implies ': ' → ',
            ' IMPLIES ': ' → ',
            ' if and only if ': ' ↔ ',
            ' iff ': ' ↔ ',
            ' != ': ' ≠ ',
            ' >= ': ' ≥ ',
            ' <= ': ' ≤ ',
            ' == ': ' = '
        }
        
        for old, new in replacements.items():
            logical_form = logical_form.replace(old, new)
        
        # Handle probabilistic statements
        logical_form = self.convert_probabilistic_statements(logical_form)
        
        return logical_form
    
    def convert_probabilistic_statements(self, logical_form: str) -> str:
        """
        Convert probabilistic statements to boolean predicates
        """
        # P(X) > threshold → X_likely = 1
        prob_patterns = [
            (r'P\(([^)]+)\)\s*>\s*([\d.]+)', r'\1_likely = 1'),
            (r'P\(([^)]+)\)\s*<\s*([\d.]+)', r'\1_unlikely = 1'),
            (r'P\(([^)]+)\)\s*≥\s*([\d.]+)', r'\1_likely = 1'),
            (r'P\(([^)]+)\)\s*≤\s*([\d.]+)', r'\1_unlikely = 1'),
        ]
        
        for pattern, replacement in prob_patterns:
            logical_form = re.sub(pattern, replacement, logical_form)
        
        # Risk score adjustments
        score_patterns = [
            (r'(\w+)\s*\+=\s*[\d.]+', r'\1_increased = 1'),
            (r'(\w+)\s*-=\s*[\d.]+', r'\1_decreased = 1'),
            (r'(\w+)\s*=\s*"?([^"\s]+)"?', r'\1 = \2'),
        ]
        
        for pattern, replacement in score_patterns:
            logical_form = re.sub(pattern, replacement, logical_form)
        
        return logical_form
    
    def tokenize(self, logical_form: str) -> List[Tuple[str, str]]:
        """
        Tokenize logical form into (token, type) pairs
        Returns list of tuples: (token_value, token_type)
        """

        # Preprocess to handle fancy quotes
        logical_form = logical_form.replace(''', "'").replace(''', "'")
        logical_form = logical_form.replace('"', '"').replace('"', '"')
    
        tokens = []
        
        # Define token patterns with their types - ORDER IS CRITICAL!
        token_patterns = [
            # Multi-character operators (order matters!)
            (r'≥|≤|≠|→|↔|∧|∨|↑|↓', 'OPERATOR'),
            # Numbers MUST come before single characters
            (r'\d+(?:\.\d+)?', 'NUMBER'),
            # Keywords
            (r'\b(?:IF|THEN|ELSE|AND|OR|NOT)\b', 'KEYWORD'),
            # Handle fancy quotes explicitly - try multiple approaches
            (r'[''""‚„]', 'OPERATOR'),  # Direct Unicode characters
            # Single character operators
            (r'[=<>¬+\-*/%()\[\]|]', 'OPERATOR'),
            # Identifiers/variables
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
            # Whitespace (to be ignored)
            (r'\s+', 'WHITESPACE'),
        ]
        
        # Combine all patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern, _ in token_patterns)
        
        pos = 0
        while pos < len(logical_form):
            match = re.match(combined_pattern, logical_form[pos:])
            
            if match:
                token_value = match.group()
                
                # Determine token type
                token_type = 'UNKNOWN'
                for i, (pattern, t_type) in enumerate(token_patterns):
                    if match.group(i + 1):  # Check which group matched
                        token_type = t_type
                        break
                
                # Skip whitespace tokens
                if token_type != 'WHITESPACE':
                    tokens.append((token_value, token_type))
                    logger.debug(f"Token: '{token_value}' ({token_type})")
                
                pos += len(token_value)
            else:
                # Handle unrecognized character
                logger.warning(f"Unrecognized character: '{logical_form[pos]}'")
                pos += 1
        
        return tokens
    
    def compute_godel_number(self, tokens: List[Tuple[str, str]]) -> int:
        """
        Compute Gödel number using improved encoding scheme
        Uses: G = ∏(p_i^encode(token_i)) where p_i is the i-th prime
        """
        if not tokens:
            return 1
        
        godel_number = 1
        position_prime = 2  # Start with first prime for position encoding
        
        for token_value, token_type in tokens:
            # Get encoding for this token
            token_encoding = self.encode_token(token_value, token_type)
            
            # Apply position-based encoding: position_prime^token_encoding
            try:
                # Prevent overflow by limiting exponents
                if token_encoding > 100:
                    token_encoding = token_encoding % 100 + 1
                
                contribution = position_prime ** token_encoding
                godel_number *= contribution
                
                # Prevent excessive growth
                if godel_number > 10**100:
                    godel_number = godel_number % (10**50) + 1
                
                logger.debug(f"Token '{token_value}': {position_prime}^{token_encoding} = {contribution}")
                
            except OverflowError:
                logger.warning(f"Overflow in Gödel number computation for token '{token_value}'")
                godel_number = godel_number % (10**50) + 1
            
            # Move to next prime
            position_prime = nextprime(position_prime)
        
        return godel_number
    
    def encode_token(self, token_value: str, token_type: str) -> int:
        """
        Encode a single token based on its type and value
        """
        # Handle numbers FIRST - this is the key fix
        if token_type == 'NUMBER':
            try:
                if '.' in token_value:
                    # Float number - scale up to avoid decimals
                    num_value = int(float(token_value) * 10)
                else:
                    # Integer number
                    num_value = int(token_value)
                return max(1, num_value % 1000 + 200)  # Offset by 200, limit range
            except ValueError:
                logger.warning(f"Could not parse number: {token_value}")
                return 211  # Default for unparseable numbers
        
        elif token_type == 'OPERATOR' or token_type == 'KEYWORD':
            # Check predefined operator/keyword primes
            if token_value in self.operator_primes:
                return self.operator_primes[token_value]
            else:
                # Assign new prime for unknown operator
                logger.warning(f"Unknown operator/keyword: {token_value}")
                return self.get_next_prime()
        
        elif token_type == 'IDENTIFIER':
            # Determine if it's a variable or constant
            if self.is_likely_variable(token_value):
                return self.assign_variable_prime(token_value)
            else:
                return self.assign_constant_prime(token_value)
        
        else:
            # Unknown token type
            logger.warning(f"Unknown token type: {token_type} for '{token_value}'")
            return self.get_next_prime()
    
    def is_likely_variable(self, identifier: str) -> bool:
        """
        Heuristic to determine if an identifier is likely a variable vs constant
        """
        # Variables often have these characteristics
        variable_indicators = [
            identifier.islower(),  # lowercase
            '_' in identifier,     # contains underscore
            len(identifier) > 2,   # reasonably long
            any(char.isdigit() for char in identifier),  # contains digits
        ]
        
        # Constants often have these characteristics  
        constant_indicators = [
            identifier.isupper(),  # all uppercase
            identifier.isdigit(),  # pure number (already handled)
            identifier in ['True', 'False', 'High', 'Low', 'Yes', 'No'],
        ]
        
        # Score-based decision
        var_score = sum(variable_indicators)
        const_score = sum(constant_indicators)
        
        return var_score > const_score
    
    def encode_axiom(self, axiom_text: str, axiom_name: str = None) -> Optional[int]:
        """
        Main method to encode an axiom into its Gödel number
        """
        try:
            logger.info(f"Encoding axiom: {axiom_name or 'Unnamed'}")
            
            # Step 1: Extract logical form
            logical_form = self.extract_logical_form(axiom_text)
            if not logical_form:
                logger.error(f"Could not extract logical form from axiom")
                self.parsing_stats['failed_parses'] += 1
                return None
            
            # Step 2: Clean logical form
            cleaned_form = self.clean_logical_form(logical_form)
            
            # Step 3: Tokenize
            tokens = self.tokenize(cleaned_form)
            if not tokens:
                logger.error(f"No tokens found in logical form: {cleaned_form}")
                self.parsing_stats['failed_parses'] += 1
                return None
            
            # Step 4: Compute Gödel number
            godel_number = self.compute_godel_number(tokens)
            
            # Step 5: Store results
            if axiom_name:
                self.axiom_encodings[axiom_name] = godel_number
                self.axiom_metadata[axiom_name] = {
                    'original_text': axiom_text[:200],
                    'logical_form': logical_form,
                    'cleaned_form': cleaned_form,
                    'tokens': tokens,
                    'godel_number': godel_number
                }
            
            self.parsing_stats['successful_parses'] += 1
            logger.info(f"Successfully encoded axiom with Gödel number: {godel_number}")
            
            return godel_number
            
        except Exception as e:
            logger.error(f"Error encoding axiom {axiom_name}: {e}")
            self.parsing_stats['failed_parses'] += 1
            return None
    
    def get_axiom_summary(self) -> pd.DataFrame:
        """
        Return comprehensive summary of encoded axioms
        """
        if not self.axiom_encodings:
            return pd.DataFrame(columns=[
                'Axiom_Name', 'Godel_Number', 'Token_Count', 
                'Variables_Used', 'Operators_Used', 'Success'
            ])
        
        summary_data = []
        for name, godel_num in self.axiom_encodings.items():
            metadata = self.axiom_metadata.get(name, {})
            tokens = metadata.get('tokens', [])
            
            # Analyze tokens
            variables = [t[0] for t in tokens if t[1] == 'IDENTIFIER' and self.is_likely_variable(t[0])]
            operators = [t[0] for t in tokens if t[1] in ['OPERATOR', 'KEYWORD']]
            
            summary_data.append({
                'Axiom_Name': name,
                'Godel_Number': godel_num,
                'Token_Count': len(tokens),
                'Variables_Used': len(set(variables)),
                'Operators_Used': len(set(operators)),
                'Success': True,
                'Logical_Form': metadata.get('cleaned_form', 'N/A')
            })
        
        return pd.DataFrame(summary_data)
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """
        Return parsing statistics and system state
        """
        return {
            'parsing_stats': self.parsing_stats.copy(),
            'total_axioms_encoded': len(self.axiom_encodings),
            'unique_variables': len(self.variable_primes),
            'unique_constants': len(self.constant_primes),
            'next_available_prime': self.next_prime,
            'variable_prime_mappings': self.variable_primes.copy(),
            'constant_prime_mappings': self.constant_primes.copy()
        }