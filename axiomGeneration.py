import pandas as pd
import json
import os
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMAxiomGenerator:
    def __init__(self, provider: str = "anthropic", api_key: Optional[str] = None):
        """
        Initialize the axiom generator with specified LLM provider
        
        Supported providers: 'anthropic', 'openai', 'gemini'
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not found. Set {provider.upper()}_API_KEY environment variable or pass api_key parameter")
    
    def prepare_data_summary(self, df: pd.DataFrame, corr_df: pd.DataFrame, cross_tab_df: pd.DataFrame) -> str:
        """Prepare a comprehensive data summary for the LLM"""
        
        summary = f"""
**DATASET OVERVIEW:**
- Total records: {len(df)}
- Total features: {len(df.columns)}
- Target variable: {df.columns[-1] if 'target' in df.columns[-1].lower() else 'retention outcome'}

**SAMPLE RECORDS:**
{df.head(3).to_string()}

**FEATURE SUMMARY:**
{df.describe(include='all').to_string()}

**KEY CORRELATIONS:**
{corr_df.to_string()}

**CROSS-TABULATIONS:**
{cross_tab_df.to_string()}

**FEATURE TYPES:**
Categorical: {list(df.select_dtypes(include=['object', 'category']).columns)}
Numerical: {list(df.select_dtypes(include=['int64', 'float64']).columns)}
"""
        return summary

    def create_axiom_prompt(self, data_summary: str) -> str:
        """Create the complete prompt for axiom generation"""
        
        prompt = f"""You are tasked with creating formal axioms for student retention analysis based on the provided datasets. Your goal is to identify logical rules that capture causal relationships, business rules, and domain knowledge about student success.

**CONTEXT & OBJECTIVE:**
We're building a formal logical system to understand student retention patterns. Axioms should represent fundamental truths about how student characteristics, behaviors, and outcomes relate to each other and ultimately to retention.

**DATA ANALYSIS:**
{data_summary}

**AXIOM REQUIREMENTS:**
Generate 15-25 axioms across these categories:

1. **Risk Factor Axioms** (4-6 axioms)
   - Financial stress indicators (debtor, tuition status, scholarship)
   - Academic preparation deficits
   - Support system gaps

2. **Performance Prediction Axioms** (4-6 axioms)  
   - Early warning signs from 1st semester data
   - Academic struggle patterns
   - Engagement vs. outcome relationships

3. **Hierarchical/Conditional Axioms** (3-5 axioms)
   - IF-THEN rules with multiple conditions
   - Interaction effects between risk factors
   - Threshold-based rules

4. **Causal Chain Axioms** (3-5 axioms)
   - Multi-step relationships (A → B → C → retention)
   - Compounding effects
   - Protective factors

5. **Domain Knowledge Axioms** (2-4 axioms)
   - Known retention theory principles
   - Institutional policy implications

**FORMAT FOR EACH AXIOM:**
```
Axiom [N]: [Descriptive Name]
Logical Form: [Formal logical statement using variable names]
Natural Language: [Plain English explanation]
Justification: [Why this relationship should hold based on data/theory]
Variables Used: [List the specific columns involved]
```

**GUIDELINES:**
- Use actual variable names from the datasets
- Make axioms specific enough to be testable
- Include both positive (retention-promoting) and negative (risk) axioms
- Consider non-linear relationships and thresholds
- Balance obvious relationships with potentially novel insights
- Ensure axioms can be converted to mathematical expressions

**EXAMPLE:**
```
Axiom 1: Financial Stress Cascade
Logical Form: (Debtor = 1 ∧ Tuition_fees_up_to_date = 0) → Academic_struggle_high = 1
Natural Language: Students who are debtors AND have overdue tuition will likely experience high academic struggle
Justification: Financial stress directly impacts academic focus and resource availability
Variables Used: Debtor, Tuition_fees_up_to_date, [derived academic struggle measure]
```

Analyze the provided data first, then generate axioms that capture both obvious and subtle patterns you observe.

please provide 20-25 axioms representing each of the 5 types
"""
        return prompt

    def call_anthropic(self, prompt: str) -> str:
        """Call Anthropic's Claude API"""
        try:
            import anthropic
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
            # Fallback to requests if anthropic package not available
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

    def call_openai(self, prompt: str) -> str:
        """Call OpenAI's GPT API"""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.1
        )
        return response.choices[0].message.content

    def call_gemini(self, prompt: str) -> str:
        """Call Google's Gemini API"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text

    def generate_axioms(self, df: pd.DataFrame, corr_df: pd.DataFrame, cross_tab_df: pd.DataFrame) -> str:
        """Generate axioms using the specified LLM provider"""
        
        # Prepare data summary
        data_summary = self.prepare_data_summary(df, corr_df, cross_tab_df)
        
        # Create prompt
        prompt = self.create_axiom_prompt(data_summary)
        
        # Call appropriate LLM
        if self.provider == "anthropic":
            return self.call_anthropic(prompt)
        elif self.provider == "openai":
            return self.call_openai(prompt)
        elif self.provider == "gemini":
            return self.call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def save_axioms(self, axioms: str, filename: str = "generated_axioms.txt"):
        """Save the generated axioms to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(axioms)
        print(f"Axioms saved to {filename}")

# Example usage
def main():
    """Example of how to use the axiom generator"""
    
    # Initialize the generator (change provider as needed)
    generator = LLMAxiomGenerator(provider="anthropic")  # or "openai" or "gemini"
    
    # Load your dataframes (replace with your actual loading code)
    # df = pd.read_csv("your_dataset.csv")
    # corr_df = df.corr()
    # cross_tab_df = pd.crosstab(df['some_feature'], df['target'])
    
    # Generate axioms
    # axioms = generator.generate_axioms(df, corr_df, cross_tab_df)
    
    # Save results
    # generator.save_axioms(axioms)
    # print("Axiom generation complete!")
    
    print("Axiom generator initialized. Use generate_axioms() method with your dataframes.")

if __name__ == "__main__":
    main()