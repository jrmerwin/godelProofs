# godelProofs

# Axiom Notation Guide for Student Retention Analysis

This guide explains the logical notation used in our axioms, making them ready for conversion to Gödel numbering.

## Basic Logical Operators

| Symbol | Meaning | Example | Natural Language |
|--------|---------|---------|------------------|
| `∧` | AND (conjunction) | `A ∧ B` | Both A and B are true |
| `∨` | OR (disjunction) | `A ∨ B` | Either A or B (or both) is true |
| `¬` | NOT (negation) | `¬A` | A is not true |
| `→` | IMPLIES (implication) | `A → B` | If A then B |
| `↔` | IF AND ONLY IF (biconditional) | `A ↔ B` | A if and only if B |
| `=` | EQUALS | `x = 1` | Variable x equals 1 |
| `>`, `<`, `≥`, `≤` | Comparisons | `x > 0.5` | x is greater than 0.5 |

## Quantifiers (for more complex axioms)

| Symbol | Meaning | Example | Natural Language |
|--------|---------|---------|------------------|
| `∀` | FOR ALL (universal) | `∀x` | For all students x |
| `∃` | EXISTS (existential) | `∃x` | There exists a student x |

## Common Patterns in Our Axioms

### 1. Simple Conditional Rules
```
Pattern: (Condition) → (Outcome)
Example: (Debtor = 1) → (Risk_high = 1)
Meaning: If a student is a debtor, then they are high risk
```

### 2. Compound Conditions
```
Pattern: (Condition1 ∧ Condition2) → (Outcome)
Example: (Debtor = 1 ∧ Tuition_fees_up_to_date = 0) → (Academic_struggle = 1)
Meaning: If a student is both a debtor AND has overdue tuition, then they will struggle academically
```

### 3. Multiple Outcome Possibilities
```
Pattern: (Condition) → (Outcome1 ∨ Outcome2)
Example: (International = 1) → (Support_needs = 1 ∨ Cultural_adjustment = 1)
Meaning: If a student is international, then they need either support or cultural adjustment (or both)
```

### 4. Threshold Rules
```
Pattern: (Variable > threshold) → (Outcome)
Example: (Curricular_units_1st_sem_enrolled - Curricular_units_1st_sem_approved > 3) → (Academic_risk = 1)
Meaning: If the gap between enrolled and approved units exceeds 3, then academic risk is high
```

### 5. Negation Rules
```
Pattern: ¬(Condition) → (Outcome)
Example: ¬(Scholarship_holder = 1) ∧ (Debtor = 1) → (Financial_stress = 1)
Meaning: If NOT a scholarship holder AND is a debtor, then financial stress is high
```

### 6. Biconditional Rules (Equivalences)
```
Pattern: (Condition) ↔ (Outcome)
Example: (Mother_qualification = high ∧ Father_qualification = high) ↔ (Family_academic_support = 1)
Meaning: High parental qualifications if and only if strong family academic support
```

## Variable Types in Our System

### Categorical Variables (Binary: 0 or 1)
- `Debtor`, `International`, `Scholarship_holder`, `Gender`, etc.
- Example: `Debtor = 1` means "is a debtor"

### Categorical Variables (Multiple values)
- `Course`, `Application_mode`, `Nationality`, etc.
- Example: `Course = "Engineering"` or coded as `Course = 1` for Engineering

### Numerical Variables
- `Age_at_enrollment`, `Curricular_units_1st_sem_enrolled`, etc.
- Used with comparison operators: `Age_at_enrollment > 25`

### Derived Variables (Created from combinations)
- `Academic_struggle_ratio = Curricular_units_1st_sem_enrolled / Curricular_units_1st_sem_approved`
- `Financial_stress = (Debtor = 1) ∧ (Tuition_fees_up_to_date = 0)`

## Examples from Student Retention Context

### Financial Risk Axiom
```
Logical Form: (Debtor = 1 ∧ Tuition_fees_up_to_date = 0 ∧ ¬(Scholarship_holder = 1)) → (Retention_risk = 1)

Natural Language: Students who are debtors, have overdue tuition, and don't have scholarships are at high retention risk
```

### Academic Preparation Axiom
```
Logical Form: (Mother_qualification ≥ higher_ed ∨ Father_qualification ≥ higher_ed) → (Academic_support = 1)

Natural Language: If either parent has higher education qualifications, the student has academic support
```

### Performance Warning Axiom
```
Logical Form: (Curricular_units_1st_sem_approved / Curricular_units_1st_sem_enrolled < 0.7) → (Academic_intervention_needed = 1)

Natural Language: If a student approves less than 70% of enrolled units, they need academic intervention
```

## Converting to Gödel Numbers

Each axiom will eventually be assigned a unique Gödel number based on:
1. **Prime factorization** of the logical structure
2. **Variable mappings** (each variable gets a unique prime number)
3. **Operator mappings** (each logical operator gets a unique prime number)

For example, a simple axiom like `(A = 1) → (B = 1)` might become:
- A = prime 2
- → = prime 3  
- B = prime 5
- Resulting in Gödel number: 2³ × 3¹ × 5² (or similar encoding)

## Tips for Reading Generated Axioms

1. **Start with the structure**: Identify the main logical pattern (implication, conjunction, etc.)
2. **Identify the variables**: Look for the actual column names from your dataset
3. **Understand the conditions**: What combination of factors triggers the outcome?
4. **Check domain logic**: Does the relationship make sense for student retention?

This notation system allows us to convert business logic into formal mathematical proofs while maintaining interpretability!
