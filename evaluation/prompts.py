"""Prompt templates and builders for the inference and judge stages."""

INFERENCE_PROMPT_TEMPLATES = {
    "stepfun": "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n{informal_problem}\n\nYour code should start with:\n```Lean4\nimport Mathlib\n```\n",
    "kimina": "You are an expert in mathematics and Lean 4. Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n{informal_problem}",
    "formalizer": "Please convert the following informal math problem to a formal one in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n{informal_problem}",
    "general": "Please convert the following informal math problem to a formal one in Lean 4 with a header. Do not provide the proof, end with `by sorry`. Use the following theorem names: my_favorite_theorem.\n\n{informal_problem}",
    "reform": "Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency.\n{informal_problem}",
    "goedelv2": "Please autoformalize the following natural language problem statement in Lean 4. Use the following theorem name: my_favorite_theorem\nThe natural language statement is: \n{informal_problem}Think before you provide the lean statement.",
    "mathesis": "[Question]:\n{informal_problem}\n\nYou are an expert in formal mathematics. Your task is to convert the above [question] to lean 4\ntheorems by completing the following lean 4 code:\n\nlean4\nimport Mathlib\nimport Aesop\nset-option maxHeartbeats 0\nset-option pp.numericTypes true\nset-option pp.coercions true\nset-option pp.letVarTypes true\nset-option pp.structureInstanceTypes true\nset-option pp.instanceTypes true\nset-option pp.mvars.withType true\nset-option pp.coercions true\nset-option pp.funBinderTypes true\nset-option pp.piBinderTypes true\nopen BigOperators Real Nat Topology Rat\n\n/-\n{informal_problem}\n-/",
}
DEFAULT_INFER_PROMPT_TEMPLATE = "formalizer"

JUDGE_PROMPT_TEMPLATE = """
# Lean 4 Formalization Semantic Consistency Check

**Role:**
Act as an expert in Lean 4 formal verification and mathematical logic. Your task is to perform a rigorous semantic consistency review, comparing a Natural Language (NL) mathematical statement against its corresponding Lean 4 formalization.

**Objective:**
Determine if the provided Lean 4 code is a faithful, completely accurate, and logically equivalent translation of the Natural Language statement. Only focus on the formalization itself, do not discuss any proof process.

**Analysis Steps:**
Before generating the output, analyze the input pairs based on the following criteria:

1. **Deconstruction of the Natural Language Statement:**
   * Identify all key mathematical objects, definitions, and properties.
   * Map out the logical flow (antecedents, consequents, quantifiers).
   * Identify the implicit domain of discourse.

2. **Analysis of the Lean 4 Code Structure:**
   * Verify type hierarchy compliance (classes vs. structures).
   * Check variable declarations and hypothesis scope.
   * Ensure standard library usage matches the mathematical intent.

3. **Semantic Mapping and Gap Analysis:**
   * **Bi-directional Fidelity:** Ensure every constraint in the NL maps to the code, and the code adds no unintended constraints.
   * **Quantifier Precision:** Rigorously check the order and dependency of `for all` vs `there exists`.
   * **Condition Strength:** Ensure predicates are neither strictly stronger nor strictly weaker than required.

**Output Format Requirements:**
Your output must be **exactly** two XML tags in this order, with no other text before, between, or after them. Do not use code fences or markdown.

1. <comments></comments>
   * Provide your detailed evaluation and reasoning.
   * If inconsistent, describe the specific semantic mismatch.
2. <result></result>
   * Output exactly one word: `correct` or `incorrect`.

**Strict Output Template:**
<comments>...</comments>
<result>...</result>

---

**Input Natural Language Statement:**
{mathematical_statement}

**Input Lean 4 Code:**
{autoformalization_placeholder}
"""


def build_inference_prompt(informal_problem: str, template_name: str) -> str:
    """Build the prompt used to generate Lean code from an informal problem."""
    template = INFERENCE_PROMPT_TEMPLATES.get(template_name, INFERENCE_PROMPT_TEMPLATES[DEFAULT_INFER_PROMPT_TEMPLATE])
    return template.replace("{informal_problem}", informal_problem)


def build_prompt(mathematical_text: str, lean_code: str) -> str:
    """Build the prompt used by the judge model."""
    prompt = JUDGE_PROMPT_TEMPLATE.replace("{mathematical_statement}", mathematical_text)
    prompt = prompt.replace("{autoformalization_placeholder}", lean_code)
    return prompt
