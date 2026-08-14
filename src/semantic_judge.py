"""Semantic consistency check between statements and Lean code."""

import re
from typing import List, Optional, Tuple

from api_client import APIClient
from utils import Logger, TextProcessor


class SemanticJudge:
    
    def __init__(
        self,
        api_client: APIClient,
        logger: Optional[Logger] = None,
    ):
        self.api_client = api_client
        self.logger = logger or Logger()
    
    def _build_judge_prompt(self, statement: str, lean_code: str) -> str:
        return f"""# Lean 4 Formalization Semantic Consistency Check

**Role:**
Act as an expert in Lean 4 formal verification and mathematical logic. Your task is to perform a rigorous semantic consistency review, comparing a Natural Language (NL) mathematical statement against its corresponding Lean 4 formalization.

**Objective:**
Determine if the provided Lean 4 code is a faithful, completely accurate, and logically equivalent translation of the Natural Language statement.

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
{statement}

**Input Lean 4 Code:**
{lean_code}
"""
    
    def _parse_judge_response(self, content: Optional[str]) -> Tuple[bool, str, List[str]]:
        if not content:
            return False, "", []
        comments = ""
        result_text = ""
        queries_text = ""
        comments_match = re.search(r"<comments>(.*?)</comments>", content, re.DOTALL | re.IGNORECASE)
        if comments_match:
            comments = comments_match.group(1).strip()
        result_match = re.search(r"<result>\s*(.+?)\s*</result>", content, re.DOTALL | re.IGNORECASE)
        if result_match:
            result_text = re.sub(r"\s+", "", result_match.group(1)).lower()
        queries_match = re.search(r"<queries>(.*?)</queries>", content, re.DOTALL | re.IGNORECASE)
        if queries_match:
            queries_text = queries_match.group(1).strip()
        queries = TextProcessor.parse_query_list(queries_text)
        return result_text == "correct", comments, queries
    
    def judge_semantic(
        self,
        statement: str,
        lean_code: str,
    ) -> Tuple[bool, str, List[str]]:
        """Judge semantic consistency.
        
        Returns:
            (is_correct, feedback, additional_queries)
        """
        prompt = self._build_judge_prompt(statement, lean_code)
        response = self.api_client.call_judge_model(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=16384,
        )
        return self._parse_judge_response(response)
