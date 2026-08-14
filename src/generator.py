"""Lean code generation: prompt construction and model calls."""

from typing import Any, Dict, List, Optional

from api_client import APIClient
from config import PipelineConfig
from utils import Logger, TextProcessor


class CodeGenerator:
    
    def __init__(
        self,
        api_client: APIClient,
        pipeline_config: PipelineConfig,
        logger: Optional[Logger] = None,
    ):
        self.api_client = api_client
        self.pipeline_config = pipeline_config
        self.logger = logger or Logger()
    
    def build_generation_prompt(
        self,
        statement: str,
        retrieval_context: str,
        last_bad_code: str = "",
        compile_error: str = "",
        semantic_feedback: str = "",
    ) -> str:
        prompt = f"""You are an expert in Lean 4 theorem proving and the Mathlib mathematical library.

Your task is to formalize the given mathematical statement into correct Lean 4 code using Mathlib.

FORMALIZATION REQUIREMENTS:
1. Use proper Lean 4 syntax and Mathlib conventions
2. Include ALL necessary headers
3. Define appropriate variables and assumptions
4. The theorem statement must be mathematically correct and equivalent to the original
5. This task is only about automatic formalization. Do NOT output any proof steps, tactics, or reasoning.
6. Retrieved information may be helpful; identify what is truly relevant and use it as needed.
7. Only generate the translation. Do not try to solve or prove the problem."""

        if retrieval_context:
            prompt += f"""

RETRIEVED MATHLIB INFORMATION:
{retrieval_context}"""

        if last_bad_code and compile_error:
            prompt += f"""

PREVIOUS COMPILATION FAILURE - FIX REQUIRED:
Failed code:
{last_bad_code}

Compiler error:
{compile_error}

Please analyze the error and provide corrected code."""

        if last_bad_code and semantic_feedback:
            prompt += f"""

SEMANTIC CONSISTENCY ISSUE - FIX REQUIRED:
Previous code:
{last_bad_code}

Semantic feedback:
{semantic_feedback}

Please revise the formalization to match the mathematical meaning."""

        prompt += f"""

STATEMENT TO FORMALIZE:
{statement}"""

        return prompt
    
    def build_base_model_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": prompt},
        ]
    
    def generate_code(
        self,
        statement: str,
        retrieval_context: str = "",
        last_bad_code: str = "",
        compile_error: str = "",
        semantic_feedback: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Generate Lean code.
        
        Returns:
            {
                "prompt": str,
                "response": str,
                "reasoning_content": str,
                "lean_code": str,
                "has_proof_steps": bool,
            } or None if failed
        """
        prompt = self.build_generation_prompt(
            statement=statement,
            retrieval_context=retrieval_context,
            last_bad_code=last_bad_code,
            compile_error=compile_error,
            semantic_feedback=semantic_feedback,
        )
        
        response = self.api_client.call_base_model(
            self.build_base_model_messages(prompt),
            temperature=self.pipeline_config.temperature,
            max_tokens=16384,
        )
        
        if not response:
            return None
        
        content = response.get("content", "")
        reasoning_content = response.get("reasoning_content", "")
        lean_code = TextProcessor.extract_lean_code(content)
        has_proof_steps = TextProcessor.has_proof_steps(lean_code) if lean_code else False
        
        return {
            "prompt": prompt,
            "response": content,
            "reasoning_content": reasoning_content,
            "lean_code": lean_code,
            "has_proof_steps": has_proof_steps,
        }
