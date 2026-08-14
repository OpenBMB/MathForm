"""Per-sample pipeline mode."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from api_client import APIClient
from compiler import LeanCompiler
from config import Config
from generator import CodeGenerator
from retrieval import RetrievalService
from semantic_judge import SemanticJudge
from utils import FileIO, Logger, get_statement


class SamplePipeline:
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(config.io.log_file)
        self.file_io = FileIO()
        
        self.api_client = APIClient(config.api, self.logger)
        self.retrieval_service = RetrievalService(
            self.api_client, config.api, config.pipeline, self.logger
        )
        self.compiler = LeanCompiler(config.api, self.logger)
        self.semantic_judge = SemanticJudge(self.api_client, self.logger)
        self.generator = CodeGenerator(self.api_client, config.pipeline, self.logger)
    
    def _init_state(self, item: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
        statement = get_statement(item).strip()
        if not statement:
            self.logger.log(f"[Item {idx + 1}/{total}] Missing statement, skipping.")
            return {
                "informal_statement": get_statement(item).strip(),
                "statement": "",
                "status": "failed",
                "reason": "missing statement",
                "retrieval_history": [],
                "history_feedback": ["Missing statement"],
                "last_prompt": "",
                "last_response": "",
                "lean_code": "",
            }
        return {
            "original": item,
            "statement": statement,
            "status": "pending",
            "retrieval_history": [],
            "history_feedback": [],
            "last_bad_code": "",
            "last_compile_error": "",
            "last_semantic_feedback": "",
            "last_prompt": "",
            "last_response": "",
            "last_reasoning_content": "",
            "lean_code": "",
            "compile_passed": False,
            "iterations": 0,
        }
    
    def _process_single(self, item: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
        statement = get_statement(item).strip()
        item_prefix = f"[Item {idx + 1}/{total}]"
        
        if not statement:
            self.logger.log(f"{item_prefix} Missing statement, skipping.")
            return {
                "status": "failed",
                "reason": "missing statement",
                "informal_statement": get_statement(item).strip(),
            }
        
        retrieval_history: List[Dict[str, Any]] = []
        history_feedback: List[str] = []
        last_bad_code = ""
        last_compile_error = ""
        last_semantic_feedback = ""
        last_prompt = ""
        last_response = ""
        last_reasoning_content = ""
        final_code = ""
        
        for iteration in range(1, self.config.pipeline.max_iterations + 1):
            self.logger.log(f"{item_prefix} Iteration {iteration}/{self.config.pipeline.max_iterations}")
            
            queries = self.retrieval_service.generate_queries(
                statement,
                retrieval_history,
                compile_error=last_compile_error,
                semantic_feedback=last_semantic_feedback,
            )
            retrieval_entries = self.retrieval_service.run_retrieval(queries)
            self.retrieval_service.append_retrieval_history(retrieval_history, retrieval_entries)
            retrieval_context = self.retrieval_service.format_retrieval_context(retrieval_history)
            
            gen_result = self.generator.generate_code(
                statement=statement,
                retrieval_context=retrieval_context,
                last_bad_code=last_bad_code,
                compile_error=last_compile_error,
                semantic_feedback=last_semantic_feedback,
            )
            
            if not gen_result:
                history_feedback.append("Base model failed to return a response")
                continue
            
            last_prompt = gen_result["prompt"]
            last_response = gen_result["response"]
            last_reasoning_content = gen_result.get("reasoning_content", "")
            lean_code = gen_result["lean_code"]
            
            if not lean_code:
                history_feedback.append("Failed to extract Lean code")
                continue
            
            if gen_result["has_proof_steps"]:
                history_feedback.append("Generated code includes proof steps")
                last_bad_code = lean_code
                last_compile_error = "Proof steps detected; expected no proof"
                last_semantic_feedback = ""
                continue
            
            final_code = lean_code
            
            compile_ok, compile_error = self.compiler.verify_compilation(lean_code)
            if not compile_ok:
                self.logger.log(f"{item_prefix} Compilation failed: {compile_error[:120]}")
                history_feedback.append(f"Compilation failed: {compile_error}")
                last_bad_code = lean_code
                last_compile_error = compile_error
                last_semantic_feedback = ""
                continue
            
            semantic_ok, semantic_feedback, _judge_queries = self.semantic_judge.judge_semantic(
                statement, lean_code
            )
            if not semantic_ok:
                self.logger.log(f"{item_prefix} Semantic check failed: {semantic_feedback[:120]}")
                history_feedback.append(f"Semantic failed: {semantic_feedback}")
                last_bad_code = lean_code
                last_compile_error = ""
                last_semantic_feedback = semantic_feedback
                
                # Follow-up retrieval guided by judge feedback
                follow_queries = self.retrieval_service.generate_queries(
                    statement,
                    retrieval_history,
                    compile_error=last_compile_error,
                    semantic_feedback=last_semantic_feedback,
                )
                if follow_queries:
                    follow_retrievals = self.retrieval_service.run_retrieval(follow_queries)
                    self.retrieval_service.append_retrieval_history(retrieval_history, follow_retrievals)
                continue
            
            record = {
                "informal_statement": get_statement(item).strip(),
                "status": "success",
                "iterations": iteration,
                "lean_code": lean_code,
                "reasoning_content": last_reasoning_content,
                "history_feedback": history_feedback,
                "retrieval_history": retrieval_history,
                "sharegpt": {
                    "messages": [
                        {"role": "user", "content": last_prompt},
                        {"role": "assistant", "content": last_response},
                    ]
                },
            }
            return record
        
        return {
            "informal_statement": get_statement(item).strip(),
            "status": "failed",
            "iterations": self.config.pipeline.max_iterations,
            "lean_code": final_code,
            "history_feedback": history_feedback,
            "retrieval_history": retrieval_history,
            "last_prompt": last_prompt,
            "last_response": last_response,
            "last_reasoning_content": last_reasoning_content,
        }
    
    def run(self) -> None:
        data = self.file_io.load_jsonl(self.config.io.input_file)
        total = len(data)
        self.logger.log(f"Loaded {total} items from {self.config.io.input_file}")
        self.logger.log(f"Output directory: {self.config.io.run_dir}")
        
        states = [self._init_state(item, idx, total) for idx, item in enumerate(data)]
        
        self.logger.log(f"Running pipelined processing with max_workers={self.config.pipeline.max_workers}")
        
        total_items = len(states)
        completed_items = sum(1 for s in states if s.get("status") != "pending")
        
        with ThreadPoolExecutor(max_workers=self.config.pipeline.max_workers) as executor:
            for iteration in range(1, self.config.pipeline.max_iterations + 1):
                pending_states = [s for s in states if s.get("status") == "pending"]
                if not pending_states:
                    break
                
                pending_count = len(pending_states)
                self.logger.log(
                    f"Iteration {iteration}/{self.config.pipeline.max_iterations}: "
                    f"processing {pending_count} pending items "
                    f"(completed: {completed_items}/{total_items})"
                )
                
                for s in pending_states:
                    s["iterations"] = iteration
                
                futures = {
                    executor.submit(self._process_single, s["original"], idx, total): (idx, s)
                    for idx, s in enumerate(states)
                    if s.get("status") == "pending"
                }
                
                completed_count = 0
                success_count = 0
                fail_count = 0
                total_pending = len(futures)
                
                with tqdm(
                    total=total_pending,
                    desc=f"Iter {iteration}/{self.config.pipeline.max_iterations}",
                    unit="item",
                    ncols=120,
                    disable=False
                ) as pbar:
                    for future in as_completed(futures):
                        idx, state = futures[future]
                        try:
                            result = future.result()
                            if result.get("status") == "success":
                                state.update(result)
                                state["status"] = "success"
                                state["success_record"] = result
                                success_count += 1
                            elif result.get("status") == "failed":
                                state.update(result)
                                state["status"] = "failed"
                                fail_count += 1
                        except Exception as exc:
                            self.logger.log(f"Error processing item {idx}: {exc}")
                            state["status"] = "failed"
                            state["error"] = str(exc)
                            fail_count += 1
                        
                        completed_count += 1
                        pbar.update(1)
                        pbar.set_description(
                            f"Iter {iteration}/{self.config.pipeline.max_iterations} "
                            f"[{completed_count}/{total_pending}]"
                        )
                        pbar.set_postfix({
                            "success": success_count,
                            "failed": fail_count,
                            "rate": f"{success_count/completed_count:.1%}" if completed_count > 0 else "0%"
                        })
                
                completed_items += total_pending
                
                for s in states:
                    if (s.get("status") == "success" and
                        not s.get("success_written") and
                        "success_record" in s):
                        self.file_io.append_jsonl(s["success_record"], self.config.io.success_file)
                        s["success_written"] = True
        
        for s in states:
            if s.get("status") != "success":
                record = {
                    "informal_statement": get_statement(s.get("original", {})).strip(),
                    "status": "failed",
                    "iterations": s.get("iterations", self.config.pipeline.max_iterations),
                    "lean_code": s.get("lean_code", ""),
                    "history_feedback": s.get("history_feedback", []),
                    "retrieval_history": s.get("retrieval_history", []),
                    "last_prompt": s.get("last_prompt", ""),
                    "last_response": s.get("last_response", ""),
                }
                self.file_io.append_jsonl(record, self.config.io.failed_file)
