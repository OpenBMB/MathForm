"""Staged batch pipeline: generate all -> compile all -> semantic-check all -> retry failed."""

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


class BatchPipeline:
    """Staged batch pipeline.

    1. Generation: build queries, retrieve, and generate code for all samples
    2. Compilation: compile-check every generated candidate
    3. Semantic check: judge every candidate that compiled
    4. Iterate: failed samples repeat the loop with accumulated feedback
    """
    
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
    
    def _init_state(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        statement = get_statement(item).strip()
        return {
            "idx": idx,
            "original": item,
            "statement": statement,
            "status": "pending" if statement else "failed",
            "phase": "init",  # init -> generated -> compiled -> semantic_passed/failed
            "retrieval_history": [],
            "history_feedback": [] if statement else ["Missing statement"],
            "pending_queries": [],
            "last_bad_code": "",
            "last_compile_error": "",
            "last_semantic_feedback": "",
            "last_prompt": "",
            "last_response": "",
            "last_reasoning_content": "",
            "lean_code": "",
            "compile_passed": False,
            "semantic_passed": False,
            "iterations": 0,
        }
    
    def _generate_queries_single(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["status"] != "pending" or not state["statement"]:
            return state
        queries = self.retrieval_service.generate_queries(
            state["statement"],
            state["retrieval_history"],
            compile_error=state.get("last_compile_error", ""),
            semantic_feedback=state.get("last_semantic_feedback", ""),
        )
        state["pending_queries"] = queries
        return state

    def _generate_code_single(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["status"] != "pending" or not state["statement"]:
            return state

        retrieval_context = self.retrieval_service.format_retrieval_context(
            state["retrieval_history"]
        )

        gen_result = self.generator.generate_code(
            statement=state["statement"],
            retrieval_context=retrieval_context,
            last_bad_code=state["last_bad_code"],
            compile_error=state["last_compile_error"],
            semantic_feedback=state["last_semantic_feedback"],
        )

        if not gen_result:
            state["history_feedback"].append("Base model failed to return a response")
            return state

        state["last_prompt"] = gen_result["prompt"]
        state["last_response"] = gen_result["response"]
        state["last_reasoning_content"] = gen_result.get("reasoning_content", "")
        lean_code = gen_result["lean_code"]

        if not lean_code:
            state["history_feedback"].append("Failed to extract Lean code")
            return state

        if gen_result["has_proof_steps"]:
            state["history_feedback"].append("Generated code includes proof steps")
            state["last_bad_code"] = lean_code
            state["last_compile_error"] = "Proof steps detected; expected no proof"
            state["last_semantic_feedback"] = ""
            return state

        state["lean_code"] = lean_code
        state["phase"] = "generated"
        return state
    
    def batch_generate(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch generation: build queries -> batch retrieval -> generate code."""
        pending = [s for s in states if s["status"] == "pending" and s["statement"]]
        if not pending:
            return states

        self.logger.log(f"[Generation] Processing {len(pending)} samples...")

        # Step 1: build retrieval queries for all samples
        planner_workers = self.config.pipeline.retrieval_planner_workers or self.config.pipeline.max_workers
        with ThreadPoolExecutor(max_workers=planner_workers) as executor:
            futures = {executor.submit(self._generate_queries_single, s): s for s in pending}

            with tqdm(total=len(futures), desc="Gen Queries", unit="sample", ncols=120) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        state = futures[future]
                        state["history_feedback"].append(f"Query generation error: {exc}")
                        self.logger.log(f"Query generation error for item {state['idx']}: {exc}")
                    pbar.update(1)

        # Step 2: run batch retrieval over all aggregated queries
        all_queries: List[str] = []
        for s in pending:
            all_queries.extend(s.get("pending_queries", []))

        if all_queries:
            self.logger.log(
                f"[Retrieval] Running batch search for {len(all_queries)} queries "
                f"(from {len(pending)} samples; generation progress is per-sample)"
            )
            mapping = self.retrieval_service.run_retrieval_batch(all_queries, batch_size=32)

            for s in pending:
                queries = s.get("pending_queries", [])
                if not queries:
                    continue
                entries = self.retrieval_service.build_entries_from_mapping(queries, mapping)
                self.retrieval_service.append_retrieval_history(s["retrieval_history"], entries)

        # Step 3: generate code
        base_workers = self.config.pipeline.base_model_workers or self.config.pipeline.max_workers
        with ThreadPoolExecutor(max_workers=base_workers) as executor:
            futures = {executor.submit(self._generate_code_single, s): s for s in pending}

            with tqdm(total=len(futures), desc="Generating", unit="sample", ncols=120) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        state = futures[future]
                        state["history_feedback"].append(f"Generation error: {exc}")
                        self.logger.log(f"Generation error for item {state['idx']}: {exc}")
                    pbar.update(1)

        generated_count = sum(1 for s in pending if s["phase"] == "generated")
        self.logger.log(f"[Generation] Completed: {generated_count}/{len(pending)} generated code")
        return states
    
    def _compile_single(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["phase"] != "generated" or not state["lean_code"]:
            return state
        
        compile_ok, compile_error = self.compiler.verify_compilation(state["lean_code"])
        
        if not compile_ok:
            state["history_feedback"].append(f"Compilation failed: {compile_error}")
            state["last_bad_code"] = state["lean_code"]
            state["last_compile_error"] = compile_error
            state["last_semantic_feedback"] = ""
            state["compile_passed"] = False
            state["phase"] = "compile_failed"
        else:
            state["compile_passed"] = True
            state["phase"] = "compiled"
        
        return state
    
    def batch_compile(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        to_compile = [s for s in states if s["phase"] == "generated" and s["lean_code"]]
        if not to_compile:
            return states
        
        self.logger.log(f"[Compilation] Processing {len(to_compile)} samples...")
        codes = [s["lean_code"] for s in to_compile]
        
        try:
            results = self.compiler.verify_batch(codes, batch_size=len(codes))
        except Exception as exc:
            self.logger.log(f"Batch compilation failed: {exc}")
            results = [(False, str(exc))] * len(to_compile)
        
        with tqdm(total=len(to_compile), desc="Compiling", unit="sample", ncols=120) as pbar:
            for state, (compile_ok, compile_error) in zip(to_compile, results):
                if not compile_ok:
                    state["history_feedback"].append(f"Compilation failed: {compile_error}")
                    state["last_bad_code"] = state["lean_code"]
                    state["last_compile_error"] = compile_error
                    state["last_semantic_feedback"] = ""
                    state["compile_passed"] = False
                    state["phase"] = "compile_failed"
                else:
                    state["compile_passed"] = True
                    state["phase"] = "compiled"
                pbar.update(1)
        
        compiled_count = sum(1 for s in to_compile if s["phase"] == "compiled")
        self.logger.log(f"[Compilation] Completed: {compiled_count}/{len(to_compile)} passed")
        return states
    
    def _semantic_single(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["phase"] != "compiled":
            return state
        
        semantic_ok, semantic_feedback, _judge_queries = self.semantic_judge.judge_semantic(
            state["statement"], state["lean_code"]
        )
        
        if not semantic_ok:
            state["history_feedback"].append(f"Semantic failed: {semantic_feedback}")
            state["last_bad_code"] = state["lean_code"]
            state["last_compile_error"] = ""
            state["last_semantic_feedback"] = semantic_feedback
            state["semantic_passed"] = False
            state["phase"] = "semantic_failed"
        else:
            state["semantic_passed"] = True
            state["status"] = "success"
            state["phase"] = "success"
        
        return state
    
    def batch_semantic(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        to_check = [s for s in states if s["phase"] == "compiled"]
        if not to_check:
            return states
        
        self.logger.log(f"[Semantic] Processing {len(to_check)} samples...")
        
        judge_workers = self.config.pipeline.judge_model_workers or self.config.pipeline.max_workers
        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            futures = {executor.submit(self._semantic_single, s): s for s in to_check}
            
            with tqdm(total=len(futures), desc="Semantic Check", unit="sample", ncols=120) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        state = futures[future]
                        state["history_feedback"].append(f"Semantic error: {exc}")
                        state["phase"] = "semantic_failed"
                        self.logger.log(f"Semantic error for item {state['idx']}: {exc}")
                    pbar.update(1)
        
        success_count = sum(1 for s in to_check if s["status"] == "success")
        self.logger.log(f"[Semantic] Completed: {success_count}/{len(to_check)} passed")
        return states
    
    def _reset_failed_for_retry(self, states: List[Dict[str, Any]]) -> int:
        retry_count = 0
        for s in states:
            if s["status"] == "pending" and s["phase"] in ("compile_failed", "semantic_failed"):
                # Back to pending; keep history for the next round
                s["phase"] = "init"
                s["lean_code"] = ""
                s["compile_passed"] = False
                s["semantic_passed"] = False
                s["pending_queries"] = []
                retry_count += 1
        return retry_count
    
    def _mark_remaining_failed(self, states: List[Dict[str, Any]]) -> None:
        for s in states:
            if s["status"] != "success":
                s["status"] = "failed"
    
    def run(self) -> None:
        data = self.file_io.load_jsonl(self.config.io.input_file)
        total = len(data)
        self.logger.log(f"Loaded {total} items from {self.config.io.input_file}")
        self.logger.log(f"Output directory: {self.config.io.run_dir}")
        self.logger.log(f"Running BATCH mode with max_iterations={self.config.pipeline.max_iterations}")
        
        states = [self._init_state(item, idx) for idx, item in enumerate(data)]
        
        initial_failed = sum(1 for s in states if s["status"] == "failed")
        if initial_failed > 0:
            self.logger.log(f"Initial failed (missing statement): {initial_failed}")
        
        batch_size = max(1, self.config.pipeline.pipeline_batch_size)
        total_batches = (len(states) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(states), batch_size):
            batch_no = batch_idx // batch_size + 1
            batch_states = states[batch_idx:batch_idx + batch_size]
            batch_total = len(batch_states)
            
            self.logger.log(f"\n{'='*60}")
            self.logger.log(f"BATCH {batch_no}/{total_batches} (size={batch_total})")
            self.logger.log(f"{'='*60}")
            
            for iteration in range(1, self.config.pipeline.max_iterations + 1):
                pending_count = sum(1 for s in batch_states if s["status"] == "pending")
                success_count = sum(1 for s in batch_states if s["status"] == "success")
                
                if pending_count == 0:
                    self.logger.log(
                        f"No more pending items in batch {batch_no}. "
                        f"Success: {success_count}/{batch_total}"
                    )
                    break
                
                self.logger.log(f"\n{'-'*60}")
                self.logger.log(f"BATCH {batch_no} ITERATION {iteration}/{self.config.pipeline.max_iterations}")
                self.logger.log(f"Pending: {pending_count}, Success: {success_count}, Total: {batch_total}")
                self.logger.log(f"{'-'*60}")
                
                for s in batch_states:
                    if s["status"] == "pending":
                        s["iterations"] = iteration
                
                batch_states = self.batch_generate(batch_states)
                
                batch_states = self.batch_compile(batch_states)
                
                batch_states = self.batch_semantic(batch_states)
                
                new_success = 0
                for s in batch_states:
                    if s["status"] == "success" and not s.get("success_written"):
                        record = self._build_success_record(s)
                        self.file_io.append_jsonl(record, self.config.io.success_file)
                        s["success_written"] = True
                        new_success += 1
                
                self.logger.log(f"Batch {batch_no} iteration {iteration} completed. New success: {new_success}")
                
                retry_count = self._reset_failed_for_retry(batch_states)
                if retry_count > 0:
                    self.logger.log(f"Reset {retry_count} failed samples for next iteration")
            
            self._mark_remaining_failed(batch_states)
            
            failed_written = 0
            for s in batch_states:
                if s["status"] == "failed" and not s.get("failed_written"):
                    record = self._build_failed_record(s)
                    self.file_io.append_jsonl(record, self.config.io.failed_file)
                    s["failed_written"] = True
                    failed_written += 1
            
            self.logger.log(
                f"Batch {batch_no} finished. "
                f"Success: {sum(1 for s in batch_states if s['status'] == 'success')}/{batch_total}, "
                f"Failed written: {failed_written}"
            )
        
        final_success = sum(1 for s in states if s["status"] == "success")
        final_failed = sum(1 for s in states if s["status"] == "failed")
        self.logger.log(f"\n{'='*60}")
        self.logger.log(f"FINAL RESULTS")
        self.logger.log(f"Success: {final_success}/{total} ({final_success/total*100:.1f}%)")
        self.logger.log(f"Failed: {final_failed}/{total} ({final_failed/total*100:.1f}%)")
        self.logger.log(f"{'='*60}")
    
    def _build_success_record(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "informal_statement": get_statement(state["original"]).strip(),
            "status": "success",
            "iterations": state["iterations"],
            "lean_code": state["lean_code"],
            "reasoning_content": state.get("last_reasoning_content", ""),
            "history_feedback": state["history_feedback"],
            "retrieval_history": state["retrieval_history"],
            "sharegpt": {
                "messages": [
                    {"role": "user", "content": state["last_prompt"]},
                    {"role": "assistant", "content": state["last_response"]},
                ]
            },
        }
    
    def _build_failed_record(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "informal_statement": get_statement(state["original"]).strip(),
            "status": "failed",
            "iterations": state.get("iterations", self.config.pipeline.max_iterations),
            "lean_code": state.get("lean_code", ""),
            "history_feedback": state.get("history_feedback", []),
            "retrieval_history": state.get("retrieval_history", []),
            "last_prompt": state.get("last_prompt", ""),
            "last_response": state.get("last_response", ""),
            "last_reasoning_content": state.get("last_reasoning_content", ""),
            "phase": state.get("phase", "unknown"),
        }
