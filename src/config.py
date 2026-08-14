"""Pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def _parse_url_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass
class APIConfig:
    base_model_url: str = "http://localhost:8001/v1/chat/completions"
    retrieval_planner_url: Optional[str] = None
    judge_model_url: str = "http://localhost:8002/v1/chat/completions"
    lean_server_url: str = "http://localhost:8000"
    lean_explore_url: Optional[str] = None
    api_key: str = "EMPTY"
    request_timeout: int = 300
    
    base_model_name: Optional[str] = None
    retrieval_planner_name: Optional[str] = None
    judge_model_name: Optional[str] = None
    
    base_model_urls: List[str] = field(default_factory=list, repr=False)
    retrieval_planner_urls: List[str] = field(default_factory=list, repr=False)
    judge_model_urls: List[str] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        if self.retrieval_planner_url is None:
            self.retrieval_planner_url = self.base_model_url
        self.base_model_urls = _parse_url_list(self.base_model_url) or [self.base_model_url]
        self.retrieval_planner_urls = _parse_url_list(self.retrieval_planner_url) or [self.retrieval_planner_url]
        self.judge_model_urls = _parse_url_list(self.judge_model_url) or [self.judge_model_url]


@dataclass
class PipelineConfig:
    max_iterations: int = 3
    retrieval_limit: int = 15
    query_top_k: int = 5
    max_workers: int = 16
    retrieval_workers: int = 32
    # Per-model concurrency; None falls back to max_workers
    base_model_workers: Optional[int] = None
    retrieval_planner_workers: Optional[int] = None
    judge_model_workers: Optional[int] = None
    temperature: float = 0.7
    pipeline_batch_size: int = 200
    
    use_batch_mode: bool = False  # True: staged batch mode, False: per-sample mode
    
    # Batch-mode settings
    generation_batch_size: int = 1000
    compile_batch_size: int = 500
    semantic_batch_size: int = 500


@dataclass
class IOConfig:
    input_file: str = ""
    output_dir: str = "./output"
    run_dir: Optional[Path] = field(default=None, repr=False)
    
    # Output paths (created at runtime)
    log_file: Optional[Path] = field(default=None, repr=False)
    success_file: Optional[Path] = field(default=None, repr=False)
    failed_file: Optional[Path] = field(default=None, repr=False)
    
    checkpoint_file: Optional[Path] = field(default=None, repr=False)
    
    def setup_run_dir(self) -> None:
        """Create a run directory named by date plus an incrementing index."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d")
        prefix = f"run_{date_str}_"
        max_index = 0
        for child in output_path.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                continue
            max_index = max(max_index, int(suffix))
        
        next_index = max_index + 1
        self.run_dir = output_path / f"{prefix}{next_index}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.run_dir / "pipeline.log"
        self.success_file = self.run_dir / "success.jsonl"
        self.failed_file = self.run_dir / "failed.jsonl"
        self.checkpoint_file = self.run_dir / "checkpoint.jsonl"


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    io: IOConfig = field(default_factory=IOConfig)
    
    @classmethod
    def from_args(cls, args) -> "Config":
        api_config = APIConfig(
            base_model_url=args.base_model_url,
            retrieval_planner_url=args.retrieval_planner_url,
            judge_model_url=args.judge_model_url,
            lean_server_url=args.lean_server_url,
            lean_explore_url=args.lean_explore_url,
            api_key=args.api_key,
            request_timeout=args.request_timeout,
            base_model_name=args.base_model_name,
            retrieval_planner_name=args.retrieval_planner_name,
            judge_model_name=args.judge_model_name,
        )
        
        pipeline_config = PipelineConfig(
            max_iterations=args.max_iterations,
            retrieval_limit=args.retrieval_limit,
            query_top_k=args.query_top_k,
            max_workers=args.max_workers,
            retrieval_workers=args.retrieval_workers,
            base_model_workers=getattr(args, 'base_model_workers', None),
            retrieval_planner_workers=getattr(args, 'retrieval_planner_workers', None),
            judge_model_workers=getattr(args, 'judge_model_workers', None),
            temperature=args.temperature,
            pipeline_batch_size=getattr(args, 'pipeline_batch_size', 200),
            use_batch_mode=getattr(args, 'use_batch_mode', False),
            generation_batch_size=getattr(args, 'generation_batch_size', 1000),
            compile_batch_size=getattr(args, 'compile_batch_size', 500),
            semantic_batch_size=getattr(args, 'semantic_batch_size', 500),
        )
        
        io_config = IOConfig(
            input_file=args.input,
            output_dir=args.output_dir,
        )
        io_config.setup_run_dir()
        
        return cls(api=api_config, pipeline=pipeline_config, io=io_config)
