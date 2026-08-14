import asyncio
import json
import os
import platform
import signal
import tempfile
from asyncio.subprocess import Process
from datetime import datetime
from uuid import UUID, uuid4

import psutil
from kimina_client import (
    Command,
    CommandResponse,
    Diagnostics,
    Error,
    Infotree,
    ReplResponse,
    Snippet,
)
from loguru import logger
from rich.syntax import Syntax

from .db import db
from .errors import LeanError, ReplError
from .logger import console
from .models import ReplStatus
from .prisma_client import prisma
from .settings import Environment, settings
from .utils import is_blank

log_lock = asyncio.Lock()


async def log_snippet(uuid: UUID, snippet_id: str, code: str) -> None:
    if settings.environment == Environment.prod:
        header = f"[{uuid.hex[:8]}] Running snippet {snippet_id}:"
        async with log_lock:
            logger.info(header)
            # Log the code as part of the message or in a separate log entry
            logger.info(f"Code snippet:\n{code or '<empty>'}")
    else:
        header = f"\\[{uuid.hex[:8]}] Running snippet [bold magenta]{snippet_id}[/bold magenta]:"
        syntax = Syntax(
            code or "<empty>",
            "lean",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )

        async with log_lock:
            logger.info(header)
            if console:
                console.print(syntax)


class Repl:
    def __init__(
        self,
        uuid: UUID,
        created_at: datetime,
        header: str = "",
        *,
        max_repl_mem: int,
        max_repl_uses: int,
    ) -> None:
        self.uuid = uuid
        self.header = header
        self.use_count = 0
        self.created_at = created_at
        self.last_check_at = created_at

        # Stores the response received when running the import header.
        self.header_cmd_response: ReplResponse | None = None

        self.proc: Process | None = None
        self.error_file = tempfile.TemporaryFile("w+")
        self.max_memory_bytes = max_repl_mem * 1024 * 1024
        self.max_repl_uses = max_repl_uses

        self._loop: asyncio.AbstractEventLoop | None = None

        # REPL statistics
        self.cpu_per_exec: dict[int, float] = {}
        self.mem_per_exec: dict[int, int] = {}

        # Vars that hold max CPU / mem usage per proof.
        self._cpu_max: float = 0.0  # CPU as a percentage of a single core
        self._mem_max: int = 0

        self._ps_proc: psutil.Process | None = None
        self._cpu_task: asyncio.Task[None] | None = None
        self._mem_task: asyncio.Task[None] | None = None

    @classmethod
    async def create(cls, header: str, max_repl_uses: int, max_repl_mem: int) -> "Repl":
        if db.connected:
            record = await prisma.repl.create(
                data={
                    "header": header,
                    "max_repl_uses": max_repl_uses,
                    "max_repl_mem": max_repl_mem,
                }
            )
            return cls(
                uuid=UUID(record.uuid),
                created_at=record.created_at,
                header=record.header,
                max_repl_uses=record.max_repl_uses,
                max_repl_mem=record.max_repl_mem,
            )
        return cls(
            uuid=uuid4(),
            created_at=datetime.now(),
            header=header,
            max_repl_uses=max_repl_uses,
            max_repl_mem=max_repl_mem,
        )

    @property
    def exhausted(self) -> bool:
        if self.max_repl_uses < 0:
            return False
        if self.header and not is_blank(self.header):
            # Header does not count towards uses.
            return self.use_count >= self.max_repl_uses + 1
        return self.use_count >= self.max_repl_uses

    async def start(self) -> None:
        # TODO: try/catch this bit and raise as REPL startup error.
        self._loop = asyncio.get_running_loop()

        def _preexec() -> None:
            import resource

            # Memory limit: Use RLIMIT_DATA instead of RLIMIT_AS
            # RLIMIT_AS is too restrictive for multi-threaded programs like Lean
            # because it includes all virtual memory (thread stacks, shared libraries, etc.)
            # RLIMIT_DATA only limits the data segment (heap), allowing threads to be created
            if platform.system() != "Darwin":  # Only for Linux
                try:
                    # Use RLIMIT_DATA to limit heap memory without blocking thread creation
                    resource.setrlimit(
                        resource.RLIMIT_DATA, (self.max_memory_bytes, self.max_memory_bytes)
                    )
                except (ValueError, OSError):
                    # If RLIMIT_DATA fails, don't set any memory limit
                    # This is safer than blocking thread creation
                    pass

            # Note: RLIMIT_NPROC is per-user, not per-process, so we don't set it here
            # Thread creation is controlled by available memory and system limits
            # No CPU limit on REPL, most Lean proofs take up to one core.
            # The adjustment variables are the maximum number of REPLs and the timeout.
            # See https://github.com/leanprover-community/repl/issues/91

            os.setsid()

        # Use a custom environment to ensure clean output
        clean_env = os.environ.copy()
        # Remove any variables that might affect logging output
        clean_env.pop('PYTHONPATH', None)
        clean_env.pop('LOGURU_LEVEL', None)
        
        self.proc = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            settings.repl_path,
            cwd=settings.project_dir,
            env=clean_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_preexec,
        )

        self._ps_proc = psutil.Process(self.proc.pid)
        now = self._loop.time()
        self._last_check = now
        self._last_cpu_time = self._sum_cpu_times(self._ps_proc)

        self._cpu_max = 0.0
        self._mem_max = 0
        self._cpu_task = self._loop.create_task(self._cpu_monitor())
        self._mem_task = self._loop.create_task(self._mem_monitor())

        # Log memory limits after process creation (safe to log in parent process)
        try:
            mem_mb = self.max_memory_bytes / (1024 * 1024)
            logger.info(f"\\[{self.uuid.hex[:8]}] Started REPL with memory limit: {mem_mb:.0f}MB (using RLIMIT_DATA)")
        except Exception:
            logger.info(f"\\[{self.uuid.hex[:8]}] Started REPL")

    @staticmethod
    def _sum_cpu_times(proc: psutil.Process) -> float:
        total = proc.cpu_times().user + proc.cpu_times().system
        for c in proc.children(recursive=True):
            t = c.cpu_times()
            total += t.user + t.system
        return float(total)

    async def _cpu_monitor(self) -> None:
        while self.is_running and self._ps_proc and self._loop:
            await asyncio.sleep(1)
            now = self._loop.time()

            cur_cpu = self._sum_cpu_times(self._ps_proc)
            delta_cpu = cur_cpu - self._last_cpu_time
            delta_t = now - self._last_check
            usage_pct = (delta_cpu / delta_t) * 100
            self._cpu_max = max(self._cpu_max, usage_pct)
            self._last_cpu_time = cur_cpu
            self._last_check = now

    async def _mem_monitor(self) -> None:
        while self.is_running and self._ps_proc:
            await asyncio.sleep(1)
            total = self._ps_proc.memory_info().rss
            for child in self._ps_proc.children(recursive=True):
                total += child.memory_info().rss
            self._mem_max = max(self._mem_max, total)

    @property
    def is_running(self) -> bool:
        if not self.proc:
            return False
        return self.proc.returncode is None

    async def send_timeout(
        self,
        snippet: Snippet,
        timeout: float,
        is_header: bool = False,
        infotree: Infotree | None = None,
    ) -> ReplResponse:
        cmd_response = None
        elapsed_time = (
            0.0  # TODO: check what's the best time to check elapsed time, time lib?
        )
        diagnostics = Diagnostics(repl_uuid=str(self.uuid))

        try:
            cmd_response, elapsed_time, diagnostics = await asyncio.wait_for(
                self.send(snippet, is_header=is_header, infotree=infotree),
                timeout=timeout,
            )
        except TimeoutError as e:
            logger.error(
                "\\[{}] Lean REPL command timed out in {} seconds",
                self.uuid.hex[:8],
                timeout,
            )
            raise e
        except LeanError as e:
            logger.exception("Lean REPL error: %s", e)
            raise e
        except ReplError as e:
            logger.exception("REPL error: %s", e)
            raise e

        return ReplResponse(
            id=snippet.id,
            response=cmd_response,
            time=elapsed_time,
            diagnostics=diagnostics if len(diagnostics) > 0 else None,
        )

    async def send(
        self,
        snippet: Snippet,
        is_header: bool = False,
        infotree: Infotree | None = None,
    ) -> tuple[CommandResponse | Error, float, Diagnostics]:
        await log_snippet(self.uuid, snippet.id, snippet.code)

        self._cpu_max = 0.0
        self._mem_max = 0

        if not self.proc or self.proc.returncode is not None:
            logger.error("REPL process not started or shut down")
            raise ReplError("REPL process not started or shut down")

        loop = self._loop or asyncio.get_running_loop()

        if self.proc.stdin is None:
            raise ReplError("stdin pipe not initialized")
        if self.proc.stdout is None:
            raise ReplError("stdout pipe not initialized")

        input: Command = {"cmd": snippet.code}

        if self.use_count != 0 and not is_header:  # remove is_header
            input["env"] = 0
            input["gc"] = True

        if infotree:
            input["infotree"] = infotree

        payload = (json.dumps(input, ensure_ascii=False) + "\n\n").encode("utf-8")

        start = loop.time()
        logger.debug("Sending payload to REPL")

        try:
            self.proc.stdin.write(payload)
            await self.proc.stdin.drain()
        except BrokenPipeError:
            logger.error("Broken pipe while writing to REPL stdin")
            raise LeanError("Lean process broken pipe")
        except Exception as e:
            logger.error("Failed to write to REPL stdin: %s", e)
            raise LeanError("Failed to write to REPL stdin")

        logger.debug("Reading response from REPL stdout")
        raw = await self._read_response()
        elapsed = loop.time() - start

        # Check if process is still alive and capture exit info
        if self.proc.returncode is not None:
            logger.error(f"REPL process exited with code {self.proc.returncode} during command execution")
            
            # Try to read any stderr data
            if self.proc.stderr:
                try:
                    stderr_data = await asyncio.wait_for(self.proc.stderr.read(8192), timeout=0.1)
                    if stderr_data:
                        logger.error(f"REPL stderr: {stderr_data.decode('utf-8', errors='replace')}")
                except asyncio.TimeoutError:
                    pass  # No stderr data available

        # Check if we got empty response and the process might have crashed
        if not raw:
            logger.error("Empty response from REPL - checking process status")
            if self.proc.returncode is not None:
                logger.error(f"REPL process exited with code: {self.proc.returncode}")
                
                # Try to read stderr for crash information
                if self.proc.stderr:
                    try:
                        stderr_data = await asyncio.wait_for(self.proc.stderr.read(8192), timeout=0.1)
                        if stderr_data:
                            logger.error(f"REPL stderr content: {stderr_data.decode('utf-8', errors='replace')}")
                    except asyncio.TimeoutError:
                        logger.error("No stderr data available")
                        
                # Check error file for any error messages
                self.error_file.seek(0)
                err = self.error_file.read().strip()
                if err:
                    logger.error(f"Error file content: {err}")
                
                raise ReplError(f"REPL process crashed with exit code {self.proc.returncode}")
            else:
                # Process is still running but not producing output - this suggests a hang
                logger.error("REPL process is still running but produced no output")
                
                # Try to read any stderr that might be available
                if self.proc.stderr:
                    try:
                        stderr_data = await asyncio.wait_for(self.proc.stderr.read(8192), timeout=0.1)
                        if stderr_data:
                            logger.error(f"REPL stderr while hanging: {stderr_data.decode('utf-8', errors='replace')}")
                        else:
                            logger.error("No stderr data while hanging")
                    except asyncio.TimeoutError:
                        logger.error("Timeout reading stderr while hanging")
                
                # Check error file
                self.error_file.seek(0)
                err = self.error_file.read().strip()
                if err:
                    logger.error(f"Error file content while hanging: {err}")
                else:
                    logger.error("No error file content while hanging")
                
                # Log process status
                if self._ps_proc:
                    try:
                        status = self._ps_proc.status()
                        memory_info = self._ps_proc.memory_info()
                        logger.error(f"Process status: {status}, Memory: {memory_info.rss / 1024 / 1024:.1f}MB")
                    except Exception as e:
                        logger.error(f"Failed to get process info: {e}")
                
                raise ReplError("REPL process hanging - no output but still running")

        logger.debug("Raw response from REPL: %r", raw)
        try:
            resp: CommandResponse | Error = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("JSON decode error: %r", raw)
            raise ReplError("JSON decode error")

        self.error_file.seek(0)
        err = self.error_file.read().strip()
        self.error_file.seek(0)
        self.error_file.truncate(0)
        if err:
            logger.error("Stderr: %s", err)
            raise LeanError(err)

        elapsed_time = round(elapsed, 6)
        diagnostics: Diagnostics = {
            "repl_uuid": str(self.uuid),
            "cpu_max": self._cpu_max,
            "memory_max": self._mem_max,
        }

        self.cpu_per_exec[self.use_count] = self._cpu_max
        self.mem_per_exec[self.use_count] = self._mem_max

        self.use_count += 1
        return resp, elapsed_time, diagnostics

    async def _read_response(self) -> bytes:
        if not self.proc or self.proc.stdout is None:
            logger.error("REPL process not started or stdout pipe not initialized")
            raise ReplError("REPL process not started or stdout pipe not initialized")

        lines: list[bytes] = []
        try:
            chunk_count = 0
            read_timeout = 120.0  # 120 second timeout for each read - exact? and library_search can be very slow
            
            while True:
                try:
                    # Add timeout to readline to detect hanging
                    chunk = await asyncio.wait_for(self.proc.stdout.readline(), timeout=read_timeout)
                    chunk_count += 1
                    logger.debug(f"Read chunk {chunk_count}: {repr(chunk)}")
                    
                    # EOF encountered (no more data)
                    if not chunk:
                        logger.debug("EOF encountered, stopping read")
                        break
                    
                    # Stop when we encounter an empty line (just newline or whitespace)
                    if chunk.strip() == b"":
                        logger.debug(f"Empty line detected after {len(lines)} content lines, stopping")
                        break
                    
                    # Add the line to our response
                    lines.append(chunk)
                    logger.debug(f"Added line {len(lines)}: {repr(chunk[:100])}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Timeout waiting for REPL output after {read_timeout}s")
                    # If we've already read some content, break and try to parse it
                    if lines:
                        logger.info("Got timeout but have some content, trying to parse")
                        break
                    else:
                        logger.error("Timeout with no content - REPL is hanging")
                        raise ReplError("REPL hanging - timeout waiting for output")
                    
        except Exception as e:
            logger.error("Failed to read from REPL stdout: %s", e)
            raise LeanError("Failed to read from REPL stdout")
        
        result = b"".join(lines)
        logger.debug(f"Final response length: {len(result)}, content: {repr(result[:200])}")
        return result

    async def close(self) -> None:
        if self.proc:
            self.last_check_at = datetime.now()
            assert self.proc.stdin is not None, "stdin pipe not initialized"
            self.proc.stdin.close()
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            await self.proc.wait()
            if self._cpu_task:
                self._cpu_task.cancel()
            if self._mem_task:
                self._mem_task.cancel()

            if db.connected:
                await prisma.repl.update(
                    where={"uuid": str(self.uuid)},
                    data={"status": ReplStatus.STOPPED},  # type: ignore
                )


async def close_verbose(repl: Repl) -> None:
    uuid = repl.uuid
    logger.info(f"Closing REPL {uuid.hex[:8]}")
    await repl.close()
    del repl
    logger.info(f"Closed REPL {uuid.hex[:8]}")
