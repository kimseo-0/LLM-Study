# olmocr_execute.py
# Windows-safe runner for olmOCR (DeepInfra / vLLM), with robust temp dir & encoding handling

import os
import sys
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_decode(b: bytes) -> str:
    # vLLM/olmocr가 가끔 바이너리 바이트 섞어서 내보낼 수 있음 → UTF-8로 안전 디코딩
    return b.decode("utf-8", errors="ignore")


def run_cmd(cmd_list: List[str], env: Optional[dict] = None) -> None:
    """
    subprocess를 바이너리 모드로 캡처해서 UTF-8로 안전 디코딩.
    Windows 인코딩(cp949) 문제/UnicodeDecodeError 방지.
    """
    log.info("Running: %s", " ".join(cmd_list))
    p = subprocess.run(
        cmd_list,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        # encoding/Errors 지정 금지
    )
    out = safe_decode(p.stdout)
    err = safe_decode(p.stderr)

    if out.strip():
        print("---- STDOUT ----\n", out, sep="")
    if err.strip():
        print("---- STDERR ----\n", err, sep="")

    if p.returncode != 0:
        raise RuntimeError(f"olmocr pipeline failed (exit={p.returncode})")


def copy_pdf_to_workspace(pdf_src: Path, workspace: Path) -> Path:
    """
    Temp/락/권한 이슈 회피를 위해 PDF를 워크스페이스 내부로 복제해서 사용.
    """
    input_dir = ensure_dir(workspace / "input")
    dst = input_dir / f"{uuid.uuid4().hex}{pdf_src.suffix.lower()}"
    shutil.copy2(str(pdf_src), str(dst))
    log.info("Copied PDF to: %s", dst)
    return dst


def build_cmd(
    workspace_abs: Path,
    pdf_abs: Path,
    api_key: str,
    *,
    server: str = "https://api.deepinfra.com/v1/openai",
    model: str = "allenai/olmOCR-7B-0825",
    pages_per_group: int = 2,
    gpu_memory_utilization: float = 0.30,
    max_model_len: int = 128,
    max_num_seqs: int = 2,
    swap_space_gb: int = 24,
    markdown: bool = True,
) -> List[str]:
    """
    Windows에서는 문자열 한 줄로 조합하지 말고 '리스트 인자'로 넘기는 게 가장 안전.
    """
    cmd = [
        sys.executable, "-m", "olmocr.pipeline",
        str(workspace_abs),
        "--server", server,
        "--api_key", api_key,
        "--model", model,
        "--pages_per_group", str(pages_per_group),
        "--pdfs", str(pdf_abs),
        "--gpu_memory_utilization", str(gpu_memory_utilization),
        "--max_model_len", str(max_model_len),
        "--max_num_seqs", str(max_num_seqs),
        "--swap-space", str(swap_space_gb),
    ]
    if markdown:
        cmd.append("--markdown")
    return cmd


def make_env_with_custom_tmp(base_env: Optional[dict], tmp_dir: Path) -> dict:
    env = (base_env or os.environ).copy()
    ensure_dir(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMPDIR"] = str(tmp_dir)
    return env


# ----------------------------
# Public API
# ----------------------------
def run_olmocr_remote(
    workspace_abs_path: str,
    pdf_abs_path: str,
    api_key: Optional[str] = None,
    *,
    # vLLM / 메모리 안전 프리셋(필요시 조정)
    pages_per_group: int = 2,
    gpu_memory_utilization: float = 0.30,
    max_model_len: int = 128,
    max_num_seqs: int = 2,
    swap_space_gb: int = 24,
    server: str = "https://api.deepinfra.com/v1/openai",
    model: str = "allenai/olmOCR-7B-0825",
    markdown: bool = True,
    copy_pdf: bool = True,
    clean_tmp: bool = True,
) -> None:
    """
    - 임시 디렉터리 강제 → Permission denied 회피
    - PDF 복제본 사용(선택) → 파일 락/미리보기 충돌 회피
    - stdout/stderr 안전 디코딩
    """
    ws = Path(workspace_abs_path).resolve()
    pdf = Path(pdf_abs_path).resolve()
    ensure_dir(ws)

    if api_key is None:
        api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        raise ValueError("API key is required. Set 'api_key' or env DEEPINFRA_API_KEY.")

    # 깨끗한 temp 폴더 준비
    tmp_dir = ws / "tmp"
    if clean_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    ensure_dir(tmp_dir)

    # PDF 복제(옵션)
    pdf_for_run = copy_pdf_to_workspace(pdf, ws) if copy_pdf else pdf

    # 커맨드/환경 구성
    cmd = build_cmd(
        workspace_abs=ws,
        pdf_abs=pdf_for_run,
        api_key=api_key,
        server=server,
        model=model,
        pages_per_group=pages_per_group,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        swap_space_gb=swap_space_gb,
        markdown=markdown,
    )
    env = make_env_with_custom_tmp(os.environ, tmp_dir)

    # 실행
    run_cmd(cmd, env=env)


# ----------------------------
# CLI entry (example)
# ----------------------------
if __name__ == "__main__":
    # 예시 값: 필요에 맞게 수정하거나, argparse로 바꿔도 됨
    WORKSPACE = r"C:\Potenup\LLM-Study\olmocr_run"
    PDF_PATH  = r"C:\Potenup\LLM-Study\data\checkup_2024.pdf"

    API_KEY = os.environ.get("DEEPINFRA_API_KEY", "").strip()

    run_olmocr_remote(
        workspace_abs_path=WORKSPACE,
        pdf_abs_path=PDF_PATH,
        api_key=API_KEY,                  # 없으면 위에서 에러
        pages_per_group=2,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        max_num_seqs=2,
        swap_space_gb=24,
        server="https://api.deepinfra.com/v1/openai",
        model="allenai/olmOCR-7B-0825",
        markdown=True,
        copy_pdf=True,                    # 락/권한 이슈 피하려고 True 권장
        clean_tmp=True,                   # 매번 깨끗한 temp 보장
    )
