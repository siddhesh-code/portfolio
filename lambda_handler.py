import os
import sys
import subprocess
import traceback

def _run_script(path_or_module: str):
    """
    Runs a Python file or module inside the Lambda container.
    Prints debug logs and streams script output to CloudWatch.
    """
    task_root = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
    full_path = os.path.join(task_root, path_or_module)

    if path_or_module.endswith(".py") and os.path.isfile(full_path):
        cmd = [sys.executable, full_path]
        print(f"[Lambda] Running file: {full_path}")
    else:
        cmd = [sys.executable, "-m", path_or_module.replace(".py", "")]
        print(f"[Lambda] Running module: {path_or_module}")

    try:
        # Capture stdout/stderr so we can debug in CloudWatch
        result = subprocess.run(
            cmd,
            cwd=task_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240  # safety timeout (seconds)
        )
        print(f"[Lambda] Script finished: {cmd}")
        print(f"[Lambda] STDOUT:\n{result.stdout}")
        print(f"[Lambda] STDERR:\n{result.stderr}")
        if result.returncode != 0:
            raise RuntimeError(f"Script failed with code {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[Lambda] ERROR: Script timed out: {cmd}")
        raise
    except Exception as e:
        print(f"[Lambda] ERROR running {cmd}: {e}")
        raise

def lambda_handler(event, context):
    """
    Calls reversals.py then scripts/send_reversals_email.py.
    """
    try:
        _run_script("reversals.py")
        _run_script("scripts/send_reversals_email.py")
        return {"ok": True}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}
