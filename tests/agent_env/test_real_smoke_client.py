import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_smoke import Client, RESULT_PREFIX  # noqa: E402


def test_client_drains_text_stream_read_ahead_without_select_deadlock() -> None:
    payload = json.dumps({"status": "ready", "level": 3})
    program = (
        "import sys, time; "
        f"sys.stdout.write('startup log\\n' * 5000 + {RESULT_PREFIX!r} + {payload!r} + '\\n'); "
        "sys.stdout.flush(); time.sleep(0.5)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = Client(process, timeout=5.0)
    try:
        assert client.result("ready") == {"status": "ready", "level": 3}
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
        client.reader_thread.join(timeout=2)
