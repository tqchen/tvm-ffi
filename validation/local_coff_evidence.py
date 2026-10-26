"""Assert the local objects replace only the COFF dispatcher at final linkage."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys


def sdk_snapshot():
    prefix = Path(subprocess.check_output(
        [sys.executable, "-m", "mlc_llvm_static.config", "--prefix"], text=True
    ).strip())
    return {
        p.relative_to(prefix).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(prefix.rglob("*")) if p.is_file()
    }


snapshot = sdk_snapshot()
if sys.argv[1] == "before":
    Path("sdk-before.json").write_text(json.dumps(snapshot, indent=2))
else:
    assert json.loads(Path("sdk-before.json").read_text()) == snapshot, "SDK was changed"
    maps = list(Path("addons/tvm_ffi_orcjit/build").rglob("*.map"))
    assert maps, "Link map missing"
    text = "\n".join(p.read_text(errors="replace") for p in maps)
    assert "coff_dispatch.obj" in text.lower(), "Local dispatcher missing from link map"
    assert "coff_aarch64.obj" in text.lower(), "Local backend missing from link map"
    assert not re.search(r"(?:[:/\\(]|\b)COFF\.cpp\.obj\b", text, re.I), "Stock COFF member extracted"
    assert not re.search(r"/FORCE(?::MULTIPLE)?|/WHOLEARCHIVE", Path("native-build.log").read_text(errors="replace"), re.I)
    evidence = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sdk_version": importlib.metadata.version("mlc-llvm-static"),
        "sdk_unchanged": True,
        "sdk_file_count": len(snapshot),
        "LLVMJITLink_sha256": {k: v for k, v in snapshot.items() if k.endswith("LLVMJITLink.lib")},
        "local_coff_objects_selected": True,
        "stock_coff_member_selected": False,
        "maps": [p.as_posix() for p in maps],
    }
    Path("native-evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
