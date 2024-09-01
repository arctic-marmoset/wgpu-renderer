import subprocess
import sys

INVALID_ARGS = [
    "-Wl,-dylib"
]

def is_invalid_arg(arg: str) -> bool:
    if arg.startswith("-Wl,-exported_symbols_list"):
        return True

    return arg in INVALID_ARGS

target = sys.argv[1]
args = filter(lambda x: not is_invalid_arg(x), sys.argv[2:])

result = subprocess.run([
    "zig",
    "cc",
    "-target",
    target,
    *args
], stderr=sys.stderr)

sys.exit(result.returncode)
