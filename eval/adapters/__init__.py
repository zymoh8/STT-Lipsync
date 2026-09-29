from .base import Adapter, Result, REGISTRY, build, register  # noqa: F401

# Import side effects register each engine. A missing third-party library
# must not stop the others loading - you will often have vosk installed
# but not moonshine, or vice versa.
for _m in ("dummy_adapter", "vosk_adapter", "whispercpp_adapter",
           "moonshine_adapter"):
    try:
        __import__(f"{__name__}.{_m}", fromlist=["*"])
    except Exception as _e:  # pragma: no cover
        import sys
        print(f"[adapters] {_m} unavailable: {_e}", file=sys.stderr)
