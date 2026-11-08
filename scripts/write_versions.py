"""Record the exact installed versions for the run record."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import RESULTS, env_info, write_json  # noqa: E402


def main() -> None:
    info = env_info()
    info["numpy"] = __import__("numpy").__version__
    info["pillow"] = __import__("PIL").__version__
    write_json(RESULTS / "versions.json", info)
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
