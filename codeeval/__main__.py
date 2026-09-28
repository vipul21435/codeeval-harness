"""``python -m codeeval``: see :mod:`codeeval.cli`."""

import sys

from codeeval.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    sys.exit(main())
