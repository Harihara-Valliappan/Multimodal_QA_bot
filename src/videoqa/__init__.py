"""VideoQA adaptation of InternVL3.5 with DoRA.

Public surface is deliberately small: everything the scripts need is
re-exported here so callers never reach into private modules.
"""

__version__ = "0.1.0"

from videoqa.config import ProjectConfig, load_config  # noqa: E402
from videoqa.logging_utils import get_logger, write_report  # noqa: E402

__all__ = ["ProjectConfig", "load_config", "get_logger", "write_report", "__version__"]
