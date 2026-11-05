"""ADA research package.

Only stable, explicitly supported foundations are exported here.
"""

from .config import ADAConfig, ConfigurationError

__all__ = ["ADAConfig", "ConfigurationError"]
