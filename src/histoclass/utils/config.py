"""Compatibility exports for unified config module."""

from ..config import (
    AppConfig,
    SeedConfig,
    config_to_dict,
    default_config_path,
    load_config,
    parse_config_dict,
    project_root,
    save_config,
)

__all__ = [
    "AppConfig",
    "SeedConfig",
    "config_to_dict",
    "default_config_path",
    "load_config",
    "parse_config_dict",
    "project_root",
    "save_config",
]
