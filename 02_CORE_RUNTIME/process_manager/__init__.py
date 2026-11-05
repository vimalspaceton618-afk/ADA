"""Safe local subprocess lifecycle management."""

from .supervisor import ProcessRecord, ProcessSupervisor

__all__ = ["ProcessRecord", "ProcessSupervisor"]
