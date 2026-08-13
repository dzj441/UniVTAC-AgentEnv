"""Agent-facing protocols for UniVTAC environments.

The package deliberately has no Isaac Sim imports.  Protocol invariants can
therefore be tested without launching the simulator, while simulator-specific
runners remain thin adapters under ``scripts/``.
"""

from .profiles import AgentEnvProfile, get_profile, list_profiles
from .protocol import EpisodeProtocol, ProtocolError

__all__ = [
    "AgentEnvProfile",
    "EpisodeProtocol",
    "ProtocolError",
    "get_profile",
    "list_profiles",
]
