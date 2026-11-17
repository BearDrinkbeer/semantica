"""
Thread-safe per-workspace GraphSession registry for multi-tenant graph isolation.

Each workspace maps to a distinct FalkorDB graph identifier (``graph:ws_<id>``)
and its own ``GraphSession`` instance. The registry handles lazy initialization,
caching, and graceful shutdown of all sessions.
"""

import logging
import threading
from typing import Callable, Dict, Optional

from ..context.context_graph import ContextGraph
from .session import GraphSession

logger = logging.getLogger(__name__)

# Default graph key builder used when no custom mapper is supplied.
DEFAULT_GRAPH_KEY_PREFIX = "graph:ws_"


def default_graph_key(ws_id: str) -> str:
    """Map a workspace ID to its FalkorDB graph key.

    The ``default`` workspace uses the legacy key ``graph:default`` for
    backwards compatibility with existing single-tenant installations.
    """
    if ws_id == "default":
        return "graph:default"
    return f"{DEFAULT_GRAPH_KEY_PREFIX}{ws_id}"


class WorkspaceSessionRegistry:
    """Concurrent registry that creates, caches, and retrieves per-workspace
    ``GraphSession`` instances.

    Thread-safety is provided by an internal ``RLock``; all public methods are
    safe to call from concurrent request threads.

    Parameters:
        graph_key_fn:
            Optional callable mapping ``ws_id -> str`` that returns the
            FalkorDB graph key for a workspace. Defaults to
            :func:`default_graph_key`.
        provenance_storage_path:
            Optional path forwarded to each ``GraphSession`` for its
            provenance database.
        session_factory:
            Optional callable ``(graph_key: str) -> GraphSession`` for custom
            session construction (e.g. connecting to a real FalkorDB instance).
            When omitted the registry creates an in-memory ``ContextGraph``
            session for each workspace.
    """

    def __init__(
        self,
        graph_key_fn: Optional[Callable[[str], str]] = None,
        provenance_storage_path: Optional[str] = None,
        session_factory: Optional[Callable[[str], GraphSession]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, GraphSession] = {}
        self._graph_keys: Dict[str, str] = {}  # ws_id -> graph_key
        self._graph_key_fn = graph_key_fn or default_graph_key
        self._provenance_storage_path = provenance_storage_path
        self._session_factory = session_factory
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, ws_id: str) -> GraphSession:
        """Return the ``GraphSession`` for *ws_id*, creating it lazily if needed.

        Raises:
            RuntimeError: If the registry has been shut down.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("WorkspaceSessionRegistry is shut down")
            session = self._sessions.get(ws_id)
            if session is not None:
                return session
            return self._create_locked(ws_id)

    def get_if_exists(self, ws_id: str) -> Optional[GraphSession]:
        """Return the cached session for *ws_id* without creating one."""
        with self._lock:
            return self._sessions.get(ws_id)

    def register(self, ws_id: str, session: GraphSession) -> None:
        """Externally register a pre-built session (e.g. the legacy default session).

        If the workspace already has a session, it is **replaced** silently.
        """
        graph_key = self._graph_key_fn(ws_id)
        with self._lock:
            self._sessions[ws_id] = session
            self._graph_keys[ws_id] = graph_key
            logger.info(
                "Registered external session for workspace %s (graph_key=%s)",
                ws_id,
                graph_key,
            )

    def remove(self, ws_id: str) -> Optional[GraphSession]:
        """Remove and return the session for *ws_id*, or ``None``."""
        with self._lock:
            self._graph_keys.pop(ws_id, None)
            return self._sessions.pop(ws_id, None)

    def graph_key_for(self, ws_id: str) -> str:
        """Return the FalkorDB graph key for a workspace without creating a session."""
        with self._lock:
            cached = self._graph_keys.get(ws_id)
            if cached:
                return cached
        return self._graph_key_fn(ws_id)

    @property
    def active_workspaces(self) -> list[str]:
        """List workspace IDs with live sessions."""
        with self._lock:
            return list(self._sessions.keys())

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def shutdown(self) -> None:
        """Gracefully close all cached sessions and prevent new ones from being created.

        Idempotent: calling twice has no additional effect.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            ws_ids = list(self._sessions.keys())

        for ws_id in ws_ids:
            try:
                session = self._sessions.pop(ws_id, None)
                if session is not None:
                    logger.info("Shutting down session for workspace %s", ws_id)
                    # GraphSession has no explicit close(), but clear heavy
                    # cached state to free memory promptly.
                    session.invalidate_embedding_cache()
            except Exception:
                logger.exception("Error shutting down session for workspace %s", ws_id)

        with self._lock:
            self._sessions.clear()
            self._graph_keys.clear()

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_locked(self, ws_id: str) -> GraphSession:
        """Create a new ``GraphSession`` for *ws_id*. Must be called under ``_lock``."""
        graph_key = self._graph_key_fn(ws_id)
        self._graph_keys[ws_id] = graph_key

        if self._session_factory is not None:
            session = self._session_factory(graph_key)
        else:
            # In-memory ContextGraph for each workspace.
            graph = ContextGraph(advanced_analytics=False)
            session = GraphSession(
                graph,
                provenance_storage_path=self._provenance_storage_path,
            )

        self._sessions[ws_id] = session
        logger.info(
            "Lazily initialized session for workspace %s (graph_key=%s)",
            ws_id,
            graph_key,
        )
        return session
