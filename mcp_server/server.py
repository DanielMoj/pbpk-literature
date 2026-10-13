#!/usr/bin/env python3
"""
MCP Server entry point for pbpk-literature.
Registers all tools, supports stdio, SSE and streamable HTTP transport.

Startup is light: Qdrant and the embedding models are only touched when a
tool is called for the first time. Only under Windows with stdio transport
are the libraries imported at start (see _preload_libraries).
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _transport_security() -> TransportSecuritySettings:
    """
    DNS-Rebinding-Schutz für die HTTP-Transporte (sse, streamable-http).

    Standard: nur Host-Header von localhost, 127.0.0.1 und [::1] (mit Port).
    Mit PBPK_MCP_ALLOW_ANY_HOST=1, etwa hinter einem Tunnel, wird jeder Host
    zugelassen; dann muss der Zugang anderweitig abgesichert sein.
    """
    if os.getenv("PBPK_MCP_ALLOW_ANY_HOST") == "1":
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
    )


def create_server(port: int = 8000) -> FastMCP:
    """Create the MCP server and register all tools."""
    server = FastMCP(
        "pbpk-literature",
        port=port,
        transport_security=_transport_security(),
    )

    from mcp_server.tools.search_papers import search_papers
    from mcp_server.tools.get_paper import get_paper
    server.tool()(search_papers)
    server.tool()(get_paper)

    from mcp_server.tools.annotate_paper import annotate_paper
    from mcp_server.tools.find_similar import find_similar
    from mcp_server.tools.search_by_author import search_by_author
    from mcp_server.tools.get_gap_landscape import get_gap_landscape
    server.tool()(annotate_paper)
    server.tool()(find_similar)
    server.tool()(search_by_author)
    server.tool()(get_gap_landscape)

    return server


def _preload_libraries() -> None:
    """
    Import the compiled libraries of the tools before the stdio transport starts.

    Under Windows the stdio transport reads stdin with a blocking call in a
    worker thread. A tool that loads a compiled extension for the first time
    (numpy via qdrant_client, torch via sentence-transformers) then hangs.
    Only the imports happen here: the embedding models are still loaded and
    Qdrant is still contacted on the first tool call.
    """
    import fastembed  # noqa: F401  BM25
    import sentence_transformers  # noqa: F401  torch, transformers
    import retrieval.search  # noqa: F401  qdrant_client, numpy


def main():
    parser = argparse.ArgumentParser(description="pbpk-literature MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",  # ASCII: stderr ist bei stdio eine Pipe
        datefmt="%H:%M:%S",
    )

    from config.loader import settings

    transport = args.transport or settings.mcp.transport
    port = args.port or settings.mcp.sse_port

    server = create_server(port=port)

    if transport == "stdio" and sys.platform == "win32":
        logger.info("Windows/stdio: lade Bibliotheken vor (Modelle und Qdrant erst beim ersten Aufruf)")
        _preload_libraries()

    tool_count = len(asyncio.run(server.list_tools()))
    logger.info(
        f"pbpk-literature MCP server starting "
        f"({tool_count} tools, transport={transport})"
    )

    server.run(transport=transport)


if __name__ == "__main__":
    main()
