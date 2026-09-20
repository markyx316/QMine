"""QMine over the Model Context Protocol — the program as tools a chat app calls.

`qmine mcp` speaks MCP on stdio. A harness that bridges MCP tools (DeepSeek
Harness via `dsh-mcp-client`, or any other) then gets the whole program: inspect
raw exports, plan and build a pooled corpus, propose a run, and — the half that
only exists once the mining is done — ask questions about a finished study and
get cited, bounded, quote-guarded answers.
"""

from .authority import Authority
from .server import QMineServer, main

__all__ = ["Authority", "QMineServer", "main"]
