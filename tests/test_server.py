"""MCP-Server: Werkzeugnamen, Eingabeschemata, Host-Header-Schutz und leichter Start."""
import asyncio
import json
import subprocess
import sys

import pytest
from starlette.testclient import TestClient

from mcp_server.server import create_server

from hilfen import PROJEKT

WERKZEUGE = {
    # Name: (alle Parameter, Pflichtparameter)
    "search_papers": (
        {"query", "journal", "year_min", "year_max", "tags", "gap_status", "relevant_for", "top_k"},
        {"query"},
    ),
    "get_paper": ({"doi"}, {"doi"}),
    "find_similar": ({"doi", "section", "top_k"}, {"doi"}),
    "search_by_author": ({"author", "top_k"}, {"author"}),
    "annotate_paper": ({"doi", "gap_status", "tags", "relevant_for", "note"}, {"doi"}),
    "get_gap_landscape": ({"topic", "relevant_for", "gap_status"}, set()),
}


@pytest.fixture(scope="module")
def werkzeuge():
    return {t.name: t for t in asyncio.run(create_server().list_tools())}


def test_genau_die_sechs_werkzeuge_aus_dem_readme(werkzeuge):
    assert set(werkzeuge) == set(WERKZEUGE)


@pytest.mark.parametrize("name", sorted(WERKZEUGE))
def test_eingabeschema(werkzeuge, name):
    parameter, pflicht = WERKZEUGE[name]
    schema = werkzeuge[name].inputSchema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == parameter
    assert set(schema.get("required", [])) == pflicht
    # Jedes Werkzeug und jeder Parameter ist für das Sprachmodell beschrieben
    assert werkzeuge[name].description
    for param, definition in schema["properties"].items():
        assert definition.get("description"), f"{name}.{param} ohne Beschreibung"


def test_typen_und_standardwerte(werkzeuge):
    props = werkzeuge["search_papers"].inputSchema["properties"]
    assert props["query"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"
    assert props["top_k"]["default"] == 10
    assert {"type": "integer"} in props["year_min"]["anyOf"]
    assert {"type": "null"} in props["year_min"]["anyOf"]
    assert werkzeuge["find_similar"].inputSchema["properties"]["section"]["default"] == "abstract"
    assert werkzeuge["search_by_author"].inputSchema["properties"]["top_k"]["default"] == 20


# ---------------------------------------------------------------------------
# Host-Header-Schutz (DNS-Rebinding) für die HTTP-Transporte
# ---------------------------------------------------------------------------

KOPF = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "pytest", "version": "0"}},
}


def anfrage(transport: str, host: str) -> int:
    """Statuscode einer Anfrage mit diesem Host-Header (je Aufruf ein neuer Server)."""
    server = create_server(port=8080)
    if transport == "sse":
        with TestClient(server.sse_app()) as client:
            # ohne session_id: 400 nach bestandener Prüfung, 421 bei abgelehntem Host
            return client.post("/messages/", headers={**KOPF, "host": host}, json={}).status_code
    with TestClient(server.streamable_http_app()) as client:
        return client.post("/mcp", headers={**KOPF, "host": host}, json=INITIALIZE).status_code


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_fremder_host_wird_abgelehnt(monkeypatch, transport):
    monkeypatch.delenv("PBPK_MCP_ALLOW_ANY_HOST", raising=False)
    assert anfrage(transport, "angreifer.example") == 421


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
@pytest.mark.parametrize("host", ["127.0.0.1:8080", "localhost:8080"])
def test_lokaler_host_wird_zugelassen(monkeypatch, transport, host):
    monkeypatch.delenv("PBPK_MCP_ALLOW_ANY_HOST", raising=False)
    status = anfrage(transport, host)
    assert status != 421
    assert status == (400 if transport == "sse" else 200)


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_fremder_host_mit_allow_any_host(monkeypatch, transport):
    monkeypatch.setenv("PBPK_MCP_ALLOW_ANY_HOST", "1")
    assert anfrage(transport, "angreifer.example") == (400 if transport == "sse" else 200)


def test_andere_werte_der_variable_schalten_den_schutz_nicht_ab(monkeypatch):
    monkeypatch.setenv("PBPK_MCP_ALLOW_ANY_HOST", "true")
    assert anfrage("sse", "angreifer.example") == 421


# ---------------------------------------------------------------------------
# Start: ohne Qdrant, ohne Embedding-Modell
# ---------------------------------------------------------------------------

def test_start_laedt_weder_modell_noch_qdrant_client():
    """create_server() importiert keine schweren Pakete; das geschieht erst beim Werkzeugaufruf."""
    code = (
        "import json, sys; sys.path.insert(0, '.');"
        "from mcp_server.server import create_server; create_server();"
        "schwer = ['torch', 'sentence_transformers', 'transformers', 'fastembed', 'onnxruntime', 'qdrant_client'];"
        "print(json.dumps([m for m in schwer if m in sys.modules]))"
    )
    ausgabe = subprocess.run([sys.executable, "-c", code], cwd=PROJEKT, capture_output=True,
                             text=True, check=True, timeout=120)
    assert json.loads(ausgabe.stdout.strip().splitlines()[-1]) == []


def test_stdio_initialize_und_list_tools():
    """Echter Start über stdio mit dem MCP-Client aus dem Paket mcp (kein Qdrant nötig)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def abfrage():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(PROJEKT / "mcp_server" / "server.py")],
            cwd=str(PROJEKT),
        )
        async with stdio_client(params) as (lesen, schreiben):
            async with ClientSession(lesen, schreiben) as sitzung:
                init = await sitzung.initialize()
                tools = await sitzung.list_tools()
                return init.serverInfo.name, {t.name for t in tools.tools}

    name, namen = asyncio.run(asyncio.wait_for(abfrage(), timeout=60))
    assert name == "pbpk-literature"
    assert namen == set(WERKZEUGE)


def test_stdio_werkzeugaufruf_mit_kompilierten_bibliotheken():
    """
    Erster Werkzeugaufruf über stdio, der qdrant_client und damit numpy importiert.
    Unter Windows blieb der Server dabei hängen, solange stdin in einem Thread
    gelesen wurde; main() lädt die Bibliotheken deshalb vor. annotate_paper mit
    ungültigem Status bricht vor jeder Verbindung zu Qdrant ab.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def aufruf():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(PROJEKT / "mcp_server" / "server.py")],
            cwd=str(PROJEKT),
        )
        async with stdio_client(params) as (lesen, schreiben):
            async with ClientSession(lesen, schreiben) as sitzung:
                await sitzung.initialize()
                ergebnis = await sitzung.call_tool(
                    "annotate_paper", {"doi": "10.5555/synth.002", "gap_status": "UNGUELTIG"}
                )
                return json.loads(ergebnis.content[0].text)

    antwort = asyncio.run(asyncio.wait_for(aufruf(), timeout=120))
    assert "Invalid gap_status" in antwort["error"]


@pytest.mark.parametrize("plattform, transport, vorgeladen", [
    ("win32", "stdio", True),
    ("win32", "sse", False),
    ("win32", "streamable-http", False),
    ("linux", "stdio", False),
])
def test_vorladen_nur_unter_windows_mit_stdio(monkeypatch, plattform, transport, vorgeladen):
    import mcp_server.server as server_modul
    from mcp.server.fastmcp import FastMCP

    aufrufe = []
    monkeypatch.setattr(server_modul, "_preload_libraries", lambda: aufrufe.append("vorgeladen"))
    monkeypatch.setattr(FastMCP, "run", lambda self, transport: aufrufe.append(f"run:{transport}"))
    monkeypatch.setattr(sys, "platform", plattform)
    monkeypatch.setattr(sys, "argv", ["server.py", "--transport", transport])

    server_modul.main()
    assert aufrufe == (["vorgeladen"] if vorgeladen else []) + [f"run:{transport}"]
