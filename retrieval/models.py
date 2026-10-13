"""
Pydantic models shared between retrieval and MCP server.
Paper, Chunk, SearchResult, Annotation, GapStatus enum.
"""
from enum import Enum
from pydantic import BaseModel


class GapStatus(str, Enum):
    REAL_GAP = "REAL_GAP"
    FAST_REAL_GAP = "FAST_REAL_GAP"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CROWDED = "CROWDED"
    UNASSESSED = "UNASSESSED"


class Annotation(BaseModel):
    gap_status: GapStatus = GapStatus.UNASSESSED
    relevant_for: list[str] = []      # ["P4", "B_pm", "NP3", ...]
    tags: list[str] = []              # ["misspecification", "TOST", "DDI"]
    note: str = ""


class PaperMetadata(BaseModel):
    doi: str
    title: str
    authors: list[str]
    journal: str
    year: int
    abstract: str = ""
    annotation: Annotation = Annotation()


class Chunk(BaseModel):
    chunk_id: str                     # f"{doi}___{section_index}"
    doi: str
    section_type: str                 # abstract, methods, results, discussion, ...
    text: str
    metadata: PaperMetadata


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    snippet: str                      # Relevantester Textausschnitt
