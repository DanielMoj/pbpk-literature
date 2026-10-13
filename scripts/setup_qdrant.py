#!/usr/bin/env python3
"""
Creates the Qdrant collection with correct schema:
- Dense vectors: 1024-dim cosine (bge-large-en-v1.5)
- Sparse vectors: BM25 (term weights from fastembed, IDF computed by Qdrant)
- Payload indices on: doi, journal, year, authors, section_type, tags, gap_status, relevant_for

Usage:
    python scripts/setup_qdrant.py [--recreate]

    --recreate: Löscht die Collection falls sie existiert und legt sie neu an.
                ACHTUNG: Alle Daten gehen verloren!
"""
import argparse
import sys
from pathlib import Path

# Projekt-Root zum sys.path hinzufügen, damit config importierbar ist
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Modifier,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
)

from config.loader import settings


PAYLOAD_INDICES = {
    "doi":           PayloadSchemaType.KEYWORD,
    "journal":       PayloadSchemaType.KEYWORD,
    "year":          PayloadSchemaType.INTEGER,
    "authors":       PayloadSchemaType.KEYWORD,
    "section_type":  PayloadSchemaType.KEYWORD,
    "tags":          PayloadSchemaType.KEYWORD,
    "gap_status":    PayloadSchemaType.KEYWORD,
    "relevant_for":  PayloadSchemaType.KEYWORD,
}


def create_collection(
    client: QdrantClient,
    collection: str,
    recreate: bool = False,
    dense_size: int = 1024,  # bge-large-en-v1.5; Tests nutzen kleinere Attrappen
):
    """Create the papers collection with dense + sparse vectors."""
    exists = client.collection_exists(collection)

    if exists and not recreate:
        print(f"Collection '{collection}' existiert bereits. Nichts zu tun.")
        print("  -> Benutze --recreate um sie neu anzulegen (Datenverlust!).")
        return False

    if exists and recreate:
        print(f"Lösche existierende Collection '{collection}'...")
        client.delete_collection(collection)

    print(f"Erstelle Collection '{collection}'...")
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(
                size=dense_size,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,   # fastembed liefert nur den TF-Teil von BM25
            ),
        },
    )
    print(f"  OK Collection erstellt (dense: {dense_size}-dim cosine, sparse: BM25 mit IDF)")
    return True


def create_payload_indices(client: QdrantClient, collection: str):
    """Create payload indices for filtered search."""
    print("Erstelle Payload-Indizes...")
    for field_name, schema_type in PAYLOAD_INDICES.items():
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=schema_type,
        )
        print(f"  OK {field_name} ({schema_type.value})")


def main():
    parser = argparse.ArgumentParser(description="Setup Qdrant collection for pbpk-literature")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection (ALL DATA WILL BE LOST)",
    )
    args = parser.parse_args()

    cfg = settings.qdrant
    print(f"Verbinde mit Qdrant @ {cfg.host}:{cfg.port}...")

    client = QdrantClient(host=cfg.host, port=cfg.port)

    # Verbindung testen
    try:
        client.get_collections()
    except Exception as e:
        print(f"FEHLER: Qdrant nicht erreichbar ({e})")
        print("  -> Ist 'docker compose up -d' gelaufen?")
        sys.exit(1)

    created = create_collection(client, cfg.collection, recreate=args.recreate)

    if created:
        create_payload_indices(client, cfg.collection)

    # Status ausgeben
    info = client.get_collection(cfg.collection)
    print(f"\nCollection '{cfg.collection}' Status:")
    print(f"  Points:         {info.points_count}")
    print(f"  Payload-Indizes: {list(PAYLOAD_INDICES.keys())}")
    print(f"  Status:         {info.status}")


if __name__ == "__main__":
    main()
