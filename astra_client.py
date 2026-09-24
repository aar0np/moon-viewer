"""
astra_client.py
---------------
Thin wrapper around astrapy (>= 2.0) for the Moon Viewer project.

One collection — "moon_features" — stores documents of the form:

    {
        "_id":         str,              # slug  e.g. "mare_tranquillitatis"
        "name":        str,              # human name
        "type":        str,              # "mare" | "crater" | "highland" | …
        "description": str,             # plain-text description
        "lat":         float,           # selenographic latitude  (−90 … 90)
        "lon":         float,           # selenographic longitude (−180 … 180)
        "diameter_km": float | None,    # feature diameter in km
        "source":      str,             # e.g. "USGS / IAU"
        "$vector":     list[float],     # 768-d embedding (image or text)
    }
"""

from __future__ import annotations

import os
from typing import Any, Optional

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import CollectionDefinition, CollectionVectorOptions

COLLECTION_NAME = "moon_features"
VECTOR_DIMENSION = 768


def _get_db():
    """Return an authenticated Astra DB Database object."""
    token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]
    keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")

    client = DataAPIClient(token)
    db = client.get_database(endpoint, keyspace=keyspace)
    return db


def get_or_create_collection():
    """
    Return the moon_features collection, creating it (with a cosine vector
    index of dimension 768) if it does not already exist.
    """
    db = _get_db()

    # Create collection definition
    collection_definition = CollectionDefinition(
        vector=CollectionVectorOptions(
            dimension=VECTOR_DIMENSION, metric=VectorMetric.DOT_PRODUCT
        ),
    )

    collection = db.create_collection(
        COLLECTION_NAME,
        definition=collection_definition,
    )
    return collection


def upsert_feature(doc: dict[str, Any]) -> None:
    """
    Insert or replace a moon feature document.  The document must include
    the "$vector" key with a 768-d embedding list.
    """
    collection = get_or_create_collection()
    collection.find_one_and_replace(
        filter={"_id": doc["_id"]},
        replacement=doc,
        upsert=True,
    )


def upsert_features(docs: list[dict[str, Any]]) -> None:
    """
    Bulk-upsert a list of feature documents using insert_many with
    ordered=False so partial batches don't abort on duplicate _id.
    """
    collection = get_or_create_collection()
    # astrapy 2.x insert_many does not natively upsert; delete-then-insert
    # for simplicity (seed data scenario, not high-write production path).
    ids = [d["_id"] for d in docs]
    collection.delete_many(filter={"_id": {"$in": ids}})
    collection.insert_many(docs)


def vector_search(
    query_vector: list[float],
    limit: int = 8,
    feature_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Return the top-`limit` most similar feature documents to `query_vector`.

    Optionally filter by `feature_type` (e.g. "mare", "crater").
    """
    collection = get_or_create_collection()
    filter_doc: dict[str, Any] = {}
    if feature_type:
        filter_doc["type"] = feature_type

    cursor = collection.find(
        filter=filter_doc,
        sort={"$vector": query_vector},
        limit=limit,
        include_similarity=True,
        projection={"$vector": False},  # don't return the raw vector bytes
    )
    return list(cursor)


def get_feature(feature_id: str) -> Optional[dict[str, Any]]:
    """Fetch a single feature by its slug _id."""
    collection = get_or_create_collection()
    return collection.find_one(
        filter={"_id": feature_id},
        projection={"$vector": False},
    )


def get_feature_vector(feature_id: str) -> Optional[list[float]]:
    """Return only the stored $vector for a feature, or None if not found."""
    collection = get_or_create_collection()
    doc = collection.find_one(
        filter={"_id": feature_id},
        projection={"$vector": True},
    )
    if doc is None:
        return None
    return doc.get("$vector")


def count_features() -> int:
    """Return the number of documents currently in the collection."""
    collection = get_or_create_collection()
    return collection.count_documents(filter={}, upper_bound=10_000)
