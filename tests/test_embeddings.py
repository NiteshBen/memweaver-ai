"""Tests for deterministic feature embedder & vector math."""

import math
import numpy as np
import pytest
from memweaver.embeddings import EmbeddingEngine


def test_embed_dimension_and_norm():
    dim = 128
    embedder = EmbeddingEngine(dimension=dim)
    vec = embedder.embed("Natural language agent memory item")
    assert len(vec) == dim
    norm = np.linalg.norm(vec)
    assert math.isclose(norm, 1.0, rel_tol=1e-4)


def test_embed_empty_string():
    embedder = EmbeddingEngine(dimension=64)
    vec = embedder.embed("")
    assert len(vec) == 64
    assert vec[0] == 1.0


def test_embed_batch():
    embedder = EmbeddingEngine(dimension=64)
    texts = ["First interaction", "Second interaction", "Third interaction"]
    batch = embedder.embed_batch(texts)
    assert len(batch) == 3
    assert all(len(v) == 64 for v in batch)


def test_cosine_similarity_identical_and_orthogonal():
    embedder = EmbeddingEngine(dimension=64)
    v1 = embedder.embed("Python multi-agent framework")
    v2 = embedder.embed("Python multi-agent framework")
    sim_identical = embedder.cosine_similarity(v1, v2)
    assert math.isclose(sim_identical, 1.0, rel_tol=1e-4)

    v3 = embedder.embed("Astronomical telescope optics")
    sim_diff = embedder.cosine_similarity(v1, v3)
    assert sim_diff < sim_identical


def test_cosine_similarity_empty():
    embedder = EmbeddingEngine()
    assert embedder.cosine_similarity([], []) == 0.0
    assert embedder.cosine_similarity([1.0], [1.0, 2.0]) == 0.0
