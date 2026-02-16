"""Core engine: manages a collection of beliefs and their compatibility matrix."""

from __future__ import annotations

import numpy as np

from cache import RateLimiter, ResultCache
from models import Belief, TensionCategory, TensionResult

MAX_BELIEFS = 50
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 128
TENSION_MODEL = "claude-sonnet-4-5-20250929"

TENSION_SYSTEM_PROMPT = """\
You are a formal logic and philosophy expert. Compare two beliefs. \
Determine if they are:

1. Mutually Entailed — One necessitates the other.
2. Compatible/Harmonious — They support the same worldview.
3. Neutral — Unrelated.
4. Tensioned — Difficult to hold both without cognitive dissonance.
5. Contradictory — Logically impossible to hold both.

Output ONLY valid JSON (no markdown fences) with exactly these keys:
{
  "score": <float from -1.0 (Contradictory) to +1.0 (Entailed)>,
  "category": <one of "mutually_entailed", "compatible_harmonious", \
"neutral", "tensioned", "contradictory">,
  "justification": "<one-sentence justification for the score>"
}

Score guide:
  +1.0       mutually_entailed
  +0.5..+0.9 compatible_harmonious
   0.0       neutral
  -0.5..-0.1 tensioned
  -1.0..-0.6 contradictory\
"""


class BeliefMap:
    """Holds up to 50 beliefs and a 50x50 compatibility-score matrix.

    The matrix is symmetric: ``scores[i][j] == scores[j][i]``.
    Diagonal entries are always 1.0 (a belief is fully compatible with itself).
    Unset pairs default to ``np.nan``.
    """

    def __init__(self) -> None:
        self.beliefs: dict[int, Belief] = {}
        self.scores: np.ndarray = np.full(
            (MAX_BELIEFS, MAX_BELIEFS), np.nan, dtype=np.float64
        )
        np.fill_diagonal(self.scores, 1.0)

        self.similarity: np.ndarray = np.full(
            (MAX_BELIEFS, MAX_BELIEFS), np.nan, dtype=np.float64
        )
        np.fill_diagonal(self.similarity, 1.0)

        # Optional — set by the CLI layer or caller.
        self.cache: ResultCache | None = None
        self.rate_limiter: RateLimiter | None = None

    # ------------------------------------------------------------------
    # Belief CRUD
    # ------------------------------------------------------------------

    def add_belief(self, text: str, expanded: str = "", embedding: list[float] | None = None) -> Belief:
        """Add a belief and return it.  Raises ValueError when the map is full."""
        if len(self.beliefs) >= MAX_BELIEFS:
            raise ValueError(f"Cannot exceed {MAX_BELIEFS} beliefs")

        next_id = self._next_id()
        belief = Belief(
            id=next_id,
            text=text,
            expanded=expanded,
            embedding=embedding or [],
        )
        self.beliefs[next_id] = belief
        return belief

    def remove_belief(self, belief_id: int) -> None:
        """Remove a belief and clear its row/column in the score matrix."""
        if belief_id not in self.beliefs:
            raise KeyError(f"Belief {belief_id} not found")
        del self.beliefs[belief_id]
        self.scores[belief_id, :] = np.nan
        self.scores[:, belief_id] = np.nan
        self.scores[belief_id, belief_id] = 1.0  # keep diagonal convention

    def get_belief(self, belief_id: int) -> Belief:
        if belief_id not in self.beliefs:
            raise KeyError(f"Belief {belief_id} not found")
        return self.beliefs[belief_id]

    def list_beliefs(self) -> list[Belief]:
        return sorted(self.beliefs.values(), key=lambda b: b.id)

    # ------------------------------------------------------------------
    # Compatibility scores
    # ------------------------------------------------------------------

    def set_score(self, id_a: int, id_b: int, score: float) -> None:
        """Set a symmetric compatibility score between two beliefs (-1..1)."""
        if not -1.0 <= score <= 1.0:
            raise ValueError("Score must be in [-1, 1]")
        for bid in (id_a, id_b):
            if bid not in self.beliefs:
                raise KeyError(f"Belief {bid} not found")
        self.scores[id_a, id_b] = score
        self.scores[id_b, id_a] = score

    def get_score(self, id_a: int, id_b: int) -> float:
        return float(self.scores[id_a, id_b])

    def scored_pairs(self) -> list[tuple[int, int, float]]:
        """Return all scored (non-NaN, non-diagonal) pairs."""
        pairs: list[tuple[int, int, float]] = []
        ids = sorted(self.beliefs)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                val = self.scores[a, b]
                if not np.isnan(val):
                    pairs.append((a, b, float(val)))
        return pairs

    # ------------------------------------------------------------------
    # Embeddings & similarity
    # ------------------------------------------------------------------

    def generate_embeddings(self, model: str = EMBEDDING_MODEL) -> int:
        """Embed every belief that lacks an embedding.

        Resolution order for each belief:
        1. Already has an in-memory embedding → skip.
        2. SQLite cache hit → restore without an API call.
        3. Cache miss → call OpenAI, then write to cache.

        Returns the total number of beliefs that received an embedding
        (cache hits + API calls).
        """
        missing = [b for b in self.list_beliefs() if not b.embedding]
        if not missing:
            return 0

        # --- phase 1: fill from cache --------------------------------
        api_needed: list[Belief] = []
        cache_hits = 0
        for b in missing:
            text = b.expanded or b.text
            if self.cache:
                cached = self.cache.get_embedding(text, model)
                if cached:
                    b.embedding = cached
                    cache_hits += 1
                    continue
            api_needed.append(b)

        if not api_needed:
            return cache_hits

        # --- phase 2: call OpenAI for the remainder ------------------
        from openai import OpenAI

        client = OpenAI()
        api_count = 0
        for i in range(0, len(api_needed), EMBEDDING_BATCH_SIZE):
            batch = api_needed[i : i + EMBEDDING_BATCH_SIZE]
            texts = [b.expanded or b.text for b in batch]
            response = client.embeddings.create(input=texts, model=model)
            for belief, item in zip(batch, response.data):
                belief.embedding = item.embedding
                if self.cache:
                    self.cache.put_embedding(
                        belief.expanded or belief.text, model, item.embedding
                    )
                api_count += 1

        return cache_hits + api_count

    def calculate_initial_similarity(self) -> np.ndarray:
        """Build a cosine-similarity matrix from belief embeddings.

        Only beliefs that already have an embedding are included.
        The result is written into ``self.similarity`` (a MAX_BELIEFS x
        MAX_BELIEFS matrix, NaN where no data exists) and also returned.
        """
        embedded = [b for b in self.list_beliefs() if b.embedding]
        if len(embedded) < 2:
            return self.similarity

        ids = np.array([b.id for b in embedded])
        E = np.array([b.embedding for b in embedded], dtype=np.float64)

        # L2-normalise each row, then cos_sim = E_norm @ E_norm.T
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E_norm = E / norms
        cos_sim = E_norm @ E_norm.T

        # Write the dense sub-matrix into the sparse full-size matrix
        self.similarity[np.ix_(ids, ids)] = cos_sim
        return self.similarity

    def interesting_pairs(
        self,
        top_n: int = 20,
        threshold: float = 0.7,
    ) -> list[tuple[int, int, float]]:
        """Return belief pairs that are worth a deep LLM comparison.

        A pair (a, b) is *interesting* when **either**:

        * Their cosine similarity >= *threshold* (semantic overlap), **or**
        * They share at least one thematic tag (``belief.tags``).

        Pairs below the threshold that share no tags are likely unrelated
        (e.g. "I like apples" vs. "I believe in the gold standard") and
        are skipped to conserve LLM tokens.

        Returns up to *top_n* pairs sorted by similarity descending.
        """
        ids = sorted(self.beliefs)
        pairs: list[tuple[int, int, float]] = []

        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                sim = float(self.similarity[a, b])

                # Condition 1: high semantic similarity
                above_threshold = not np.isnan(sim) and sim >= threshold

                # Condition 2: shared thematic tag
                tags_a = set(self.beliefs[a].tags)
                tags_b = set(self.beliefs[b].tags)
                shared_tag = bool(tags_a & tags_b)

                if above_threshold or shared_tag:
                    # Use similarity as sort key (NaN → -inf so tagged-only
                    # pairs without embeddings sink to the bottom).
                    sort_val = sim if not np.isnan(sim) else -float("inf")
                    pairs.append((a, b, sort_val))

        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:top_n]

    # ------------------------------------------------------------------
    # LLM logical-tension analysis
    # ------------------------------------------------------------------

    def analyze_logical_tension(
        self,
        belief_a: Belief,
        belief_b: Belief,
        model: str = TENSION_MODEL,
    ) -> TensionResult:
        """Ask Claude to judge the logical relationship between two beliefs.

        Resolution order:
        1. SQLite cache hit → return immediately (no API call).
        2. Cache miss → wait for rate-limiter, call Claude, write to cache.

        Returns a :class:`TensionResult` with ``score``, ``category``,
        and ``justification``.
        """
        import json as _json

        from anthropic import Anthropic

        desc_a = belief_a.expanded or belief_a.text
        desc_b = belief_b.expanded or belief_b.text

        # --- cache lookup --------------------------------------------
        if self.cache:
            cached = self.cache.get_tension(desc_a, desc_b, model)
            if cached:
                return cached

        # --- rate-limit then call ------------------------------------
        if self.rate_limiter:
            self.rate_limiter.wait()

        if not hasattr(self, "_anthropic_client"):
            self._anthropic_client = Anthropic()

        response = self._anthropic_client.messages.create(
            model=model,
            max_tokens=256,
            system=TENSION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Belief A: {desc_a}\nBelief B: {desc_b}",
                }
            ],
        )

        raw = response.content[0].text.strip()
        result = TensionResult(**_json.loads(raw))

        # --- write-through to cache ----------------------------------
        if self.cache:
            self.cache.put_tension(desc_a, desc_b, model, result)

        return result

    def analyze_pair(
        self,
        id_a: int,
        id_b: int,
        model: str = TENSION_MODEL,
    ) -> TensionResult:
        """Analyse a pair by ID and write the score into the matrix."""
        belief_a = self.get_belief(id_a)
        belief_b = self.get_belief(id_b)
        result = self.analyze_logical_tension(belief_a, belief_b, model=model)
        self.set_score(id_a, id_b, result.score)
        return result

    def analyze_interesting(
        self,
        top_n: int = 20,
        threshold: float = 0.7,
        model: str = TENSION_MODEL,
        force: bool = False,
    ) -> list[tuple[int, int, TensionResult]]:
        """Run tension analysis on interesting pairs.

        Pairs that already have a score in the matrix are skipped
        unless *force* is ``True``.  Even when a pair is *not* skipped,
        the cache may still satisfy it without an API call.

        Returns a list of ``(id_a, id_b, TensionResult)`` tuples.
        """
        candidates = self.interesting_pairs(top_n=top_n, threshold=threshold)
        results: list[tuple[int, int, TensionResult]] = []
        for id_a, id_b, _sim in candidates:
            if not force and not np.isnan(self.scores[id_a, id_b]):
                continue
            result = self.analyze_pair(id_a, id_b, model=model)
            results.append((id_a, id_b, result))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        """Return the lowest unused ID in [0, MAX_BELIEFS)."""
        used = set(self.beliefs)
        for i in range(MAX_BELIEFS):
            if i not in used:
                return i
        raise ValueError("No available IDs")  # unreachable if len check passes
