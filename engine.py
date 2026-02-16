"""Core engine: manages a collection of beliefs and their compatibility matrix."""

from __future__ import annotations

import logging
import re as _re
import time

import numpy as np

from cache import RateLimiter, ResultCache
from models import Belief, TensionCategory, TensionResult

logger = logging.getLogger(__name__)

MAX_BELIEFS = 50
EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "local-tfidf"
EMBEDDING_BATCH_SIZE = 128
TENSION_MODEL = "claude-sonnet-4-5-latest"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

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


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    m = _re.match(pattern, text, _re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


_SCORE_CATEGORY_RANGES: dict[str, tuple[float, float]] = {
    "mutually_entailed": (0.8, 1.0),
    "compatible_harmonious": (0.1, 1.0),
    "neutral": (-0.3, 0.3),
    "tensioned": (-1.0, -0.1),
    "contradictory": (-1.0, -0.4),
}


def _validate_score_category(score: float, category: str) -> None:
    """Log a warning if score and category are inconsistent."""
    expected = _SCORE_CATEGORY_RANGES.get(category)
    if expected and not (expected[0] <= score <= expected[1]):
        logger.warning(
            "Score %.2f is inconsistent with category '%s' (expected %.1f to %.1f)",
            score, category, expected[0], expected[1],
        )


def _local_tfidf_embeddings(texts: list[str], dims: int = 256) -> list[list[float]]:
    """Generate TF-IDF embeddings locally using scikit-learn.

    Uses TfidfVectorizer with SVD dimensionality reduction to produce
    dense vectors that capture term importance and handle stop words.
    Falls back to BOW if scikit-learn is not installed.
    """
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning(
            "scikit-learn not installed; falling back to local-bow embeddings. "
            "Install with: pip install scikit-learn"
        )
        return _local_bow_embeddings(texts, dims=128)

    n_texts = len(texts)
    if n_texts == 0:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Reduce dimensions with SVD; cap at available features
    n_components = min(dims, tfidf_matrix.shape[1], n_texts)
    if n_components < 2:
        # Too few documents/features for SVD; fall back to sparse-to-dense
        dense = tfidf_matrix.toarray()
        vectors = []
        for row in dense:
            norm = float(np.linalg.norm(row)) or 1.0
            vectors.append((row / norm).tolist())
        return vectors

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    # L2-normalize each row
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = reduced / norms
    return normalized.tolist()


def _local_bow_embeddings(texts: list[str], dims: int = 128) -> list[list[float]]:
    """Generate bag-of-words embeddings locally (no API or extra deps needed).

    Uses feature hashing to map tokens to a fixed-size vector space.
    The result is deterministic and produces meaningful cosine similarities
    for texts with shared words.
    """
    import hashlib

    def tokenize(t: str) -> list[str]:
        return _re.findall(r"[a-z']+", t.lower())

    token_lists = [tokenize(t) for t in texts]
    vectors: list[list[float]] = []
    for tokens in token_lists:
        vec = [0.0] * dims
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            bucket = h % dims
            sign = 1.0 if (h // dims) % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


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

    def add_belief(
        self,
        text: str,
        expanded: str = "",
        embedding: list[float] | None = None,
        tags: list[str] | None = None,
    ) -> Belief:
        """Add a belief and return it.

        Raises ValueError when the map is full or when a belief with
        identical text already exists.
        """
        if len(self.beliefs) >= MAX_BELIEFS:
            raise ValueError(f"Cannot exceed {MAX_BELIEFS} beliefs")

        existing_texts = {b.text for b in self.beliefs.values()}
        if text in existing_texts:
            raise ValueError(f"Duplicate belief: {text!r}")

        next_id = self._next_id()
        belief = Belief(
            id=next_id,
            text=text,
            expanded=expanded,
            embedding=embedding or [],
            tags=tags or [],
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

        # --- phase 2: generate embeddings for the remainder ----------
        if model.startswith("local"):
            texts = [b.expanded or b.text for b in api_needed]
            if model == "local-tfidf":
                vectors = _local_tfidf_embeddings(texts)
            else:
                vectors = _local_bow_embeddings(texts)
            for belief, vec in zip(api_needed, vectors):
                belief.embedding = vec
                if self.cache:
                    self.cache.put_embedding(
                        belief.expanded or belief.text, model, vec
                    )
            return cache_hits + len(api_needed)

        # OpenAI API path
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
        2. Cache miss → wait for rate-limiter, call Claude with retries,
           write to cache.

        Returns a :class:`TensionResult` with ``score``, ``category``,
        and ``justification``.

        Raises ``RuntimeError`` if all retries are exhausted.
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

        # --- rate-limit then call with retries -----------------------
        if not hasattr(self, "_anthropic_client"):
            self._anthropic_client = Anthropic()

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                if self.rate_limiter:
                    self.rate_limiter.wait()

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

                # Check for truncated response
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason == "max_tokens":
                    raise ValueError(
                        "Response truncated (max_tokens reached); "
                        "JSON is likely incomplete"
                    )

                if not response.content:
                    raise ValueError("Empty response from Claude")

                raw = _strip_markdown_fences(response.content[0].text)
                result = TensionResult(**_json.loads(raw))

                # Warn on score/category mismatch (but don't reject)
                _validate_score_category(result.score, result.category.value)

                # --- write-through to cache --------------------------
                if self.cache:
                    self.cache.put_tension(desc_a, desc_b, model, result)

                return result

            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Attempt %d/%d failed for pair (%s, %s): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1, MAX_RETRIES,
                        belief_a.text[:30], belief_b.text[:30],
                        exc, delay,
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"All {MAX_RETRIES} attempts failed for pair "
            f"({belief_a.text!r}, {belief_b.text!r}): {last_error}"
        )

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
        on_progress: callable | None = None,
        persist_fn: callable | None = None,
    ) -> list[tuple[int, int, TensionResult]]:
        """Run tension analysis on interesting pairs.

        Pairs that already have a score in the matrix are skipped
        unless *force* is ``True``.  Even when a pair is *not* skipped,
        the cache may still satisfy it without an API call.

        Parameters
        ----------
        on_progress : optional callback invoked after each pair is processed.
        persist_fn : optional callback to save state incrementally (called
            every 10 pairs).

        Returns a list of ``(id_a, id_b, TensionResult)`` tuples.
        """
        candidates = self.interesting_pairs(top_n=top_n, threshold=threshold)
        to_analyze = [
            (id_a, id_b)
            for id_a, id_b, _sim in candidates
            if force or np.isnan(self.scores[id_a, id_b])
        ]

        results: list[tuple[int, int, TensionResult]] = []
        for i, (id_a, id_b) in enumerate(to_analyze):
            try:
                result = self.analyze_pair(id_a, id_b, model=model)
                results.append((id_a, id_b, result))
            except RuntimeError as exc:
                logger.warning("Skipping pair (%d, %d): %s", id_a, id_b, exc)

            if on_progress:
                on_progress()

            # Incremental persistence every 10 pairs
            if persist_fn and (i + 1) % 10 == 0:
                persist_fn()

        # Final persist for any remaining pairs
        if persist_fn and results:
            persist_fn()

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
