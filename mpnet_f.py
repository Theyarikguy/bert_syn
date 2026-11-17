"""
Пайплайн сопоставления значений синонимов со значениями исходного слова.
Модель: all-mpnet-base-v2 (SBERT), опционально — cross-encoder reranking.

Адаптировано под схему Word/Sense/Synonym.
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.database_models import Sense as SenseModel
from database.database_models import Synonym as SynonymModel
from database.database_models import Word


# ---------------------------------------------------------------------------
# 1. Доменные модели
# ---------------------------------------------------------------------------

@dataclass
class SenseData:
    """Одно значение слова (адаптировано под модель Sense)."""
    sense_id: int
    word: str
    gloss: str
    examples: list[str] = field(default_factory=list)  # в БД примеров нет
    pos: Optional[str] = None


@dataclass
class AlignmentResult:
    target_sense: SenseData
    chosen_sense: Optional[SenseData]
    score: float
    ranked_candidates: list[tuple[SenseData, float]]
    reranked: bool = False


@dataclass
class PipelineConfig:
    gloss_weight: float = 1.0
    example_weight: float = 0.3
    ambiguity_gap: float = 0.04
    top_k_rerank: int = 3
    pos_filter: bool = True

@dataclass
class SynonymAlignment:
    """Сопоставление одного конкретного синонима с целевым значением."""
    synonym_word: str
    chosen_sense: Optional[SenseData]
    score: float
    ranked_candidates: list[tuple[SenseData, float]]
    reranked: bool = False
    note: Optional[str] = None       # причина, если chosen_sense = None


@dataclass
class SenseAlignmentResult:
    """Результат выравнивания для одного значения исходного слова."""
    target_word: str
    target_sense: SenseData
    synonym_alignments: list[SynonymAlignment]


# ---------------------------------------------------------------------------
# 2. Слой доступа к данным (реальные запросы под нашу схему)
# ---------------------------------------------------------------------------

class SenseRepository:
    """
    Репозиторий над Word/Sense/Synonym.
    Внутри кэширует результаты get_senses/get_synonyms,
    чтобы не ходить в БД повторно для одного и того же слова.
    """

    def __init__(self, session: Session, lang: Optional[str] = 'ru'):
        self._session = session
        self._lang = lang
        self._sense_cache: dict[str, list[SenseData]] = {}
        self._synonym_cache: dict[int, list[str]] = {}

    def get_senses(self, word: str) -> list[SenseData]:
        """Все значения, у которых Word.title == word."""
        if word in self._sense_cache:
            return self._sense_cache[word]

        stmt = (
            select(SenseModel.id, Word.title, Word.pos, SenseModel.definition)
            .join(Word, Word.id == SenseModel.word_id)
            .where(Word.title == word)
            .order_by(SenseModel.id)
        )
        if self._lang is not None:
            stmt = stmt.where(Word.lang == self._lang)

        rows = self._session.execute(stmt).all()
        result = [
            SenseData(
                sense_id=sense_id,
                word=title,
                gloss=(definition or '').strip(),
                examples=[],
                pos=(pos.lower() if pos else None),
            )
            for sense_id, title, pos, definition in rows
        ]
        self._sense_cache[word] = result
        return result

    def get_synonyms(self, word: str, sense: SenseData) -> list[str]:
        """Синонимы, привязанные к данному sense_id."""
        if sense.sense_id in self._synonym_cache:
            return self._synonym_cache[sense.sense_id]

        stmt = (
            select(SynonymModel.synonym)
            .where(SynonymModel.sense_id == sense.sense_id)
            .distinct()
        )
        rows = self._session.scalars(stmt).all()
        result = [s for s in rows if s]
        self._synonym_cache[sense.sense_id] = result
        return result


# ---------------------------------------------------------------------------
# 3. Кэш эмбеддингов
# ---------------------------------------------------------------------------

class EmbeddingCache:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self._store: dict[str, np.ndarray] = {}
        if self.path and self.path.exists():
            with open(self.path, "rb") as f:
                self._store = pickle.load(f)

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._store.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._store[key] = vec

    def save(self) -> None:
        if self.path:
            with open(self.path, "wb") as f:
                pickle.dump(self._store, f)


# ---------------------------------------------------------------------------
# 4. SBERT-матчер
# ---------------------------------------------------------------------------

class SBERTMatcher:
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        config: Optional[PipelineConfig] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = cache or EmbeddingCache()
        self.cfg = config or PipelineConfig()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, self.model.get_sentence_embedding_dimension()),
                dtype=np.float32,
            )

        keys = [self._hash(t) for t in texts]
        missing_pos = [i for i, k in enumerate(keys) if self.cache.get(k) is None]

        if missing_pos:
            missing_texts = [texts[i] for i in missing_pos]
            vecs = self.model.encode(
                missing_texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for i, v in zip(missing_pos, vecs):
                self.cache.put(keys[i], v)

        return np.vstack([self.cache.get(k) for k in keys]).astype(np.float32)

    def encode_senses(self, senses: list[SenseData]) -> np.ndarray:
        if not senses:
            return np.zeros(
                (0, self.model.get_sentence_embedding_dimension()),
                dtype=np.float32,
            )

        all_texts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        for s in senses:
            block = [s.gloss] + s.examples
            spans.append((len(all_texts), len(all_texts) + len(block), len(s.examples)))
            all_texts.extend(block)

        vecs = self._encode_texts(all_texts)

        out = []
        for (start, end, n_ex) in spans:
            block = vecs[start:end]
            w = np.array(
                [self.cfg.gloss_weight] + [self.cfg.example_weight] * n_ex,
                dtype=np.float32,
            )[:, None]
            combined = (block * w).sum(axis=0) / float(w.sum())
            combined /= (np.linalg.norm(combined) + 1e-12)
            out.append(combined)
        return np.vstack(out)

    def encode_sense(self, sense: SenseData) -> np.ndarray:
        return self.encode_senses([sense])[0]


# ---------------------------------------------------------------------------
# 5. Cross-encoder reranker (опционально)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    @staticmethod
    def _sense_text(s: SenseData) -> str:
        if s.examples:
            return s.gloss + " " + " ".join(s.examples)
        return s.gloss

    def score(self, target: SenseData, candidates: list[SenseData]) -> list[float]:
        pairs = [[self._sense_text(target), self._sense_text(c)] for c in candidates]
        return self.model.predict(pairs).tolist()


# ---------------------------------------------------------------------------
# 6. Утилиты
# ---------------------------------------------------------------------------

def pos_compatible(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return True
    return a == b


# ---------------------------------------------------------------------------
# 7. Главный пайплайн
# ---------------------------------------------------------------------------

def align_synonyms_for_word(
    target_word: str,
    repo: SenseRepository,
    matcher: SBERTMatcher,
    reranker: Optional[CrossEncoderReranker] = None,
    cfg: Optional[PipelineConfig] = None,
) -> list[SenseAlignmentResult]:
    """
    Для каждого значения исходного слова:
      для каждого синонима этого значения независимо:
        среди всех значений синонима выбираем наиболее близкое
        к целевому значению.
    """
    cfg = cfg or matcher.cfg
    target_senses = repo.get_senses(target_word)

    results: list[SenseAlignmentResult] = []
    for s in target_senses:
        synonym_words = repo.get_synonyms(target_word, s)

        # 1. Собираем кандидатов отдельно по каждому синониму,
        #    чтобы потом раздать им оценки (пул общий, но бакеты раздельные).
        syn_blocks: list[tuple[str, list[SenseData]]] = []
        for syn_word in synonym_words:
            cs = repo.get_senses(syn_word)
            if cfg.pos_filter:
                cs = [c for c in cs if pos_compatible(s.pos, c.pos)]
            syn_blocks.append((syn_word, cs))

        # 2. Один общий батч кодирования на все значения всех синонимов
        #    целевого sense — быстрее, чем по одному вызову на синоним.
        all_candidates = [c for _, cs in syn_blocks for c in cs]
        if all_candidates:
            v_s = matcher.encode_sense(s)             # (dim,)
            v_all = matcher.encode_senses(all_candidates)  # (N, dim)
            all_scores = (v_all @ v_s).tolist()
        else:
            all_scores = []

        # 3. Раздаём оценки обратно по синонимам и выбираем лучшего
        #    кандидата независимо внутри каждого синонима.
        alignments: list[SynonymAlignment] = []
        cursor = 0
        for syn_word, cs in syn_blocks:
            n = len(cs)
            if n == 0:
                alignments.append(SynonymAlignment(
                    synonym_word=syn_word,
                    chosen_sense=None,
                    score=0.0,
                    ranked_candidates=[],
                    note="в БД нет значений или все отсечены по POS",
                ))
                continue

            scores = all_scores[cursor:cursor + n]
            cursor += n

            ranked = sorted(zip(cs, scores), key=lambda x: -x[1])

            # 4. Reranker применяем только к «спорным» случаям внутри
            #    одного синонима: когда его top-1 и top-2 слишком близки.
            reranked = False
            if reranker is not None and len(ranked) >= 2:
                gap = ranked[0][1] - ranked[1][1]
                if gap < cfg.ambiguity_gap:
                    head = ranked[: cfg.top_k_rerank]
                    ce_scores = reranker.score(s, [c for c, _ in head])
                    head = sorted(
                        zip([c for c, _ in head], ce_scores),
                        key=lambda x: -x[1],
                    )
                    ranked = head + ranked[cfg.top_k_rerank:]
                    reranked = True

            chosen, best_score = ranked[0]
            alignments.append(SynonymAlignment(
                synonym_word=syn_word,
                chosen_sense=chosen,
                score=best_score,
                ranked_candidates=ranked,
                reranked=reranked,
            ))

        results.append(SenseAlignmentResult(
            target_word=target_word,
            target_sense=s,
            synonym_alignments=alignments,
        ))

    return results

# ---------------------------------------------------------------------------
# 8. Точка входа
# ---------------------------------------------------------------------------

