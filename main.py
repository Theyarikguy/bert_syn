"""
Модифицированный main: комбинирует SBERT-косинус и RuWordNet similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --- ваши существующие импорты ---
from database.connection import SessionLocal
from database.database_models import Sense as SenseModel
from database.database_models import Synonym as SynonymModel
from database.database_models import Word
from mpnet_f import pos_compatible, EmbeddingCache, SBERTMatcher, CrossEncoderReranker, SenseRepository

# --- новые импорты ---
from nltk_f import RuWordNetSimilarity, RuWordNetMatch

# ---------------------------------------------------------------------------
# Доменные модели (без изменений, кроме добавления полей для similarity)
# ---------------------------------------------------------------------------

@dataclass
class SenseData:
    sense_id: int | str
    word: str
    gloss: str
    examples: list[str] = field(default_factory=list)
    pos: Optional[str] = None


@dataclass
class SynonymAlignment:
    synonym_word: str

    # обязательные (без дефолтов) — идут первыми
    sbert_chosen_sense: Optional[SenseData]
    sbert_score: float
    wn_chosen_sense: Optional[SenseData]
    wn_score: float

    # необязательные (с дефолтами) — идут после
    sbert_ranked: list[tuple[SenseData, float]] = field(default_factory=list)
    wn_ranked: list[tuple[SenseData, float]] = field(default_factory=list)
    wn_match_target: Optional[RuWordNetMatch] = None
    wn_match_by_candidate: dict[int | str, Optional[RuWordNetMatch]] = field(default_factory=dict)
    note: Optional[str] = None

@dataclass
class SenseAlignmentResult:
    target_word: str
    target_sense: SenseData
    synonym_alignments: list[SynonymAlignment]


@dataclass
class PipelineConfig:
    gloss_weight: float = 1.0
    example_weight: float = 0.3
    ambiguity_gap: float = 0.04
    top_k_rerank: int = 3
    pos_filter: bool = True
    # --- новые параметры ---
    wn_measure: str = "wup"            # "path" | "wup" | "lch"
    wn_weight: float = 0.4             # вес RuWordNet в итоговой оценке
    sbert_weight: float = 0.6          # вес SBERT
    wn_min_confidence: float = 0.15    # порог уверенности привязки к синсету


# ---------------------------------------------------------------------------
# Вспомогательная функция комбинирования
# ---------------------------------------------------------------------------

def combine_scores(
    sbert: float,
    wn: float,
    cfg: PipelineConfig,
) -> float:
    """
    Комбинирует SBERT и RuWordNet similarity.
    Если wn == 0 (не удалось привязать), используется только SBERT.
    Иначе — взвешенная сумма.
    """
    if wn <= 0.0:
        return sbert
    return cfg.sbert_weight * sbert + cfg.wn_weight * wn


# ---------------------------------------------------------------------------
# Главная функция (адаптирована под два сигнала)
# ---------------------------------------------------------------------------

def align_synonyms_for_word(
    target_word: str,
    repo,
    matcher,
    wn_sim: RuWordNetSimilarity,
    reranker=None,                      # оставлен в сигнатуре для совместимости, не используется
    cfg: Optional[PipelineConfig] = None,
) -> list[SenseAlignmentResult]:
    cfg = cfg or PipelineConfig()
    target_senses = repo.get_senses(target_word)

    results: list[SenseAlignmentResult] = []
    for s in target_senses:
        synonym_words = repo.get_synonyms(target_word, s)

        syn_blocks: list[tuple[str, list[SenseData]]] = []
        for syn_word in synonym_words:
            cs = repo.get_senses(syn_word)
            if cfg.pos_filter:
                cs = [c for c in cs if pos_compatible(s.pos, c.pos)]
            syn_blocks.append((syn_word, cs))

        all_candidates = [c for _, cs in syn_blocks for c in cs]

        # ----- SBERT-оценки -----
        if all_candidates:
            v_s = matcher.encode_sense(s)
            v_all = matcher.encode_senses(all_candidates)
            all_sbert = (v_all @ v_s).tolist()
        else:
            all_sbert = []

        # ----- WN-привязка target и оценки кандидатов -----
        mt = wn_sim.match_sense_to_synset(s.word, s.gloss, s.pos)
        all_wn: list[float] = []
        mc_by_cand: dict[int | str, RuWordNetMatch] = {}
        for c in all_candidates:
            mc = wn_sim.match_sense_to_synset(c.word, c.gloss, c.pos)
            mc_by_cand[c.sense_id] = mc
            if mt.synset_id and mc.synset_id:
                if cfg.wn_measure == "path":
                    score = wn_sim.path_similarity(mt.synset_id, mc.synset_id)
                elif cfg.wn_measure == "lch":
                    score = wn_sim.lch_similarity(mt.synset_id, mc.synset_id)
                else:
                    score = wn_sim.wup_similarity(mt.synset_id, mc.synset_id)
            else:
                score = 0.0
            all_wn.append(score)

        # ----- Раздаём оценки по синонимам -----
        alignments: list[SynonymAlignment] = []
        cursor = 0
        for syn_word, cs in syn_blocks:
            n = len(cs)
            if n == 0:
                alignments.append(SynonymAlignment(
                    synonym_word=syn_word,
                    sbert_chosen_sense=None, sbert_score=0.0,
                    wn_chosen_sense=None, wn_score=0.0,
                    wn_match_target=mt,
                    note="в БД нет значений или все отсечены по POS",
                ))
                continue

            sbert_block = all_sbert[cursor:cursor + n]
            wn_block = all_wn[cursor:cursor + n]
            cands_block = cs
            cursor += n

            sbert_ranked = sorted(zip(cands_block, sbert_block), key=lambda x: -x[1])
            wn_ranked = sorted(zip(cands_block, wn_block), key=lambda x: -x[1])

            sb_best_sense, sb_best = sbert_ranked[0]
            wn_best_sense, wn_best = wn_ranked[0]

            # WN-выбор имеет смысл, только если хоть какая-то ненулевая близость
            if wn_best <= 0.0:
                wn_chosen = None
                wn_note = "нет ненулевой WN-близости ни с одним значением"
            else:
                wn_chosen = wn_best_sense
                wn_note = None

            alignments.append(SynonymAlignment(
                synonym_word=syn_word,
                sbert_chosen_sense=sb_best_sense,
                sbert_score=float(sb_best),
                sbert_ranked=sbert_ranked,
                wn_chosen_sense=wn_chosen,
                wn_score=float(wn_best),
                wn_ranked=wn_ranked,
                wn_match_target=mt,
                wn_match_by_candidate={c.sense_id: mc_by_cand.get(c.sense_id) for c in cands_block},
                note=wn_note,
            ))

        results.append(SenseAlignmentResult(
            target_word=target_word,
            target_sense=s,
            synonym_alignments=alignments,
        ))

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _fmt_sense(sense: Optional[SenseData]) -> str:
    if sense is None:
        return "—"
    return f"[{sense.sense_id}] ({sense.pos or '?'}) {sense.gloss}"


def main() -> None:
    target_word = "старый"

    cfg = PipelineConfig(wn_measure="wup", pos_filter=False)

    cache = EmbeddingCache("embeddings_cache.pkl")
    matcher = SBERTMatcher("all-mpnet-base-v2", cache=cache, config=cfg)
    wn_sim = RuWordNetSimilarity("ruwordnet/ruwordnet-2021.db")

    with SessionLocal() as db:
        repo = SenseRepository(db, lang="ru")
        results = align_synonyms_for_word(
            target_word, repo, matcher, wn_sim, None, cfg,
        )

    for r in results:
        # 1. Слово
        print(f"\n{'=' * 80}")
        print(f"СЛОВО: {r.target_word}")

        # 2. Значение слова
        ts = r.target_sense
        print(f"  ЗНАЧЕНИЕ: [{ts.sense_id}] ({ts.pos or '?'}) {ts.gloss}")

        if not r.synonym_alignments:
            print("    (нет синонимов для этого значения)")
            continue

        # 3. BERT: синоним → выбранное значение
        print("    BERT (SBERT-косинус по определениям):")
        for a in r.synonym_alignments:
            if a.sbert_chosen_sense is None:
                reason = a.note or "нет кандидатов"
                print(f"      {a.synonym_word:<40s} → —  ({reason})")
            else:
                ch = a.sbert_chosen_sense
                print(f"      {a.synonym_word:<40s} → "
                      f"[{ch.sense_id}] ({ch.pos or '?'}) {ch.gloss}")
                print(f"      {'':<40s}   score={a.sbert_score:.4f}")

        # 4. WordNet similarity: синоним → выбранное значение
        print("    WordNet similarity "
              f"({cfg.wn_measure} по RuWordNet):")
        for a in r.synonym_alignments:
            if a.wn_chosen_sense is None:
                if a.wn_match_target and a.wn_match_target.synset_id is None:
                    reason = "target-значение не привязалось к синсету RuWordNet"
                else:
                    reason = a.note or "нет ненулевой WN-близости"
                print(f"      {a.synonym_word:<40s} → —  ({reason})")
            else:
                ch = a.wn_chosen_sense
                print(f"      {a.synonym_word:<40s} → "
                      f"[{ch.sense_id}] ({ch.pos or '?'}) {ch.gloss}")
                print(f"      {'':<40s}   score={a.wn_score:.4f}")

    cache.save()


if __name__ == "__main__":
    main()