"""
Слой семантической близости на основе RuWordNet.
Работает напрямую с SQLite-базой RuWordNet — без ruwordnet и SQLAlchemy.
Адаптировано под реальную схему ruwordnet-2021.db:

  sense(id, name, lemma, synset_id)
  synset(id, title, definition, part_of_speech)
  hypernym_relation(hyponym_id, hypernym_id)
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Константы схемы (соответствуют ruwordnet-2021.db)
# ---------------------------------------------------------------------------

_SYN_TABLE = "synset"
_SYN_ID = "id"
_SYN_TITLE = "title"
_SYN_DEFINITION = "definition"
_SYN_POS = "part_of_speech"

_SENSE_TABLE = "sense"
_SENSE_LEMMA = "lemma"
_SENSE_NAME = "name"
_SENSE_SYNSET_ID = "synset_id"

_HYP_TABLE = "hypernym_relation"
_HYP_HYPONYM = "hyponym_id"
_HYP_HYPERNYM = "hypernym_id"


_POS_MAP = {
    # существительное
    "noun": "N", "n": "N", "сущ": "N", "существительное": "N",
    # глагол
    "verb": "V", "v": "V", "гл": "V", "глагол": "V",
    # прилагательное
    "adj": "A", "a": "A", "прил": "A", "прилагательное": "A",
    "adjective": "A", "adj.": "A",
    # наречие
    "adv": "ADV", "adverb": "ADV", "нар": "ADV", "наречие": "ADV", "adv.": "ADV",
    # числительное
    "num": "NUM", "numeral": "NUM", "числ": "NUM",
    # местоимение
    "pn": "PN", "pronoun": "PN", "мест": "PN",
    # предикатив
    "praedic": "PRAEDIC", "предикатив": "PRAEDIC",
}


@dataclass
class RuWordNetMatch:
    synset_id: Optional[str]
    synset_title: Optional[str]
    synset_definition: Optional[str]
    matched_sense_id: Optional[str]
    confidence: float


@dataclass
class _SynsetInfo:
    id: str
    title: str
    definition: str
    pos: str


# ---------------------------------------------------------------------------
# Поиск БД по умолчанию
# ---------------------------------------------------------------------------

def _find_default_db() -> Optional[str]:
    env = os.environ.get("RUWORDNET_DB")
    if env and Path(env).exists():
        return env
    home_dir = Path.home() / ".ruwordnet"
    if home_dir.exists():
        candidates = sorted(home_dir.glob("*.db"), reverse=True)
        if candidates:
            return str(candidates[0])
    return None


def inspect_ruwordnet_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            print(f"\n[{name}]")
            for col in conn.execute(f"PRAGMA table_info({name})"):
                print(f"  {col[1]:24s} {col[2]}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class RuWordNetSimilarity:

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = _find_default_db()
        if db_path is None or not Path(db_path).exists():
            raise FileNotFoundError(
                "RuWordNet DB не найдена. Укажите путь явно: "
                "RuWordNetSimilarity('путь/к/ruwordnet-2021.db'), "
                "или задайте RUWORDNET_DB, или положите файл в ~/.ruwordnet/."
            )

        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

        # Индекс lemma_lower -> [synset_id] и name_lower -> [synset_id].
        # Строим в памяти, потому что SQLite LOWER() не умеет кириллицу.
        self._lemma_index: dict[str, list[str]] = {}
        self._build_index()

        # Прочие кэши
        self._synset_cache: dict[str, Optional[_SynsetInfo]] = {}
        self._hypernyms_cache: dict[str, list[str]] = {}
        self._hyponyms_cache: dict[str, list[str]] = {}
        self._depth_cache: dict[str, int] = {}
        self._lcs_cache: dict[tuple[str, str], Optional[str]] = {}
        self._path_cache: dict[tuple[str, str], Optional[int]] = {}

    # ------------------------------------------------------------------
    # Индекс lemma/name -> synset_id
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        sql = f"SELECT {_SENSE_LEMMA} AS lemma, {_SENSE_NAME} AS name, {_SENSE_SYNSET_ID} AS sid FROM {_SENSE_TABLE}"
        for row in self._conn.execute(sql):
            sid = row["sid"]
            if not sid:
                continue
            for key in (row["lemma"], row["name"]):
                if not key:
                    continue
                k = str(key).strip().lower()
                if not k:
                    continue
                bucket = self._lemma_index.setdefault(k, [])
                if sid not in bucket:
                    bucket.append(sid)

    # ------------------------------------------------------------------
    # Работа с синсетами
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pos(pos: Optional[str]) -> Optional[str]:
        if pos is None:
            return None
        return _POS_MAP.get(pos.lower(), pos.upper()[:1] if pos else None)

    def _load_synset(self, synset_id: str) -> Optional[_SynsetInfo]:
        if synset_id in self._synset_cache:
            return self._synset_cache[synset_id]
        row = self._conn.execute(
            f"SELECT {_SYN_ID} AS id, {_SYN_TITLE} AS title, "
            f"{_SYN_DEFINITION} AS definition, {_SYN_POS} AS pos "
            f"FROM {_SYN_TABLE} WHERE {_SYN_ID} = ?",
            (synset_id,),
        ).fetchone()
        if row is None:
            self._synset_cache[synset_id] = None
            return None
        info = _SynsetInfo(
            id=row["id"],
            title=row["title"] or "",
            definition=row["definition"] or "",
            pos=(row["pos"] or "").strip(),
        )
        self._synset_cache[synset_id] = info
        return info

    def _get_word_synsets(
            self,
            word: str,
            pos: Optional[str] = None,
    ) -> list[_SynsetInfo]:
        """Все синсеты, в которые входит слово (с фильтром по POS, если задан)."""
        key = str(word).strip().lower()
        sids = self._lemma_index.get(key, [])
        result: list[_SynsetInfo] = []
        want_pos = self._normalize_pos(pos)

        for sid in sids:
            info = self._load_synset(sid)
            if info is None:
                continue

            if want_pos:
                db_pos = self._normalize_pos(info.pos) if info.pos else None
                # Если POS из БД не распознан — не отсекаем (permissive),
                # иначе сравниваем нормализованные формы.
                if db_pos is not None and db_pos != want_pos:
                    continue

            result.append(info)

        return result

    # ------------------------------------------------------------------
    # Привязка значения из БД к синсету RuWordNet
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        ta = set(a.lower().replace(",", " ").split())
        tb = set(b.lower().replace(",", " ").split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def match_sense_to_synset(
        self,
        word: str,
        gloss: str,
        pos: Optional[str] = None,
    ) -> RuWordNetMatch:
        """
        Привязывает SenseData(word, gloss, pos) к синсету RuWordNet.
        Среди синсетов, содержащих слово, выбираем тот, чьё definition
        ближе всего к gloss (Жаккар по токенам); title используется
        как fallback, если definition пустой.
        """
        candidates = self._get_word_synsets(word, pos)
        if not candidates:
            return RuWordNetMatch(None, None, None, None, 0.0)

        best: Optional[_SynsetInfo] = None
        best_score = -1.0
        for info in candidates:
            target_text = info.definition or info.title
            score = self._jaccard(gloss, target_text)
            if score > best_score:
                best_score = score
                best = info

        if best is None:
            return RuWordNetMatch(None, None, None, None, 0.0)

        return RuWordNetMatch(
            synset_id=best.id,
            synset_title=best.title,
            synset_definition=best.definition,
            matched_sense_id=None,
            confidence=max(0.0, best_score),
        )

    # ------------------------------------------------------------------
    # Граф гиперонимов (hypernym_relation)
    # ------------------------------------------------------------------

    def _hypernyms(self, synset_id: str) -> list[str]:
        if synset_id in self._hypernyms_cache:
            return self._hypernyms_cache[synset_id]
        sql = (
            f"SELECT {_HYP_HYPERNYM} AS h FROM {_HYP_TABLE} "
            f"WHERE {_HYP_HYPONYM} = ?"
        )
        rows = self._conn.execute(sql, (synset_id,)).fetchall()
        result = [r["h"] for r in rows]
        self._hypernyms_cache[synset_id] = result
        return result

    def _hyponyms(self, synset_id: str) -> list[str]:
        if synset_id in self._hyponyms_cache:
            return self._hyponyms_cache[synset_id]
        sql = (
            f"SELECT {_HYP_HYPONYM} AS h FROM {_HYP_TABLE} "
            f"WHERE {_HYP_HYPERNYM} = ?"
        )
        rows = self._conn.execute(sql, (synset_id,)).fetchall()
        result = [r["h"] for r in rows]
        self._hyponyms_cache[synset_id] = result
        return result

    def _all_ancestors(self, synset_id: str) -> set[str]:
        ancestors: set[str] = set()
        stack = [synset_id]
        while stack:
            cur = stack.pop()
            for h in self._hypernyms(cur):
                if h not in ancestors:
                    ancestors.add(h)
                    stack.append(h)
        return ancestors

    def _depth(self, synset_id: str) -> int:
        if synset_id in self._depth_cache:
            return self._depth_cache[synset_id]
        q: deque[tuple[str, int]] = deque([(synset_id, 0)])
        visited = {synset_id}
        while q:
            cur, d = q.popleft()
            hypers = self._hypernyms(cur)
            if not hypers:
                self._depth_cache[synset_id] = d + 1
                return d + 1
            for h in hypers:
                if h not in visited:
                    visited.add(h)
                    q.append((h, d + 1))
        self._depth_cache[synset_id] = 1
        return 1

    def _lcs(self, a: str, b: str) -> Optional[str]:
        key = (a, b) if a <= b else (b, a)
        if key in self._lcs_cache:
            return self._lcs_cache[key]
        common = (self._all_ancestors(a) | {a}) & (self._all_ancestors(b) | {b})
        if not common:
            self._lcs_cache[key] = None
            return None
        best = max(common, key=lambda s: self._depth(s))
        self._lcs_cache[key] = best
        return best

    def _shortest_path(self, a: str, b: str) -> Optional[int]:
        if a == b:
            return 0
        key = (a, b) if a <= b else (b, a)
        if key in self._path_cache:
            return self._path_cache[key]
        q: deque[tuple[str, int]] = deque([(a, 0)])
        visited = {a}
        while q:
            cur, d = q.popleft()
            if cur == b:
                self._path_cache[key] = d
                return d
            for nb in self._hypernyms(cur) + self._hyponyms(cur):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        self._path_cache[key] = None
        return None

    # ------------------------------------------------------------------
    # Меры сходства
    # ------------------------------------------------------------------

    def path_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        p = self._shortest_path(a, b)
        if p is None:
            return 0.0
        return 1.0 / (1.0 + p)

    def wup_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        lcs = self._lcs(a, b)
        if lcs is None:
            return 0.0
        d_lcs = self._depth(lcs)
        d_a = self._depth(a)
        d_b = self._depth(b)
        denom = d_a + d_b
        if denom == 0:
            return 0.0
        return (2.0 * d_lcs) / denom

    def lch_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        p = self._shortest_path(a, b)
        if p is None:
            return 0.0
        # Оценка максимальной глубины: максимум известных + запас.
        # Точная шкала для lch не критична, важен относительный порядок.
        max_depth = max(self._depth_cache.values(), default=15) + 2
        ratio = (p + 1.0) / (2.0 * max_depth)
        if ratio <= 0:
            return 0.0
        return max(0.0, -math.log(ratio))

    # ------------------------------------------------------------------
    # Удобный метод
    # ------------------------------------------------------------------

    def similarity_between(
        self,
        word_a: str, gloss_a: str, pos_a: Optional[str],
        word_b: str, gloss_b: str, pos_b: Optional[str],
        measure: str = "wup",
    ) -> tuple[float, RuWordNetMatch, RuWordNetMatch]:
        ma = self.match_sense_to_synset(word_a, gloss_a, pos_a)
        mb = self.match_sense_to_synset(word_b, gloss_b, pos_b)

        if ma.synset_id is None or mb.synset_id is None:
            return 0.0, ma, mb

        if measure == "path":
            sim = self.path_similarity(ma.synset_id, mb.synset_id)
        elif measure == "lch":
            sim = self.lch_similarity(ma.synset_id, mb.synset_id)
        else:
            sim = self.wup_similarity(ma.synset_id, mb.synset_id)

        return sim, ma, mb

    def close(self) -> None:
        self._conn.close()