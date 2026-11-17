from nltk_f import RuWordNetSimilarity

wn = RuWordNetSimilarity("ruwordnet/ruwordnet-2021.db")

# 1. Индекс должен найти слово
print("synsets for 'сильный':", [s.title for s in wn._get_word_synsets("сильный")])

# 2. Привязка target-значения
m = wn.match_sense_to_synset(
    "сильный",
    "обладающий силой, способный преодолевать значительное сопротивление, "
    "выдерживать значительные нагрузки или совершать тяжёлую работу, мощный",
    "adjective",
)
print("target match:", m)

# 3. WN-близость с кандидатом
sim, mt, mc = wn.similarity_between(
    "сильный",
    "обладающий силой, способный преодолевать значительное сопротивление, "
    "выдерживать значительные нагрузки или совершать тяжёлую работу, мощный",
    "adjective",
    "мощный",
    "обладающий большой мощностью в значении «физическая величина, "
    "измеряемая отношением работы к промежутку времени»",
    "adjective",
    measure="wup",
)
print(f"wup={sim:.4f}")
print(f"  target -> {mt.synset_title!r} ({mt.confidence:.2f})")
print(f"  cand   -> {mc.synset_title!r} ({mc.confidence:.2f})")
for s in wn._get_word_synsets("сильный"):
    print(s.id, repr(s.pos), s.title)

print("normalize('adjective'):", repr(wn._normalize_pos("adjective")))
print("normalize('Adj'):", repr(wn._normalize_pos("Adj")))
print("no pos:      ", [(s.id, s.pos) for s in wn._get_word_synsets("сильный")])
print("with 'adjective':", [(s.id, s.pos) for s in wn._get_word_synsets("сильный", "adjective")])
m = wn.match_sense_to_synset(
    "сильный",
    "обладающий силой, способный преодолевать значительное сопротивление, "
    "выдерживать значительные нагрузки или совершать тяжёлую работу, мощный",
    "adjective",
)
print("target match:", m)

sim, mt, mc = wn.similarity_between(
    "сильный", "обладающий силой, ... мощный", "adjective",
    "мощный", "обладающий большой мощностью ...", "adjective",
    measure="wup",
)
print(f"wup={sim:.4f}")
print(f"  target -> {mt.synset_title!r}")
print(f"  cand   -> {mc.synset_title!r}")