"""
Характеризационные тесты «башни» анти-галлюцинации (долг ревью: гейтов много, спеки нет).

Фиксируют ТЕКУЩЕЕ поведение детерминированных гейтов из docs/anti_hallucination_gates.md, чтобы
их совокупность стала проверяемой и башню можно было рефакторить без молчаливого дрейфа порогов.
Меняешь порог/логику гейта → падает соответствующий тест (решение становится осознанным).
"""
import itertools

from src.graph.semantic_signals import is_degenerate, is_paywall, _PaywallEmbed, _PAYWALL_POS
from src.graph.agent import _strip_ungrounded_urls

# 60 различных слов БЕЗ цифр (is_degenerate срезает цифры, иначе word0..word59 → одно «word»)
_VARIED = " ".join("".join(p) for p in itertools.islice(itertools.product("abcdefgh", repeat=2), 60))


# ── is_degenerate: ≥60 слов И доля уникальных < 0.15 ──────────────────────────
def test_degenerate_short_text_is_false():
    assert is_degenerate("sorry sorry sorry") is False           # < 60 слов → не судим


def test_degenerate_repeated_spam_is_true():
    assert is_degenerate("sorry " * 60) is True                  # 60 слов, uniq/total ≈ 0.017 < 0.15


def test_degenerate_varied_text_is_false():
    assert is_degenerate(_VARIED) is False                       # 60 РАЗНЫХ слов → доля 1.0


def test_degenerate_numbered_list_not_masked():
    # нумерация «1. 2. …» не должна раздувать долю уникальных и маскировать повтор
    assert is_degenerate(" ".join(f"{i}. Sorry" for i in range(60))) is True


# ── is_paywall: пустой текст → False (страж, без эмбеддера) ────────────────────
def test_paywall_empty_text_is_false():
    assert is_paywall("") is False
    assert is_paywall("   ") is False


# ── is_paywall: контрастивная механика (порог+маржа) на ИНЪЕКТИРОВАННОМ эмбеддере ──
def test_paywall_contrast_mechanic_offline():
    class _StubEmb:
        enabled = True

        def embed(self, t: str):
            # игрушечный 2D эмбеддинг: «paywall-носитель» → одна ось, прочее → ортогональная
            paywall = any(k in (t or "").lower() for k in ("подписк", "subscription", "abonn", "suscrip"))
            return [1.0, 0.0] if paywall else [0.0, 1.0]

    det = _PaywallEmbed(embedder=_StubEmb())
    assert det.fires(_PAYWALL_POS[0], threshold=0.58, margin=0.05) is True    # носитель стены
    assert det.fires("now playing, free to watch", threshold=0.58, margin=0.05) is False


# ── контраст-механика новых детекторов (error_page / media_playing) на стабе ───────────────
class _MembershipEmb:
    """Стаб: POS-сид → одна ось, NEG-сид → ортогональная (реальный эмбеддер их разделяет; крудовый
    keyword-стаб мог бы спутать «нажмите воспроизведение» с «воспроизводится»)."""

    enabled = True

    def __init__(self, pos, neg):
        self._pos, self._neg = set(pos), set(neg)

    def embed(self, t):
        if t in self._pos:
            return [1.0, 0.0]
        if t in self._neg:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_is_error_page_contrast():
    from src.graph.semantic_signals import _ContrastiveSignal, _ERROR_POS, _ERROR_NEG
    det = _ContrastiveSignal(_ERROR_POS, _ERROR_NEG, embedder=_MembershipEmb(_ERROR_POS, _ERROR_NEG))
    assert det.fires(_ERROR_POS[0], 0.55, 0.05) is True      # «404 not found» → ошибка
    assert det.fires(_ERROR_NEG[0], 0.55, 0.05) is False     # «now playing» → не ошибка


def test_is_media_playing_contrast():
    from src.graph.semantic_signals import _ContrastiveSignal, _MEDIA_POS, _MEDIA_NEG
    det = _ContrastiveSignal(_MEDIA_POS, _MEDIA_NEG, embedder=_MembershipEmb(_MEDIA_POS, _MEDIA_NEG))
    assert det.fires(_MEDIA_POS[1], 0.55, 0.05) is True       # «звук играет» → играет
    assert det.fires(_MEDIA_NEG[0], 0.55, 0.05) is False      # «paused» → не играет


# ── _strip_ungrounded_urls: домен не в выдаче → ссылку убираем ─────────────────
def test_grounded_markdown_link_kept():
    out = _strip_ungrounded_urls("см. [тут](https://example.com/p)", {"example.com"})
    assert "https://example.com/p" in out


def test_ungrounded_markdown_link_reduced_to_text():
    out = _strip_ungrounded_urls("см. [тут](https://fake.ru/p)", {"example.com"})
    assert "fake.ru" not in out and "тут" in out


def test_ungrounded_bare_url_removed():
    out = _strip_ungrounded_urls("зайди https://fake.ru/p сейчас", {"example.com"})
    assert "fake.ru" not in out


def test_grounded_bare_url_kept():
    out = _strip_ungrounded_urls("зайди https://example.com/p", {"example.com"})
    assert "https://example.com/p" in out
