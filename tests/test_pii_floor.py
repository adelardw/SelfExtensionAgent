"""Анти-PII пол (Thread 2c): пост-фильтр выдуманных контактов в ответе (email) + редакция
PII при коллективном промоушене. КЛЮЧЕВОЙ анти-регресс: числовые ответы (GAIA) НЕ трогаются."""
from src.improve.safety import strip_ungrounded_pii, redact_pii


def test_fabricated_email_stripped():
    ans = "Свяжитесь с менеджером по адресу sales@fakeshop.ru для заказа."
    grounded = "найди где купить наушники"  # email НЕ из находок → выдумка
    out = strip_ungrounded_pii(ans, grounded)
    assert "sales@fakeshop.ru" not in out
    assert "контакт удалён" in out


def test_grounded_email_kept():
    ans = "Твой рабочий email: jas@company.com — как ты и просил."
    grounded = "напомни мой email\n[Что я знаю о пользователе]\n- email: jas@company.com"
    out = strip_ungrounded_pii(ans, grounded)
    assert "jas@company.com" in out  # легитимный recall данных юзера — не режем


def test_numeric_answer_untouched():
    # GAIA-ответы часто числовые — пост-фильтр НЕ должен их трогать (нет '@').
    for ans in ("17", "12345678", "FINAL ANSWER: 42", "1 234 567", "+7 это не телефон а текст 89"):
        assert strip_ungrounded_pii(ans, "вопрос без контактов") == ans


def test_email_without_at_is_noop():
    ans = "ответ про погоду без всяких контактов"
    assert strip_ungrounded_pii(ans, "что по погоде") == ans


def test_redact_pii_masks_contacts():
    text = "пиши на ivan@mail.ru или звони +7 701 234 56 78, карта 1234 5678 9012 3456"
    out, n = redact_pii(text)
    assert "ivan@mail.ru" not in out
    assert "[PII]" in out
    assert n >= 2  # email + телефон/карта


def test_redact_pii_keeps_plain_numbers():
    # плотные числа (год/количество) НЕ телефон/карта — не маскируем агрессивно.
    text = "за 2025 год продано 17 единиц"
    out, n = redact_pii(text)
    assert "2025" in out and "17" in out and n == 0
