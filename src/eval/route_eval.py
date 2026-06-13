"""
Статистическая оценка universal intent-роутера (src/intent.py) на РАЗМЕЧЕННОМ наборе.

>100 кейсов, мультиязычные, фразы НОВЫЕ (НЕ из _SEED — иначе train=test утечка): это честная
проверка ГЕНЕРАЛИЗАЦИИ кодбука, а не запоминания. Метрики: overall accuracy (+Wilson CI),
per-label recall, confusion-матрица, fallback-rate.

Семантика None (классификатор не уверен → caller берёт регэксп/рассуждение):
- для self_contained None ПРИЕМЛЕМ (→ обычное рассуждение, верный исход) — считаем «ок»;
- для floor-сигналов (web_grounding/physical/play) None = «промах в регэксп-fallback».

Запуск (база/seed-only — temp кодбук):
  AGENT_INTENT_CODEBOOK=/tmp/cb.json .venv/bin/python -m src.eval.route_eval
"""
from __future__ import annotations

from collections import Counter, defaultdict

from src.intent import get_router

# (query, expected_label) — НОВЫЕ формулировки, мультиязычные. Не копии _SEED.
CASES: list[tuple[str, str]] = [
    # ── web_grounding: свежие внешние факты (купить/цена/адрес/лучшие/новости/процедура) ──
    ("how much does a Tesla Model 3 cost in 2026", "web_grounding"),
    ("сколько стоит подписка на нетфликс сейчас", "web_grounding"),
    ("какой смартфон с лучшей камерой до 50 тысяч", "web_grounding"),
    ("where is the closest 24 hour grocery store", "web_grounding"),
    ("когда выходит новая часть гта", "web_grounding"),
    ("what's the weather in Tokyo tomorrow", "web_grounding"),
    ("recommend a good Italian restaurant downtown", "web_grounding"),
    ("куда сходить с детьми в выходные в питере", "web_grounding"),
    ("¿cuál es el mejor portátil para programar?", "web_grounding"),
    ("precio del oro hoy", "web_grounding"),
    ("dónde puedo ver la película gratis", "web_grounding"),
    ("wie viel kostet ein flug nach berlin", "web_grounding"),
    ("beste kopfhörer unter 200 euro", "web_grounding"),
    ("quel est le meilleur smartphone en 2026", "web_grounding"),
    ("où acheter des billets de train pas chers", "web_grounding"),
    ("latest reviews of the new macbook", "web_grounding"),
    ("how do I renew my driver license", "web_grounding"),
    ("как получить налоговый вычет за лечение", "web_grounding"),
    ("сравни тарифы мобильных операторов", "web_grounding"),
    ("what are the visa requirements for Japan", "web_grounding"),
    ("addresses of pharmacies open now near me", "web_grounding"),
    ("свежие новости про выборы", "web_grounding"),
    ("best budget headphones reddit 2026", "web_grounding"),
    ("какой пылесос лучше купить отзывы", "web_grounding"),
    ("current bitcoin to usd rate", "web_grounding"),
    ("migliori ristoranti vicino a me", "web_grounding"),
    ("onde comprar um notebook barato", "web_grounding"),
    ("что подарить маме на день рождения идеи", "web_grounding"),
    # ── physical_browser: действие в браузере под логином/визуал ──
    ("log into my gmail account", "physical_browser"),
    ("открой мой инстаграм", "physical_browser"),
    ("put these sneakers in my cart on amazon", "physical_browser"),
    ("оформи заказ в корзине на вайлдберриз", "physical_browser"),
    ("show me my order history", "physical_browser"),
    ("зайди в мой кабинет на госуслугах", "physical_browser"),
    ("fill out the registration form on this site", "physical_browser"),
    ("нажми кнопку войти на странице", "physical_browser"),
    ("inicia sesión en mi cuenta de banco", "physical_browser"),
    ("öffne mein konto auf der webseite", "physical_browser"),
    ("connecte-toi à ma boîte mail", "physical_browser"),
    ("scroll down and click the blue button", "physical_browser"),
    ("добавь в избранное этот товар на сайте", "physical_browser"),
    ("checkout and pay on the store website", "physical_browser"),
    ("открой вкладку с моим профилем", "physical_browser"),
    ("submit this form for me", "physical_browser"),
    ("войди в мой аккаунт спотифай", "physical_browser"),
    ("agrega esto al carrito y paga", "physical_browser"),
    ("klick auf den anmelden button", "physical_browser"),
    ("go to my youtube subscriptions page", "physical_browser"),
    ("отметь товар и перейди к оплате", "physical_browser"),
    ("open my bank dashboard and check balance", "physical_browser"),
    ("заполни поле поиска на сайте и нажми найти", "physical_browser"),
    ("log in and download my invoice", "physical_browser"),
    # ── play_media: воспроизведение музыки/видео/фильма ──
    ("put on some chill electronic music", "play_media"),
    ("включи последний альбом тейлор свифт", "play_media"),
    ("play the trailer for the new batman movie", "play_media"),
    ("поставь сериал друзья первую серию", "play_media"),
    ("play relaxing piano music on youtube", "play_media"),
    ("включи рок погромче", "play_media"),
    ("pon música para concentrarme", "play_media"),
    ("spiele den neuen song von adele", "play_media"),
    ("mets de la musique jazz", "play_media"),
    ("play my workout playlist", "play_media"),
    ("включи фильм интерстеллар", "play_media"),
    ("start playing some lo-fi beats", "play_media"),
    ("поставь подкаст про историю", "play_media"),
    ("play the next episode", "play_media"),
    ("включи клип на новую песню", "play_media"),
    ("reproduce una canción de queen", "play_media"),
    ("turn on some background music", "play_media"),
    ("включи что-нибудь весёлое", "play_media"),
    ("play that documentary about space", "play_media"),
    ("поставь музыку для сна", "play_media"),
    ("play live stream of the football match", "play_media"),
    ("включи аудиокнигу", "play_media"),
    # ── self_contained: знания/рассуждение (мат/код/объяснение/перевод/приветствие) ──
    ("what is the derivative of x squared", "self_contained"),
    ("посчитай сколько будет 15 процентов от 240", "self_contained"),
    ("объясни разницу между tcp и udp", "self_contained"),
    ("write a function to reverse a linked list", "self_contained"),
    ("how does a hash table work internally", "self_contained"),
    ("в чём смысл теоремы пифагора", "self_contained"),
    ("translate good morning into japanese", "self_contained"),
    ("переведи слово свобода на французский", "self_contained"),
    ("summarize the plot of romeo and juliet", "self_contained"),
    ("реши уравнение 2x плюс 5 равно 13", "self_contained"),
    ("what are the SOLID principles", "self_contained"),
    ("напиши регулярку для email", "self_contained"),
    ("explain big O notation with examples", "self_contained"),
    ("придумай метафору для машинного обучения", "self_contained"),
    ("how many days are in a leap year", "self_contained"),
    ("объясни как работает блокчейн на пальцах", "self_contained"),
    ("hello there how is it going", "self_contained"),
    ("привет можешь помочь мне с задачей", "self_contained"),
    ("what's the difference between let and const", "self_contained"),
    ("dame un ejemplo de recursión", "self_contained"),
    ("erkläre was eine API ist", "self_contained"),
    ("convert 100 fahrenheit to celsius", "self_contained"),
    ("напиши хайку про осень", "self_contained"),
    ("what is the capital of France", "self_contained"),
    ("раздели 144 на 12 в уме", "self_contained"),
    ("explique la photosynthèse simplement", "self_contained"),
    ("give me 5 tips for better sleep", "self_contained"),
    ("объясни принцип работы двигателя внутреннего сгорания", "self_contained"),
]


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def run() -> None:
    r = get_router()
    if not r.enabled:
        print("[route_eval] эмбеддер выключен — нечего оценивать (включи memory.embeddings).")
        return
    n = len(CASES)
    correct = classified = fallback = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_total: Counter = Counter()
    per_correct: Counter = Counter()
    print(f"\n{'='*90}\nROUTE-EVAL: {n} размеченных кейсов (мультиязычные, вне seed)\n{'='*90}")
    for q, exp in CASES:
        c = r.classify(q)
        pred = c["label"] if c else None
        per_total[exp] += 1
        confusion[exp][pred or "NONE"] += 1
        if pred is None:
            fallback += 1
            ok = (exp == "self_contained")  # None для self_contained безопасен (→рассуждение)
        else:
            classified += 1
            ok = (pred == exp)
        if ok:
            correct += 1
            per_correct[exp] += 1
    acc = correct / n
    lo, hi = _wilson(correct, n)
    print(f"\nOverall accuracy: {correct}/{n} = {acc:.1%}  [95% Wilson {lo:.1%}–{hi:.1%}]")
    print(f"Classified (не-None): {classified}/{n} = {classified/n:.0%}  ·  fallback(None): {fallback}/{n} = {fallback/n:.0%}")
    print("\nPer-label recall:")
    for lbl in ("web_grounding", "physical_browser", "play_media", "self_contained"):
        t = per_total[lbl]
        if t:
            print(f"  {lbl:18} {per_correct[lbl]}/{t} = {per_correct[lbl]/t:.0%}")
    print("\nConfusion (expected → predicted):")
    for exp in ("web_grounding", "physical_browser", "play_media", "self_contained"):
        row = ", ".join(f"{p}:{cnt}" for p, cnt in confusion[exp].most_common())
        print(f"  {exp:18} → {row}")
    print("=" * 90)


if __name__ == "__main__":
    run()
