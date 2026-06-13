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

    # ═══ РАСШИРЕНИЕ до 300+ (новые формулировки, шире языки) ═══
    # ── web_grounding ──
    ("how much is an iphone 17 right now", "web_grounding"),
    ("сколько сейчас стоит доллар к рублю", "web_grounding"),
    ("what time does the pharmacy on main street close", "web_grounding"),
    ("во сколько закрывается ближайший магнит", "web_grounding"),
    ("best vpn for streaming in 2026", "web_grounding"),
    ("какой банк даёт лучшую ставку по вкладу", "web_grounding"),
    ("cheapest flights from london to rome next week", "web_grounding"),
    ("куда поехать отдыхать в сентябре недорого", "web_grounding"),
    ("¿qué móvil tiene mejor batería este año?", "web_grounding"),
    ("dónde hay un cajero automático cerca", "web_grounding"),
    ("precio de las entradas para el concierto", "web_grounding"),
    ("welches notebook ist gut für studenten", "web_grounding"),
    ("wo finde ich das nächste fitnessstudio", "web_grounding"),
    ("aktueller goldpreis pro gramm", "web_grounding"),
    ("quel restaurant ouvre tard près d'ici", "web_grounding"),
    ("combien coûte un billet de train pour paris", "web_grounding"),
    ("quanto costa un volo per new york", "web_grounding"),
    ("dove comprare scarpe da corsa economiche", "web_grounding"),
    ("qual o melhor celular custo benefício", "web_grounding"),
    ("onde fica a farmácia mais próxima", "web_grounding"),
    ("najlepszy laptop do gier 2026", "web_grounding"),
    ("gdzie kupić tani bilet lotniczy", "web_grounding"),
    ("en iyi kablosuz kulaklık hangisi", "web_grounding"),
    ("waar kan ik goedkope sneakers kopen", "web_grounding"),
    ("what's trending on the news today", "web_grounding"),
    ("результаты вчерашнего матча спартака", "web_grounding"),
    ("when is the next apple keynote", "web_grounding"),
    ("какие фильмы идут в кино на этой неделе", "web_grounding"),
    ("top rated coffee machines under 300", "web_grounding"),
    ("сравни цены на эту модель в разных магазинах", "web_grounding"),
    ("is it going to rain this weekend in madrid", "web_grounding"),
    ("how to register a company in estonia", "web_grounding"),
    ("как поменять водительские права по истечении срока", "web_grounding"),
    ("documents needed to open a bank account", "web_grounding"),
    ("где сейчас можно посмотреть новый сезон сериала", "web_grounding"),
    ("which streaming service has the most movies", "web_grounding"),
    ("стоимость подписки ютуб премиум", "web_grounding"),
    ("recommend a hotel near the airport", "web_grounding"),
    ("посоветуй хорошего стоматолога в районе", "web_grounding"),
    ("what are the best noise cancelling earbuds", "web_grounding"),
    ("где дешевле заправиться рядом", "web_grounding"),
    ("current exchange rate euro to dollar", "web_grounding"),
    ("какая сейчас погода в сочи", "web_grounding"),
    ("address of the italian embassy", "web_grounding"),
    ("сколько идёт посылка из китая сейчас", "web_grounding"),
    ("best laptops for video editing this year", "web_grounding"),
    ("где купить билеты на самолёт подешевле", "web_grounding"),
    ("how much does netflix cost per month now", "web_grounding"),
    ("какой телевизор выбрать для игр", "web_grounding"),
    ("nearest gas station that is open", "web_grounding"),
    ("сколько стоит замена экрана на айфоне", "web_grounding"),
    ("what's the score of the lakers game", "web_grounding"),
    ("где можно срочно сделать фото на документы", "web_grounding"),
    ("cheapest place to buy a playstation 5", "web_grounding"),
    ("какие сейчас акции в пятёрочке", "web_grounding"),
    # ── physical_browser ──
    ("open amazon and show my recent orders", "physical_browser"),
    ("залогинься в мою почту на яндексе", "physical_browser"),
    ("add three of these to the basket and checkout", "physical_browser"),
    ("открой озон и положи это в корзину", "physical_browser"),
    ("go to my linkedin and open messages", "physical_browser"),
    ("зайди в мой втб онлайн", "physical_browser"),
    ("fill the shipping address field on the page", "physical_browser"),
    ("прокрути страницу вниз и нажми оплатить", "physical_browser"),
    ("inicia sesión en mi cuenta de amazon", "physical_browser"),
    ("añade esto al carrito y finaliza la compra", "physical_browser"),
    ("abre mi correo y revisa los mensajes", "physical_browser"),
    ("melde dich bei meinem konto an", "physical_browser"),
    ("lege das in den warenkorb und bezahle", "physical_browser"),
    ("klicke auf den weiter button", "physical_browser"),
    ("connecte-toi à mon compte et ouvre les commandes", "physical_browser"),
    ("ajoute ceci au panier et paie", "physical_browser"),
    ("accedi al mio account e apri i messaggi", "physical_browser"),
    ("entra na minha conta e abre os pedidos", "physical_browser"),
    ("zaloguj się na moje konto", "physical_browser"),
    ("hesabıma giriş yap ve siparişleri aç", "physical_browser"),
    ("open my google drive and find the file", "physical_browser"),
    ("войди в мой телеграм веб", "physical_browser"),
    ("click on the first search result on the page", "physical_browser"),
    ("нажми добавить в избранное на этом товаре", "physical_browser"),
    ("submit the contact form on the website", "physical_browser"),
    ("открой мои заказы в деливери", "physical_browser"),
    ("log into github and open my repositories", "physical_browser"),
    ("заполни форму регистрации на сайте", "physical_browser"),
    ("go to checkout and apply the promo code", "physical_browser"),
    ("отметь все товары и перейди к оформлению", "physical_browser"),
    ("open my paypal and check the balance", "physical_browser"),
    ("войди в личный кабинет мтс", "physical_browser"),
    ("scroll to the reviews section and expand them", "physical_browser"),
    ("нажми на кнопку скачать на странице", "physical_browser"),
    ("open the dropdown and select the largest size", "physical_browser"),
    ("выбери размер 42 и добавь в корзину", "physical_browser"),
    ("sign in to my spotify on the web", "physical_browser"),
    ("открой настройки моего профиля на сайте", "physical_browser"),
    ("type my query in the search box and submit", "physical_browser"),
    ("перейди на страницу оплаты и введи карту", "physical_browser"),
    ("open my email and reply to the last message", "physical_browser"),
    ("зайди в инстаграм и открой директ", "physical_browser"),
    ("upload this photo to my profile", "physical_browser"),
    ("нажми войти через гугл на сайте", "physical_browser"),
    ("open my calendar and create an event", "physical_browser"),
    ("открой корзину и удали последний товар", "physical_browser"),
    ("log in and renew my subscription", "physical_browser"),
    ("зайди в настройки аккаунта и смени пароль", "physical_browser"),
    ("open my bank and download last month statement", "physical_browser"),
    ("оформи возврат товара в личном кабинете", "physical_browser"),
    ("click accept cookies and continue", "physical_browser"),
    ("закрой попап и нажми продолжить", "physical_browser"),
    ("open the order page and track delivery", "physical_browser"),
    ("войди в аккаунт и поставь лайк", "physical_browser"),
    # ── play_media ──
    ("put on the new drake album", "play_media"),
    ("включи плейлист для бега", "play_media"),
    ("play the latest episode of my podcast", "play_media"),
    ("поставь что-нибудь спокойное для работы", "play_media"),
    ("play the music video for blinding lights", "play_media"),
    ("включи саундтрек из интерстеллара", "play_media"),
    ("pon la radio en línea", "play_media"),
    ("reproduce mi lista de reproducción favorita", "play_media"),
    ("spiele etwas entspannende musik", "play_media"),
    ("starte den neuen film auf netflix", "play_media"),
    ("mets le dernier clip de stromae", "play_media"),
    ("lance un podcast sur l'histoire", "play_media"),
    ("riproduci una canzone rilassante", "play_media"),
    ("toca uma música animada", "play_media"),
    ("włącz jakąś muzykę do nauki", "play_media"),
    ("biraz caz müziği çal", "play_media"),
    ("speel wat achtergrondmuziek af", "play_media"),
    ("play some white noise for sleeping", "play_media"),
    ("включи трансляцию матча", "play_media"),
    ("start the documentary about the ocean", "play_media"),
    ("поставь альбом на повтор", "play_media"),
    ("play classical music on youtube", "play_media"),
    ("включи мультик детям", "play_media"),
    ("play the next song in the queue", "play_media"),
    ("поставь видео с тренировкой", "play_media"),
    ("turn on some jazz in the background", "play_media"),
    ("включи радио шансон", "play_media"),
    ("play the trailer of the new marvel film", "play_media"),
    ("поставь песню которую слушали вчера", "play_media"),
    ("play live news stream", "play_media"),
    ("включи концерт queen на ютубе", "play_media"),
    ("play an audiobook about space", "play_media"),
    ("поставь медитацию на 10 минут", "play_media"),
    ("play my liked songs", "play_media"),
    ("включи новый сезон сериала", "play_media"),
    ("play something by hans zimmer", "play_media"),
    ("поставь фоном звуки дождя", "play_media"),
    ("play a workout video on youtube", "play_media"),
    ("включи последний выпуск новостей видео", "play_media"),
    ("start streaming the football game", "play_media"),
    # ── self_contained ──
    ("what is the integral of cosine x", "self_contained"),
    ("посчитай площадь круга радиусом 5", "self_contained"),
    ("explain the difference between processes and threads", "self_contained"),
    ("напиши функцию для проверки палиндрома", "self_contained"),
    ("how does garbage collection work in java", "self_contained"),
    ("в чём разница между sql и nosql", "self_contained"),
    ("translate thank you very much into german", "self_contained"),
    ("переведи фразу я тебя люблю на испанский", "self_contained"),
    ("summarize the theory of relativity briefly", "self_contained"),
    ("реши систему уравнений x плюс y равно 10", "self_contained"),
    ("what does idempotent mean in rest apis", "self_contained"),
    ("напиши пример декоратора в питоне", "self_contained"),
    ("explain dependency injection simply", "self_contained"),
    ("придумай название для кофейни", "self_contained"),
    ("how many seconds are in a day", "self_contained"),
    ("объясни как работает индекс в базе данных", "self_contained"),
    ("hi can you assist me with something", "self_contained"),
    ("здравствуй чем можешь помочь", "self_contained"),
    ("what's the difference between == and === in js", "self_contained"),
    ("dame un ejemplo de polimorfismo", "self_contained"),
    ("erkläre den unterschied zwischen stack und heap", "self_contained"),
    ("convert 5 kilometers to miles", "self_contained"),
    ("напиши короткое стихотворение про море", "self_contained"),
    ("what year did world war two end", "self_contained"),
    ("сколько будет 256 умножить на 4", "self_contained"),
    ("explique ce qu'est une fonction pure", "self_contained"),
    ("give me three ideas for a birthday gift", "self_contained"),
    ("объясни что такое рекурсивный спуск", "self_contained"),
    ("what is the boiling point of water in celsius", "self_contained"),
    ("раздели число 1000 на 8", "self_contained"),
    ("write a haiku about the rain", "self_contained"),
    ("qual é a capital da austrália", "self_contained"),
    ("explain how dns resolution works", "self_contained"),
    ("придумай шутку про программистов", "self_contained"),
    ("how do you calculate compound interest", "self_contained"),
    ("объясни принцип единственной ответственности", "self_contained"),
    ("what is the difference between http and https", "self_contained"),
    ("переведи доброе утро на японский", "self_contained"),
    ("summarize what a blockchain is in two sentences", "self_contained"),
    ("реши квадратное уравнение x в квадрате минус 4", "self_contained"),
    ("what are the primary colors", "self_contained"),
    ("напиши sql запрос для выборки топ 10 по дате", "self_contained"),
    ("explain the cap theorem", "self_contained"),
    ("сколько планет в солнечной системе", "self_contained"),
    ("give me a mnemonic to remember the planets", "self_contained"),
    ("объясни разницу между авторизацией и аутентификацией", "self_contained"),
    ("what is a closure in javascript", "self_contained"),
    ("придумай метафору для рекурсии", "self_contained"),
    ("how to center a div in css", "self_contained"),
    ("напиши регулярное выражение для номера телефона", "self_contained"),
    ("what is the speed of light", "self_contained"),
    ("объясни как работает кэш процессора", "self_contained"),
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
