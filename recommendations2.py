from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple, Optional
from math import log, sqrt

# =========================
# SECTION A — EDIT: MENU
# =========================
MENU_CATALOG: Dict[str, Set[str]] = {
    "Nasi Goreng Ayam":     {"rice","fried","malay","spicy","chicken"},
    "Nasi Goreng Cina":     {"rice","fried","chinese","chicken"},
    "Nasi Goreng Paprik":   {"rice","fried","thai","spicy","chicken"},
    "Char Kuey Teow":       {"noodle","fried","chinese","wok","prawn"},
    "Kuey Teow Soup":       {"noodle","soup","chinese","chicken"},
    "Laksa":                {"noodle","soup","spicy","malay","prawn"},
    "Tom Yam":              {"soup","thai","spicy","seafood"},
    "Nasi Lemak":           {"rice","malay","coconut","spicy","chicken"},
    "Chicken Rice":         {"rice","chinese","chicken"},
    "Ayam Masak Merah":     {"rice","malay","spicy","chicken"},
    "Rendang Daging":       {"rice","malay","spicy","beef"},
    "Nasi Goreng Seafood":  {"rice","fried","seafood","spicy"},
    "Nasi Goreng Kampung":  {"rice","fried","malay","spicy","anchovy"},
    "Mee Goreng":           {"noodle","fried","malay","spicy"},
    "Sambal Udang":         {"rice","malay","spicy","prawn"},
    "Ikan Bakar":           {"grill","fish","malay","spicy"},
    "Satay":                {"grill","malay","peanut","chicken"},
    "Soto":                 {"soup","malay","chicken"},
    "Teh Tarik":            {"drink","sweet","malay"},
    "Milo Dinosaur":        {"drink","sweet"},
}

TAG_WEIGHT: Dict[str, float] = {
    "malay": 1.1, "chinese": 1.1, "thai": 1.1,
    "rice": 1.0, "noodle": 1.0, "soup": 1.0, "fried": 1.0, "grill": 1.0, "wok": 1.0, "drink": 0.5,
    "spicy": 1.3, "sweet": 0.8, "coconut": 1.0, "peanut": 1.0, "anchovy": 1.0,
    "chicken": 1.2, "beef": 1.2, "fish": 1.2, "prawn": 1.2, "seafood": 1.2,
}

# =========================
# SECTION B — ORDERS
# =========================
# Past history: (user_id, item) pairs. Totals & popularity derive ONLY from here.
ORDERS: List[Tuple[str, str]] = [
    ("customer_1", "Nasi Goreng Ayam"),
    ("customer_1", "Nasi Lemak"),
    ("customer_1", "Char Kuey Teow"),
    ("customer_1", "Ayam Masak Merah"),


    ("customer_2", "Kuey Teow Soup"),
    ("customer_2", "Chicken Rice"),
    ("customer_2", "Char Kuey Teow"),

    ("customer_3", "Tom Yam"),
    ("customer_3", "Nasi Goreng Seafood"),
    ("customer_3", "Laksa"),

    ("customer_4", "Soto"),
    ("customer_4", "Chicken Rice"),
    ("customer_4", "Teh Tarik"),

    ("customer_5", "Nasi Goreng Kampung"),
    ("customer_5", "Nasi Goreng Ayam"),
    
    ("customer_6", "Satay"),
    ("customer_6", "Ikan Bakar"),
    
    ("customer_7", "Mee Goreng"),
    ("customer_7", "Nasi Lemak"),
    
    ("customer_8", "Rendang Daging"),
    ("customer_8", "Nasi Lemak"),
    
    ("customer_9", "Nasi Goreng Cina"),
    ("customer_9", "Chicken Rice"),
]

# =========================
# INTERNAL STORAGE (demo)
# =========================
USER_TAG_LIKES:  Dict[str, Counter] = defaultdict(Counter)  
USER_TAG_AVOIDS: Dict[str, Counter] = defaultdict(Counter) 
USER_ITEM_COUNTS: Dict[str, Counter] = defaultdict(Counter) 

# ---------- Bootstrapping ----------
def bootstrap_from_orders():
    """Build user profiles (likes, item counts) from ORDERS."""
    USER_TAG_LIKES.clear(); USER_TAG_AVOIDS.clear(); USER_ITEM_COUNTS.clear()
    for uid, item in ORDERS:
        if item not in MENU_CATALOG:
            continue
        USER_ITEM_COUNTS[uid][item] += 1
        for tag in MENU_CATALOG[item]:
            USER_TAG_LIKES[uid][tag] += TAG_WEIGHT.get(tag, 1.0)

def apply_manual_prefs(user_id: str, likes: Optional[Set[str]] = None, avoids: Optional[Set[str]] = None):
    for tag in likes or set():
        USER_TAG_LIKES[user_id][tag] += TAG_WEIGHT.get(tag, 1.0)
    for tag in avoids or set():
        USER_TAG_AVOIDS[user_id][tag] += TAG_WEIGHT.get(tag, 1.0)


# ---------- ID management ----------
def _all_user_ids() -> Set[str]:
    ids = {uid for uid, _ in ORDERS}
    ids.update(USER_ITEM_COUNTS.keys())
    ids.update(USER_TAG_LIKES.keys())
    ids.update(USER_TAG_AVOIDS.keys())
    return ids

def next_user_id(prefix: str = "u") -> str:
    max_n = 0
    for uid in _all_user_ids():
        if uid.startswith(prefix):
            try:
                n = int(uid[len(prefix):])
                if n > max_n: max_n = n
            except ValueError:
                continue
    return f"{prefix}{max_n + 1 if max_n > 0 else 1}"

def register_new_customer(likes: Optional[Set[str]] = None, avoids: Optional[Set[str]] = None, forced_id: Optional[str] = None) -> str:
    uid = forced_id or next_user_id()
    
    _ = USER_ITEM_COUNTS[uid]; _ = USER_TAG_LIKES[uid]; _ = USER_TAG_AVOIDS[uid]
    apply_manual_prefs(uid, likes, avoids)
    return uid

# ---------- Recording purchases ----------
def record_purchase(user_id: str, item: str):
    if item not in MENU_CATALOG:
        raise ValueError(f"Unknown menu item: {item}")
    ORDERS.append((user_id, item))
    USER_ITEM_COUNTS[user_id][item] += 1
    for tag in MENU_CATALOG[item]:
        USER_TAG_LIKES[user_id][tag] += TAG_WEIGHT.get(tag, 1.0)

# ---------- Scoring helpers ----------
def favorite_bonus(user_id: str, item: str) -> float:
    cnt = USER_ITEM_COUNTS[user_id].get(item, 0)
    return 0.2 * log(1 + cnt) if cnt > 0 else 0.0

def score_item(user_id: str, item: str):
    tags = MENU_CATALOG[item]
    contribs = []
    raw = 0.0
    for tag in tags:
        like  = USER_TAG_LIKES[user_id].get(tag, 0.0)
        avoid = USER_TAG_AVOIDS[user_id].get(tag, 0.0)
        val = like - 2.0*avoid
        c = val * TAG_WEIGHT.get(tag, 1.0)
        if c > 0:
            contribs.append((tag, c))
        raw += c
    norm = raw / max(1.0, sqrt(len(tags)))
    s = norm + favorite_bonus(user_id, item)
    contribs.sort(key=lambda x: x[1], reverse=True)
    return s, [t for t,_ in contribs[:3]]

def global_top_n(n: int = 5):
    gc = Counter(item for _u, item in ORDERS)
    return gc.most_common(n)

# ---------- Recommender ----------
def recommend(user_id: str, top_n: int = 5, include_already_bought: bool = True):
    knows_purchases = len(USER_ITEM_COUNTS[user_id]) > 0
    knows_prefs     = len(USER_TAG_LIKES[user_id]) > 0 or len(USER_TAG_AVOIDS[user_id]) > 0

    if not knows_purchases and not knows_prefs:
        return [
            {"item": it, "score": float(cnt), "reason": "popular_overall", "evidence": {"global_count": cnt}}
            for it, cnt in global_top_n(top_n)
        ]

    candidates = list(MENU_CATALOG.keys())
    if not include_already_bought:
        bought = set(USER_ITEM_COUNTS[user_id].keys())
        candidates = [it for it in candidates if it not in bought]

    results = []
    for it in candidates:
        score, top_tags = score_item(user_id, it)
        if score <= 0:
            continue
        reason = "your_favorite" if USER_ITEM_COUNTS[user_id].get(it, 0) > 0 else "pref_match"
        results.append({"item": it, "score": round(score, 3), "reason": reason, "evidence": {"matched_tags": top_tags}})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]

SHOW_SCORES = False  

def _reason_line(r: dict) -> str:
    if r["reason"] == "your_favorite":
        return f"{r['item']} — because you’ve ordered it before."
    elif r["reason"] == "pref_match":
        tags = ", ".join(r["evidence"].get("matched_tags", []))
        return f"{r['item']} — matches your preferences ({tags})." if tags else f"{r['item']} — matches your preferences."
    elif r["reason"] == "popular_overall":
        cnt = r["evidence"].get("global_count", None)
        return f"{r['item']} — one of our most popular dishes{f' (x{cnt})' if cnt is not None else ''}."
    else:
        return f"{r['item']}"

def print_recs(user_id: str, top_n: int = 5, include_already_bought: bool = True):
    recs = recommend(user_id, top_n=top_n, include_already_bought=include_already_bought)
    for r in recs:
        if SHOW_SCORES:
            print(f"  {r['item']:24s} score={r['score']:.3f} reason={r['reason']:12s} evidence={r['evidence']}")
        else:
            print("  " + _reason_line(r))

def print_top(n: int = 5, title: str = "Top"):
    print(f"\n{title}:")
    for item, cnt in global_top_n(n):
        print(f"  {item:24s} x{cnt}")


# ======================
# if __name__ == "__main__":
#     # 1) Build profiles from Section B (ORDERS)
#     bootstrap_from_orders()

#     # 2) Show current popularity (derived ONLY from ORDERS)
#     # print_top(3, title="Top 3 most ordered food")

#     # 3) Existing user demo (change ID to any customer_1..customer_9 from ORDERS)
#     existing_user = "customer_1"
#     print(f"\nExisting user {existing_user}:")
#     print_recs(existing_user, top_n=3)

    # 4) Register multiple new customers (auto customer_10, customer_11, ...)
    # u_new_1 = register_new_customer(likes={"spicy","chinese","chicken"}, avoids={"beef"})
    # u_new_2 = register_new_customer(likes={"noodle","thai","spicy"})
    # print(f"\nRegistered new customers: {u_new_1}, {u_new_2}")

    # 5) Recommend for them (no history yet -> pref_match; if no prefs -> popular_overall)
    # print(f"\nRecommendations for {u_new_1}:")
    # print_recs(u_new_1, top_n=5)
    # print(f"\nRecommendations for {u_new_2}:")
    # print_recs(u_new_2, top_n=5)

    # 6) Simulate they place orders (append to ORDERS + update profiles)
    # rec1 = recommend(u_new_1, top_n=1)[0]["item"]
    # rec2 = recommend(u_new_2, top_n=1)[0]["item"]
    # record_purchase(u_new_1, rec1)
    # record_purchase(u_new_2, rec2)
    # print(f"\nRecorded purchases: {u_new_1}->{rec1}, {u_new_2}->{rec2}")

    # # 7) Show updated popularity (now includes the two new orders)
    # print_top(5, title="Most popular after new customers ordered")


def recommender(user_id: str , top_n: int = 3):
    """Run demo for a given user_id."""
    # bootstrap_from_orders()
    # print_top(3, title="Top 3 most ordered food")
    # abovve prints out the top 3 most ordered food of all time

    # below it prints out the most preferred food - preferred being defined as "most preferred categories"
    print(f"\nExisting user {user_id}:")
    print_recs(user_id, top_n=top_n)

if __name__ == "__main__":
    customer_id = "customer_1"
    recommender(customer_id)