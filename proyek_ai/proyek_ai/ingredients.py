# =============================================================================
# ingredients.py — Person B2
# Tugas: Fuzzy matching, substitution table, meal plan builder
# Dataset: FOOD-DATA-GROUP1~5.csv + daily_food_nutrition_dataset.csv
# =============================================================================

import os
import random
import pandas as pd
from rapidfuzz import process, fuzz

# =============================================================================
# 1. LOAD FOOD DATABASE
# =============================================================================

def _load_food_db():
    """
    Gabungkan FOOD-DATA-GROUP1~5.csv jadi satu database makanan.
    File harus ada di folder: data/raw/
    """
    base_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    dfs = []
    for i in range(1, 6):
        path = os.path.join(base_dir, f"FOOD-DATA-GROUP{i}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"[WARNING] File tidak ditemukan: {path}")

    if not dfs:
        print("[WARNING] Tidak ada file FOOD-DATA-GROUP ditemukan. Menggunakan data kosong.")
        # Buat dataframe kosong dengan kolom default jika file tidak ada
        df = pd.DataFrame(columns=["food", "Protein", "Carbohydrates", "Fat", "Fiber", "Calories"])
        df.set_index("food", inplace=True)
        return df

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["food"])
    combined["food"] = combined["food"].astype(str).str.lower().str.strip()
    
    # PERUBAHAN: Jadikan nama makanan sebagai index agar mudah lookup data per 100g
    combined.set_index("food", inplace=True)
    return combined


def _load_meal_db():
    """
    Load daily_food_nutrition_dataset.csv untuk template menu.
    """
    path = os.path.join(os.path.dirname(__file__), "data", "raw", "daily_food_nutrition_dataset.csv")
    if not os.path.exists(path):
        print(f"[WARNING] File tidak ditemukan: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.drop_duplicates(subset=["Food_Item", "Meal_Type"])
    return df


# Load sekali saat module diimport
FOOD_DB    = _load_food_db()
FOOD_NAMES = list(FOOD_DB.index)
MEAL_DB    = _load_meal_db()


# =============================================================================
# 2. FUZZY MATCHING
# =============================================================================

def match_ingredient(user_input: str, threshold: int = 65) -> str | None:
    """
    Cocokkan satu bahan dari user ke nama makanan di database.
    Strategi:
      1. Cek apakah user_input adalah substring dari nama di database (exact containment)
      2. Jika tidak, gunakan fuzzy matching dengan token_sort_ratio
    """
    if not FOOD_NAMES:
        return None

    query = str(user_input).lower().strip()

    # Strategi 1: exact containment — cari nama di DB yang mengandung kata query
    # Prioritaskan nama yang paling pendek agar hasilnya bersih
    containment_matches = [name for name in FOOD_NAMES if query in name]
    if containment_matches:
        # Ambil nama terpendek (paling spesifik ke bahan dasar)
        return min(containment_matches, key=len)

    # Strategi 2: fuzzy matching untuk typo / ejaan berbeda
    result = process.extractOne(
        query,
        FOOD_NAMES,
        scorer=fuzz.token_sort_ratio
    )

    if result and result[1] >= threshold:
        return result[0]

    return None


def match_ingredients(ingredient_list: list[str]) -> tuple[list[str], list[str]]:
    """
    Cocokkan list bahan dari user ke database.
    Returns: (matched, missing)
    """
    matched = []
    missing = []

    for item in ingredient_list:
        item = str(item).strip()
        if not item:
            continue
        m = match_ingredient(item)
        if m:
            matched.append(m)
        else:
            missing.append(item)

    return matched, missing


# =============================================================================
# 3. SUBSTITUTION TABLE (VERSI LENGKAP TIDAK DIUBAH)
# =============================================================================

SUBSTITUTES = {
    # --- Protein Hewani ---
    "chicken":       ["tofu", "tempeh", "canned tuna", "eggs"],
    "beef":          ["chicken", "lentils", "mushrooms", "tempeh"],
    "pork":          ["chicken", "turkey", "tofu"],
    "lamb":          ["beef", "chicken", "lentils"],
    "turkey":        ["chicken", "tofu", "canned tuna"],
    "tuna":          ["salmon", "sardines", "chicken", "tofu"],
    "salmon":        ["tuna", "mackerel", "cod", "chicken"],
    "shrimp":        ["chicken", "tofu", "canned tuna"],
    "fish":          ["chicken", "tofu", "eggs"],
    "eggs":          ["tofu scramble", "chickpeas", "silken tofu"],

    # --- Protein Nabati ---
    "tofu":          ["tempeh", "chicken", "eggs", "chickpeas"],
    "tempeh":        ["tofu", "chicken", "lentils"],
    "lentils":       ["chickpeas", "black beans", "tofu"],
    "chickpeas":     ["lentils", "black beans", "tofu"],
    "black beans":   ["lentils", "chickpeas", "kidney beans"],

    # --- Sayuran ---
    "spinach":       ["kangkung", "lettuce", "kale", "bok choy"],
    "broccoli":      ["cauliflower", "green beans", "zucchini", "asparagus"],
    "kale":          ["spinach", "kangkung", "bok choy"],
    "bok choy":      ["spinach", "kangkung", "broccoli"],
    "kangkung":      ["spinach", "bok choy", "lettuce"],
    "lettuce":       ["spinach", "arugula", "cabbage"],
    "cabbage":       ["lettuce", "spinach", "bok choy"],
    "carrot":        ["sweet potato", "pumpkin", "beetroot"],
    "sweet potato":  ["potato", "pumpkin", "carrot"],
    "potato":        ["sweet potato", "cassava", "rice"],
    "tomato":        ["red bell pepper", "roasted capsicum"],
    "asparagus":     ["green beans", "broccoli", "zucchini"],
    "zucchini":      ["cucumber", "green beans", "eggplant"],
    "eggplant":      ["zucchini", "mushrooms", "bell pepper"],
    "mushrooms":     ["eggplant", "zucchini", "tofu"],
    "bell pepper":   ["tomato", "carrot", "zucchini"],
    "corn":          ["green peas", "sweet potato", "carrot"],
    "green peas":    ["corn", "edamame", "chickpeas"],

    # --- Karbohidrat / Biji-bijian ---
    "brown rice":    ["white rice", "quinoa", "oats"],
    "white rice":    ["brown rice", "quinoa", "cauliflower rice"],
    "quinoa":        ["brown rice", "bulgur", "oats"],
    "oats":          ["quinoa", "brown rice", "whole wheat bread"],
    "pasta":         ["rice noodles", "glass noodles", "zucchini noodles", "white rice"],
    "bread":         ["whole wheat bread", "rice cake", "oats"],
    "whole wheat":   ["brown rice", "oats", "quinoa"],
    "noodles":       ["rice noodles", "pasta", "glass noodles"],

    # --- Susu / Dairy ---
    "milk":          ["soy milk", "oat milk", "coconut milk", "almond milk"],
    "cheese":        ["nutritional yeast", "tofu", "low-fat cottage cheese"],
    "butter":        ["olive oil", "coconut oil", "avocado"],
    "yogurt":        ["low-fat yogurt", "kefir", "silken tofu"],
    "cream":         ["coconut cream", "low-fat yogurt", "silken tofu"],

    # --- Lemak Sehat ---
    "avocado":       ["olive oil", "nuts", "peanut butter"],
    "olive oil":     ["avocado oil", "canola oil", "coconut oil"],
    "nuts":          ["seeds", "peanut butter", "avocado"],
    "peanut butter": ["almond butter", "tahini", "sunflower seed butter"],

    # --- Bumbu & Lainnya ---
    "garlic":        ["garlic powder", "shallots", "onion"],
    "onion":         ["shallots", "leeks", "garlic"],
    "ginger":        ["ginger powder", "galangal", "turmeric"],
    "soy sauce":     ["coconut aminos", "tamari", "fish sauce"],
    "sugar":         ["honey", "maple syrup", "stevia"],
    "honey":         ["maple syrup", "agave syrup", "sugar"],
}


def get_substitute(ingredient: str) -> str:
    """Cari satu alternatif pengganti untuk sebuah bahan."""
    ingredient_lower = str(ingredient).lower().strip()

    # Cek exact match dulu
    if ingredient_lower in SUBSTITUTES:
        return SUBSTITUTES[ingredient_lower][0]

    # Cek partial match
    for key, subs in SUBSTITUTES.items():
        if key in ingredient_lower or ingredient_lower in key:
            return subs[0]

    return "any similar ingredient"


def get_all_substitutes(ingredient: str) -> list[str]:
    """Ambil semua alternatif pengganti untuk sebuah bahan."""
    ingredient_lower = str(ingredient).lower().strip()

    if ingredient_lower in SUBSTITUTES:
        return SUBSTITUTES[ingredient_lower]

    for key, subs in SUBSTITUTES.items():
        if key in ingredient_lower or ingredient_lower in key:
            return subs

    return []


# =============================================================================
# 4. MEAL TEMPLATES (Dirombak untuk Dynamic Portions / Gram calculation)
# =============================================================================

def _build_dynamic_templates() -> dict:
    """
    Template menu baru yang mendukung sistem porsi gram.
    Menggunakan roles yang akan diisi oleh bahan user,
    serta kcal_frac yang menentukan persentase makro harian untuk meal ini.
    """
    templates = {
        "diabetes": {
            "breakfast": [
                {"slot": "breakfast", "name": "Scrambled {protein} with {veg}", "roles": {"protein": "eggs", "veg": "spinach"}, "kcal_frac": 0.25},
                {"slot": "breakfast", "name": "{grain} Bowl with {protein}", "roles": {"grain": "oats", "protein": "milk"}, "kcal_frac": 0.25},
                {"slot": "breakfast", "name": "Avocado {grain} with {protein}", "roles": {"grain": "bread", "protein": "eggs"}, "kcal_frac": 0.25}
            ],
            "lunch": [
                {"slot": "lunch", "name": "Grilled {protein} with {grain} and {veg}", "roles": {"protein": "chicken", "grain": "brown rice", "veg": "broccoli"}, "kcal_frac": 0.35},
                {"slot": "lunch", "name": "Lentil Soup with {veg} and {protein}", "roles": {"protein": "lentils", "veg": "spinach", "grain": "bread"}, "kcal_frac": 0.35},
                {"slot": "lunch", "name": "{protein} Salad with {veg}", "roles": {"protein": "tuna", "veg": "lettuce"}, "kcal_frac": 0.35}
            ],
            "dinner": [
                {"slot": "dinner", "name": "Baked {protein} with Roasted {veg}", "roles": {"protein": "salmon", "veg": "zucchini"}, "kcal_frac": 0.30},
                {"slot": "dinner", "name": "{protein} and {veg} Stir-fry", "roles": {"protein": "tofu", "veg": "mushrooms", "grain": "quinoa"}, "kcal_frac": 0.30},
                {"slot": "dinner", "name": "Steamed {protein} with {veg}", "roles": {"protein": "cod", "veg": "broccoli", "grain": "brown rice"}, "kcal_frac": 0.30}
            ]
        },
        "hypertension": {
            "breakfast": [
                {"slot": "breakfast", "name": "Scrambled {protein} with {grain}", "roles": {"protein": "eggs", "grain": "oats"}, "kcal_frac": 0.25},
                {"slot": "breakfast", "name": "Fruit and {protein} Smoothie", "roles": {"protein": "yogurt", "veg": "spinach"}, "kcal_frac": 0.25}
            ],
            "lunch": [
                {"slot": "lunch", "name": "Grilled {protein} Salad", "roles": {"protein": "chicken", "veg": "lettuce", "grain": "quinoa"}, "kcal_frac": 0.35},
                {"slot": "lunch", "name": "Tuna and {veg} Wrap", "roles": {"protein": "tuna", "veg": "spinach", "grain": "bread"}, "kcal_frac": 0.35}
            ],
            "dinner": [
                {"slot": "dinner", "name": "Salmon Grilled with {veg}", "roles": {"protein": "salmon", "veg": "asparagus", "grain": "brown rice"}, "kcal_frac": 0.30},
                {"slot": "dinner", "name": "Tofu Stir-fry with {veg}", "roles": {"protein": "tofu", "veg": "broccoli", "grain": "white rice"}, "kcal_frac": 0.30}
            ]
        },
        "malnutrition": {
            "breakfast": [
                {"slot": "breakfast", "name": "Rich {protein} and {grain} Bowl", "roles": {"protein": "eggs", "grain": "oats", "fat": "peanut butter"}, "kcal_frac": 0.30},
                {"slot": "breakfast", "name": "Loaded Avocado {grain}", "roles": {"grain": "bread", "protein": "eggs", "fat": "avocado"}, "kcal_frac": 0.30}
            ],
            "lunch": [
                {"slot": "lunch", "name": "Hearty {protein} Pasta with {veg}", "roles": {"protein": "beef", "grain": "pasta", "veg": "tomato"}, "kcal_frac": 0.35},
                {"slot": "lunch", "name": "Creamy {protein} Soup with {grain}", "roles": {"protein": "chicken", "grain": "bread", "fat": "cream"}, "kcal_frac": 0.35}
            ],
            "dinner": [
                {"slot": "dinner", "name": "Dense {protein} Stir-fry with {grain}", "roles": {"protein": "pork", "grain": "white rice", "veg": "cabbage"}, "kcal_frac": 0.35},
                {"slot": "dinner", "name": "Baked {protein} with {fat} and {grain}", "roles": {"protein": "salmon", "fat": "cheese", "grain": "potato"}, "kcal_frac": 0.35}
            ]
        },
        "none": {
            "breakfast": [
                {"slot": "breakfast", "name": "Scrambled {protein}", "roles": {"protein": "eggs", "grain": "bread"}, "kcal_frac": 0.25},
                {"slot": "breakfast", "name": "{grain} Porridge", "roles": {"grain": "oats", "protein": "milk"}, "kcal_frac": 0.25}
            ],
            "lunch": [
                {"slot": "lunch", "name": "Grilled {protein} with {veg}", "roles": {"protein": "chicken", "veg": "spinach", "grain": "white rice"}, "kcal_frac": 0.35},
                {"slot": "lunch", "name": "{protein} Salad", "roles": {"protein": "tuna", "veg": "lettuce"}, "kcal_frac": 0.35}
            ],
            "dinner": [
                {"slot": "dinner", "name": "Baked {protein}", "roles": {"protein": "fish", "veg": "potato"}, "kcal_frac": 0.30},
                {"slot": "dinner", "name": "{protein} Stir-fry", "roles": {"protein": "tofu", "veg": "broccoli", "grain": "noodles"}, "kcal_frac": 0.30}
            ]
        }
    }
    return templates

# Build templates sekali saat import
MEAL_TEMPLATES = _build_dynamic_templates()


# =============================================================================
# 5. MEAL PLAN BUILDER (DENGAN HITUNGAN GRAM)
# =============================================================================

def _get_macro_per_100g(ingredient: str, macro_col: str, default_val: float = 10.0) -> float:
    """
    Fungsi bantu untuk mengambil nilai nutrisi per 100g dari database.
    Jika tidak ada di database, gunakan default_val agar tidak error.
    """
    if ingredient not in FOOD_DB.index or macro_col not in FOOD_DB.columns:
        return default_val
    try:
        val = FOOD_DB.loc[ingredient, macro_col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val) if pd.notna(val) and val > 0 else default_val
    except Exception:
        return default_val


def _pick_meal(slot_templates: list[dict], day: int, slot_index: int) -> dict:
    """
    Pilih menu dari template list secara bergilir + sedikit variasi.
    Menggunakan kombinasi day dan slot_index agar variasi per hari.
    """
    if not slot_templates:
        return {}
    idx = (day + slot_index) % len(slot_templates)
    return slot_templates[idx]


# def _resolve_ingredients(roles: dict, matched_ingredients: list[str]) -> tuple[dict, list]:
#     """
#     Pengganti _check_availability.
#     Fungsi ini memetakan role (protein, grain, veg) dari template menu
#     ke bahan yang dimiliki user, atau menggunakan substitusi jika tidak punya.
#     """
#     resolved = {}
#     missing_reports = []

#     for role, default_ing in roles.items():
#         # Cek apakah user punya bahan default
#         if default_ing in matched_ingredients:
#             resolved[role] = default_ing
#             continue
        
#         # Cek apakah user punya substitusi dari bahan default
#         subs = get_all_substitutes(default_ing)
#         found_sub = next((s for s in subs if s in matched_ingredients), None)
        
#         if found_sub:
#             resolved[role] = found_sub
#         else:
#             # Tidak punya sama sekali, laporkan sebagai missing item
#             resolved[role] = default_ing
#             sub_suggestion = get_substitute(default_ing)
#             missing_reports.append({
#                 "role": role, 
#                 "missing": default_ing, 
#                 "substitute": sub_suggestion
#             })
            
#     return resolved, missing_reports

def _resolve_ingredients(roles: dict, matched_ingredients: list[str]) -> tuple[dict, list]:
    """
    Logika yang disempurnakan: Menggunakan pencocokan lowercase agar 
    bahan dari typo (misal: 'ciken' -> 'chicken') langsung dianggap tersedia.
    """
    resolved = {}
    missing_reports = []
    
    # Jadikan semua bahan user menjadi huruf kecil untuk perbandingan yang aman
    matched_lower = [m.lower() for m in matched_ingredients]

    for role, default_ing in roles.items():
        # Cek apakah bahan default ada di daftar matched_lower
        if default_ing.lower() in matched_lower:
            resolved[role] = default_ing
            continue
        
        # Cek substitusi jika bahan asli tidak ada
        subs = get_all_substitutes(default_ing)
        found_sub = next((s for s in subs if s.lower() in matched_lower), None)
        
        if found_sub:
            resolved[role] = found_sub
        else:
            # Tetap catat bahan yang benar-benar tidak ada agar sistem jujur
            resolved[role] = default_ing
            sub_suggestion = get_substitute(default_ing)
            missing_reports.append({
                "role": role, 
                "missing": default_ing, 
                "substitute": sub_suggestion
            })
            
    return resolved, missing_reports


def build_meal_plan(
    user_ingredients: list[str],
    macros: dict,
    disease: str,
    main_protein: str = "",
    main_carbohydrate: str = "",
    main_fiber: str = ""
) -> list[dict]:
    
    matched, unmatched = match_ingredients(user_ingredients)

    # --- FUNGSI BANTU UNTUK INJEKSI BAHAN UTAMA ---
    def inject_to_inventory(ingredient_name):
        if ingredient_name:
            ing_lower = ingredient_name.lower()
            if ing_lower not in [m.lower() for m in matched]:
                matched.append(ing_lower)

    # Suntikkan ketiga bahan pilihan user ke dalam daftar 'bahan tersedia'
    inject_to_inventory(main_protein)
    inject_to_inventory(main_carbohydrate)
    inject_to_inventory(main_fiber)
    # -----------------------------------------------

    disease_key = disease.lower() if disease.lower() in MEAL_TEMPLATES else "none"
    templates   = MEAL_TEMPLATES[disease_key]

    week_plan = []

    for day in range(1, 8):
        day_meals = {"day": day}

        for slot_idx, slot in enumerate(["breakfast", "lunch", "dinner"]):
            tmpl = _pick_meal(templates[slot], day, slot_idx)
            if not tmpl: continue

            # --- PAKSA TEMPLATE MENGGUNAKAN PILIHAN USER ---
            current_roles = tmpl["roles"].copy()
            
            if main_protein and "protein" in current_roles:
                current_roles["protein"] = main_protein
                
            if main_carbohydrate and "grain" in current_roles:
                current_roles["grain"] = main_carbohydrate
                
            if main_fiber and "veg" in current_roles:
                current_roles["veg"] = main_fiber
            # -----------------------------------------------

            # 1. Verifikasi ketersediaan (pasti lolos karena sudah diinjeksi)
            resolved_roles, missing_info = _resolve_ingredients(current_roles, matched)
            
            portions_grams = {}
            calculated_macros = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0}

            # 2. Hitung porsi gram berdasarkan target harian
            for role, ingredient in resolved_roles.items():
                if role == "protein":
                    target = macros.get("target_protein", 60) * tmpl["kcal_frac"]
                    db_col = "Protein"
                elif role == "grain":
                    target = macros.get("target_carbs", 200) * tmpl["kcal_frac"]
                    db_col = "Carbohydrates"
                elif role == "veg":
                    target = macros.get("target_fiber", 25) * tmpl["kcal_frac"]
                    db_col = "Fiber"
                elif role == "fat":
                    target = macros.get("target_fat", 60) * tmpl["kcal_frac"]
                    db_col = "Fat"
                else:
                    target = 10 
                    db_col = "Protein"

                macro_per_100g = _get_macro_per_100g(ingredient, db_col, default_val=15.0)
                grams = round((target / macro_per_100g) * 100)
                
                grams = max(10, min(grams, 500))
                portions_grams[role] = {"ingredient": ingredient, "grams": grams}

                calculated_macros["protein"] += (grams / 100) * _get_macro_per_100g(ingredient, "Protein", 5)
                calculated_macros["carbs"]   += (grams / 100) * _get_macro_per_100g(ingredient, "Carbohydrates", 10)
                calculated_macros["fat"]     += (grams / 100) * _get_macro_per_100g(ingredient, "Fat", 2)
                calculated_macros["fiber"]   += (grams / 100) * _get_macro_per_100g(ingredient, "Fiber", 1)
                calculated_macros["calories"] += (grams / 100) * _get_macro_per_100g(ingredient, "Calories", 150)

            try:
                meal_name = tmpl["name"].format(**resolved_roles).title()
            except KeyError:
                meal_name = tmpl["name"].title()

            day_meals[slot] = {
                "name":      meal_name,
                "portions":  portions_grams,
                "macros":    {k: round(v, 1) for k, v in calculated_macros.items()},
                "available": len(missing_info) == 0,
                "missing":   missing_info[0]["missing"] if missing_info else None, 
                "substitute": missing_info[0]["substitute"] if missing_info else None, 
                "missing_details": missing_info
            }

        week_plan.append(day_meals)

    return week_plan

# =============================================================================
# 6. UTILITY: SUMMARY HARIAN
# =============================================================================

def get_daily_summary(day_plan: dict) -> dict:
    """
    Hitung total nutrisi dari satu hari rencana makan (menggunakan struktur baru).
    """
    total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0}
    for slot in ["breakfast", "lunch", "dinner"]:
        m = day_plan.get(slot, {})
        macs = m.get("macros", {})
        total["calories"] += macs.get("calories", 0)
        total["protein"]  += macs.get("protein",  0)
        total["carbs"]    += macs.get("carbs",    0)
        total["fat"]      += macs.get("fat",      0)
        total["fiber"]    += macs.get("fiber",    0)
    
    return {k: round(v, 1) for k, v in total.items()}


# =============================================================================
# TEST MANUAL (jalankan: python ingredients.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST ingredients.py (Full Version with Dynamic Grams)")
    print("=" * 60)

    # Test fuzzy matching
    test_inputs = ["chiken", "spinach", "salmon", "nasi", "dragon fruit", "brokoli"]
    print("\n[1] Fuzzy Matching Test:")
    for item in test_inputs:
        result = match_ingredient(item)
        print(f"  '{item}' -> {result}")

    # Test match list
    print("\n[2] Batch Matching Test:")
    matched, missing = match_ingredients(["chicken", "rice", "spinach", "blabla", "tuna"])
    print(f"  Matched : {matched}")
    print(f"  Missing : {missing}")

    # Test substitusi
    print("\n[3] Substitution Test:")
    for ing in ["spinach", "chicken", "pasta", "dragon fruit"]:
        print(f"  '{ing}' -> {get_substitute(ing)}")

    # Inject dummy data for macro calculation if database is empty/missing columns
    if "Protein" not in FOOD_DB.columns:
        FOOD_DB["Protein"] = 15.0
        FOOD_DB["Carbohydrates"] = 20.0
        FOOD_DB["Fat"] = 5.0
        FOOD_DB["Fiber"] = 3.0
        FOOD_DB["Calories"] = 150.0

    # Test meal plan with gram calculations
    print("\n[4] Meal Plan Test (diabetes, 3 hari pertama):")
    macros = {
        "target_calories": 1800,
        "target_protein":  75,
        "target_carbs":    210,
        "target_fat":      60,
        "target_fiber":    25,
    }
    
    plan = build_meal_plan(
        user_ingredients=["chicken", "rice", "spinach", "egg", "oats"],
        macros=macros,
        disease="diabetes"
    )
    
    for day in plan[:3]:
        print(f"\n  Day {day['day']}:")
        for slot in ["breakfast", "lunch", "dinner"]:
            m = day[slot]
            status = "✓" if m["available"] else f"✗ (missing: {m['missing']} → sub: {m['substitute']})"
            print(f"    {slot.capitalize():10s}: {m['name'][:40]:40s} {status}")
            
            # Print Gram Portions
            ports = [f"{v['grams']}g {v['ingredient']}" for k, v in m['portions'].items()]
            print(f"                Porsi: {', '.join(ports)}")
            
        summary = get_daily_summary(day)
        print(f"    Total: {summary['calories']} kcal | "
              f"P:{summary['protein']}g | C:{summary['carbs']}g | "
              f"F:{summary['fat']}g | Fiber:{summary['fiber']}g")