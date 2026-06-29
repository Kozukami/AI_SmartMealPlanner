import os
import json
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# =============================================================================
# KONFIGURASI
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISEASE_MAP = {
    'diabetes':     [1, 0, 0],
    'hypertension': [0, 1, 0],
    'malnutrition': [0, 0, 1],
    'none':         [0, 0, 0]
}

BMI_CAT_MAP = {'normal': 0, 'obese': 1, 'overweight': 2, 'underweight': 3}

def bmi_category(bmi):
    if bmi < 18.5: return 'underweight'
    elif bmi < 25.0: return 'normal'
    elif bmi < 30.0: return 'overweight'
    else: return 'obese'


# =============================================================================
# ✅ FALLBACK: Kalkulasi makro berbasis BMI & kondisi penyakit
#    Digunakan karena file model (.pkl) yang tersedia adalah LabelEncoder,
#    bukan model prediksi multitarget.
# =============================================================================

def predict_macros_fallback(bmi, disease_flags):
    """
    Estimasi target makro harian berdasarkan BMI dan kondisi penyakit.
    Mengacu pada panduan gizi umum (ADA, WHO, Kemenkes RI).
    """
    # --- Estimasi kalori dasar berdasarkan kategori BMI ---
    if bmi < 18.5:
        base_calories = 2100   # underweight: perlu lebih banyak kalori
    elif bmi < 25.0:
        base_calories = 1900   # normal
    elif bmi < 30.0:
        base_calories = 1700   # overweight: sedikit kurangi
    else:
        base_calories = 1500   # obese: defisit kalori

    # --- Sesuaikan berdasarkan penyakit ---
    has_diabetes     = disease_flags[0]
    has_hypertension = disease_flags[1]
    has_malnutrition = disease_flags[2]

    if has_malnutrition:
        base_calories += 300   # malnutrisi butuh tambahan kalori

    # --- Tentukan persentase karbohidrat ---
    if has_diabetes:
        carb_pct = 0.40    # diabetes: batasi karbohidrat
    else:
        carb_pct = 0.50    # normal / hipertensi

    # --- Hitung makro (gram) ---
    calories = base_calories
    protein  = round((calories * 0.20) / 4, 1)   # 20% dari kalori, 4 kcal/g
    carbs    = round((calories * carb_pct) / 4, 1)
    fat      = round((calories * 0.30) / 9, 1)   # 30% dari kalori, 9 kcal/g
    fiber    = 28.0 if not has_diabetes else 35.0  # diabetes butuh lebih banyak serat

    return [calories, protein, carbs, fat, fiber]


# =============================================================================
# IMPORT INGREDIENTS MODULE
# =============================================================================

from ingredients import build_meal_plan


# =============================================================================
# ROUTING
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Tolak jika BMI kosong
        if not data or not str(data.get('bmi', '')).strip():
            raise ValueError("Angka BMI tidak boleh kosong! Tolong isi di website.")

        bmi      = float(data['bmi'])
        disease  = data.get('disease', 'none').lower()

        # Ambil semua pilihan taste profile dari frontend
        main_protein      = data.get('main_protein', '').strip()
        main_carbohydrate = data.get('main_carbohydrate', '').strip()
        main_fiber        = data.get('main_fiber', '').strip()

        # Ambil daftar bahan
        ingredients_raw = data.get('ingredients', '')
        if isinstance(ingredients_raw, str):
            ingredients = [i.strip() for i in ingredients_raw.split(',') if i.strip()]
        else:
            ingredients = ingredients_raw

        disease_flags = DISEASE_MAP.get(disease, [0, 0, 0])

        # ✅ Gunakan fallback kalkulasi (bukan model .pkl yang salah tipe)
        targets = predict_macros_fallback(bmi, disease_flags)

        macro_keys = ['DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT', 'DR1TFIBE']
        raw_macros = dict(zip(macro_keys, targets))

        macros = {
            'target_calories': float(raw_macros.get('DR1TKCAL', 0)),
            'target_protein':  float(raw_macros.get('DR1TPROT', 0)),
            'target_carbs':    float(raw_macros.get('DR1TCARB', 0)),
            'target_fat':      float(raw_macros.get('DR1TTFAT', 0)),
            'target_fiber':    float(raw_macros.get('DR1TFIBE', 0))
        }

        # Bangun meal plan 7 hari
        meal_plan = build_meal_plan(
            ingredients,
            macros,
            disease,
            main_protein,
            main_carbohydrate,
            main_fiber
        )

        return jsonify({
            'daily_targets': macros,
            'week_plan': meal_plan
        })

    except Exception as e:
        print("\n" + "="*50)
        print("❌ TERJADI ERROR SAAT PREDIKSI:")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
