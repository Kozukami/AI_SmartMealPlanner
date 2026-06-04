import os
import joblib
import json
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 1. LOAD MODEL & SCHEMA AI
model = joblib.load('outputs/trained_model.pkl')
with open('outputs/model_schema.json', 'r') as f:
    schema = json.load(f)

DISEASE_MAP = {
    'diabetes':     [1, 0, 0],
    'hypertension': [0, 1, 0],
    'malnutrition': [0, 0, 1],
    'none':         [0, 0, 0]
}

def bmi_category(bmi):
    if bmi < 18.5: return 'underweight'
    elif bmi < 25.0: return 'normal'
    elif bmi < 30.0: return 'overweight'
    else: return 'obese'

BMI_CAT_MAP = {'normal': 0, 'obese': 1, 'overweight': 2, 'underweight': 3}

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
        
        # PENGAMAN 1: Tolak jika form BMI kosong
        if not data or not str(data.get('bmi')).strip():
            raise ValueError("Angka BMI tidak boleh kosong! Tolong isi di website.")
            
        bmi = float(data['bmi'])
        disease = data.get('disease', 'none').lower()
        
        # --- 1. TAMBAHAN BARU: Tangkap data main_protein dari frontend ---
        main_protein = data.get('main_protein', '').strip()
        # -----------------------------------------------------------------
        
        ingredients_raw = data.get('ingredients', '')
        if isinstance(ingredients_raw, str):
            ingredients = [i.strip() for i in ingredients_raw.split(',') if i.strip()]
        else:
            ingredients = ingredients_raw

        cat = bmi_category(bmi)
        disease_flags = DISEASE_MAP.get(disease, [0, 0, 0])
        
        bmi_squared = bmi ** 2
        disease_burden = sum(disease_flags)
        age_default = 40.0
        gender_default = 1.0
        
        # PENGAMAN 2: Gunakan fallback jika Kelompok A lupa menaruh list 'features'
        feature_names = schema.get('features', [
            'BMXBMI', 'bmi_cat_encoded', 'bmi_squared', 
            'has_diabetes', 'has_hypertension', 'has_malnutrition', 
            'disease_burden', 'RIDAGEYR', 'gender_encoded'
        ])
        
        X_df = pd.DataFrame([[
            bmi, 
            BMI_CAT_MAP[cat], 
            bmi_squared, 
            disease_flags[0], 
            disease_flags[1], 
            disease_flags[2], 
            disease_burden, 
            age_default, 
            gender_default
        ]], columns=feature_names)

        targets = model.predict(X_df)[0]
        
        # PENGAMAN 3: Gunakan fallback jika Kelompok A lupa menaruh list 'targets'
        macro_keys = schema.get('targets', ['DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT', 'DR1TFIBE'])
        raw_macros = dict(zip(macro_keys, targets))

        macros = {
            'target_calories': float(raw_macros.get('DR1TKCAL', 0)),
            'target_protein':  float(raw_macros.get('DR1TPROT', 0)),
            'target_carbs':    float(raw_macros.get('DR1TCARB', 0)),
            'target_fat':      float(raw_macros.get('DR1TTFAT', 0)),
            'target_fiber':    float(raw_macros.get('DR1TFIBE', 0))
        }

        # --- 2. UBAH BARIS INI: Tambahkan variabel main_protein ke fungsi ---
        meal_plan = build_meal_plan(ingredients, macros, disease, main_protein)
        # --------------------------------------------------------------------
        
        return jsonify({
            'daily_targets': macros,
            'week_plan': meal_plan
        })
        
    except Exception as e:
        # RADAR ERROR: Akan mencetak biang kerok asli ke terminal hitammu
        print("\n" + "="*50)
        print("❌ TERJADI ERROR SAAT PREDIKSI:")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': str(e)}), 400
# Tambahkan 2 baris ini di paling bawah file app.py milikmu
if __name__ == '__main__':
    app.run(debug=True, port=5000)