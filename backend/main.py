# backend/main.py
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="Mortgage Approval Service")

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model = None
model_loaded = False

# Точный порядок признаков из вашей модели (27 признаков)
FEATURE_ORDER = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
    'person_gender_female', 'person_gender_male', 'person_education_Associate',
    'person_education_Bachelor', 'person_education_Doctorate', 'person_education_High School',
    'person_education_Master', 'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
    'person_home_ownership_OWN', 'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
    'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'previous_loan_defaults_on_file_No',
    'previous_loan_defaults_on_file_Yes'
]

# ==================== API ЭНДПОИНТЫ ====================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Главная страница с интерфейсом"""
    return HTMLResponse(content=HTML_PAGE)

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Загрузка модели"""
    global model, model_loaded
    
    if not file.filename.endswith('.pkl'):
        raise HTTPException(400, "Файл должен быть в формате .pkl")
    
    try:
        contents = await file.read()
        model = joblib.load(io.BytesIO(contents))
        model_loaded = True
        
        model_type = model.__class__.__name__
        return {
            "status": "success", 
            "message": f"Модель '{model_type}' успешно загружена",
            "model_type": model_type
        }
        
    except Exception as e:
        model_loaded = False
        raise HTTPException(500, f"Ошибка загрузки: {str(e)}")

@app.get("/model-status")
async def model_status():
    """Проверка статуса модели"""
    return {"loaded": model_loaded}

@app.post("/predict")
async def predict(data: dict):
    """Предсказание для одного клиента"""
    global model, model_loaded
    
    if not model_loaded:
        raise HTTPException(400, "Модель не загружена")
    
    try:
        # Конвертируем в DataFrame с правильным порядком колонок
        df = pd.DataFrame([data])
        df = df[FEATURE_ORDER]
        
        # Заменяем NaN на 0
        df = df.fillna(0)
        
        # Валидация
        if df['person_income'].iloc[0] <= 0:
            raise HTTPException(400, "Доход должен быть больше 0")
        
        if df['loan_percent_income'].iloc[0] > 500:
            raise HTTPException(400, "Сумма кредита слишком большая относительно дохода")
        
        # Предсказание
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]
        
        result = {
            "loan_status": "Approved" if prediction == 1 else "Rejected",
            "probability": round(float(probability), 3),
            "confidence": round(float(probability if prediction == 1 else 1 - probability), 3)
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(500, f"Ошибка предсказания: {str(e)}")

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """Предсказание из CSV файла"""
    global model, model_loaded
    
    if not model_loaded:
        raise HTTPException(400, "Модель не загружена")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Проверяем и заполняем все нужные колонки
        for col in FEATURE_ORDER:
            if col not in df.columns:
                df[col] = 0
        
        # Берем только нужные колонки в правильном порядке
        X = df[FEATURE_ORDER]
        X = X.fillna(0)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Результат
        df['Predicted_Status'] = ["Approved" if p == 1 else "Rejected" for p in predictions]
        df['Probability'] = probabilities
        
        # Конвертируем для JSON (заменяем NaN на None)
        result_data = []
        for _, row in df.head(50).iterrows():
            row_dict = {}
            for col, value in row.items():
                if pd.isna(value):
                    row_dict[col] = None
                else:
                    row_dict[col] = value
            result_data.append(row_dict)
        
        return JSONResponse({
            "total_rows": len(df),
            "approved": int((predictions == 1).sum()),
            "rejected": int((predictions == 0).sum()),
            "avg_probability": round(float(probabilities.mean()), 3),
            "data": result_data
        })
        
    except Exception as e:
        raise HTTPException(500, f"Ошибка: {str(e)}")

@app.delete("/clear-model")
async def clear_model():
    """Удалить модель"""
    global model, model_loaded
    model = None
    model_loaded = False
    return {"status": "success", "message": "Модель удалена"}

# ==================== HTML ФРОНТЕНД ====================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mortgage Approval Service</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
        .status.loaded { background: #10b981; color: white; }
        .status.not-loaded { background: #ef4444; color: white; }
        input, button, select { padding: 10px; margin: 5px; border-radius: 5px; }
        input, select { border: 1px solid #ddd; width: calc(100% - 24px); }
        button { background: #667eea; color: white; border: none; cursor: pointer; }
        button:hover { background: #5a67d8; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; overflow-x: auto; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; font-size: 12px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .section-title { grid-column: span 4; background: #e0e7ff; padding: 10px; margin: 10px 0; border-radius: 5px; font-weight: bold; }
        .info { background: #e0e7ff; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
        select { width: 100%; padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #ddd; }
        .result-approved { color: green; font-size: 24px; font-weight: bold; }
        .result-rejected { color: red; font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>🏠 Сервис одобрения ипотеки</h1>
        <p>Статус модели: <span id="statusText" class="status not-loaded">Не загружена</span></p>
    </div>

    <div class="card">
        <h2>📤 1. Загрузить модель</h2>
        <input type="file" id="modelFile" accept=".pkl">
        <button onclick="uploadModel()">Загрузить</button>
        <button onclick="clearModel()">Удалить модель</button>
        <div id="uploadMsg"></div>
    </div>

    <div class="card">
        <h2>📝 2. Одиночное предсказание</h2>
        <form id="predictForm">
            <div class="grid">
                <div class="section-title">👤 Личные данные</div>
                
                <div class="form-group">
                    <label>Возраст (лет)</label>
                    <input type="number" name="person_age" value="30" min="18" max="100" required>
                </div>
                <div class="form-group">
                    <label>Доход ($)</label>
                    <input type="number" name="person_income" value="50000" min="10000" required>
                </div>
                <div class="form-group">
                    <label>Стаж работы (годы)</label>
                    <input type="number" name="person_emp_exp" value="5" min="0" max="50">
                </div>
                <div class="form-group">
                    <label>Кредитная история (годы)</label>
                    <input type="number" name="cb_person_cred_hist_length" value="5" min="0" max="30">
                </div>
                
                <div class="section-title">🏦 Кредитные параметры</div>
                
                <div class="form-group">
                    <label>Сумма кредита ($)</label>
                    <input type="number" name="loan_amnt" value="150000" min="10000" required>
                </div>
                <div class="form-group">
                    <label>Процентная ставка (%)</label>
                    <input type="number" name="loan_int_rate" value="10.5" step="0.1" min="5" max="30">
                </div>
                <div class="form-group">
                    <label>Кредитный рейтинг</label>
                    <input type="number" name="credit_score" value="700" min="300" max="850">
                </div>
                <div class="form-group">
                    <label>Пол</label>
                    <select name="gender">
                        <option value="male">Мужской</option>
                        <option value="female">Женский</option>
                    </select>
                </div>
                
                <div class="section-title">🎓 Образование</div>
                
                <div class="form-group">
                    <label>Образование</label>
                    <select name="education">
                        <option value="Bachelor">Бакалавр</option>
                        <option value="Master">Магистр</option>
                        <option value="Doctorate">Доктор</option>
                        <option value="Associate">Ассоциированное</option>
                        <option value="High School">Среднее</option>
                    </select>
                </div>
                
                <div class="section-title">🏠 Жилье</div>
                
                <div class="form-group">
                    <label>Тип жилья</label>
                    <select name="home_ownership">
                        <option value="RENT">Аренда</option>
                        <option value="MORTGAGE">Ипотека</option>
                        <option value="OWN">Собственное</option>
                        <option value="OTHER">Другое</option>
                    </select>
                </div>
                
                <div class="section-title">🎯 Цель кредита</div>
                
                <div class="form-group">
                    <label>Цель кредита</label>
                    <select name="loan_intent">
                        <option value="PERSONAL">Личные нужды</option>
                        <option value="DEBTCONSOLIDATION">Объединение долгов</option>
                        <option value="EDUCATION">Образование</option>
                        <option value="HOMEIMPROVEMENT">Улучшение жилья</option>
                        <option value="MEDICAL">Медицина</option>
                        <option value="VENTURE">Бизнес</option>
                    </select>
                </div>
                
                <div class="section-title">📜 История дефолтов</div>
                
                <div class="form-group">
                    <label>Были ли просрочки?</label>
                    <select name="defaults">
                        <option value="No">Нет</option>
                        <option value="Yes">Да</option>
                    </select>
                </div>
            </div>
            
            <button type="submit" style="margin-top: 20px;">🔮 Предсказать</button>
        </form>
        
        <div id="predictResult" class="result" style="display:none;"></div>
    </div>

    <div class="card">
        <h2>📁 3. CSV файл</h2>
        <input type="file" id="csvFile" accept=".csv">
        <button onclick="predictCSV()">Обработать CSV</button>
        <div id="csvResult" class="result" style="display:none;"></div>
    </div>
</div>

<script>
    async function checkStatus() {
        try {
            const response = await fetch('/model-status');
            const data = await response.json();
            const statusSpan = document.getElementById('statusText');
            if (data.loaded) {
                statusSpan.textContent = '✅ Загружена';
                statusSpan.className = 'status loaded';
            } else {
                statusSpan.textContent = '❌ Не загружена';
                statusSpan.className = 'status not-loaded';
            }
        } catch(e) { console.error(e); }
    }
    
    async function uploadModel() {
        const file = document.getElementById('modelFile').files[0];
        if (!file) { alert('Выберите файл модели'); return; }
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload-model', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok) {
                document.getElementById('uploadMsg').innerHTML = `<p style="color:green;">✅ ${result.message}</p>`;
                checkStatus();
            } else {
                document.getElementById('uploadMsg').innerHTML = `<p style="color:red;">❌ ${result.detail}</p>`;
            }
        } catch(e) { alert('Ошибка: ' + e.message); }
    }
    
    async function clearModel() {
        if (!confirm('Удалить модель?')) return;
        try {
            await fetch('/clear-model', { method: 'DELETE' });
            checkStatus();
            document.getElementById('uploadMsg').innerHTML = '<p style="color:orange;">Модель удалена</p>';
        } catch(e) { alert('Ошибка'); }
    }
    
    document.getElementById('predictForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const gender = formData.get('gender');
        const education = formData.get('education');
        const home_ownership = formData.get('home_ownership');
        const loan_intent = formData.get('loan_intent');
        const defaults = formData.get('defaults');
        
        const person_income = parseFloat(formData.get('person_income'));
        const loan_amnt = parseFloat(formData.get('loan_amnt'));
        
        // Проверка дохода
        if (person_income <= 0) {
            alert('Доход должен быть больше 0');
            return;
        }
        
        // Расчет loan_percent_income
        let loan_percent_income = (loan_amnt / person_income) * 100;
        if (loan_percent_income > 500) {
            alert('Сумма кредита не должна превышать годовой доход более чем в 5 раз');
            return;
        }
        
        const data = {
            person_age: parseFloat(formData.get('person_age')),
            person_income: person_income,
            person_emp_exp: parseFloat(formData.get('person_emp_exp')),
            loan_amnt: loan_amnt,
            loan_int_rate: parseFloat(formData.get('loan_int_rate')),
            loan_percent_income: loan_percent_income,
            cb_person_cred_hist_length: parseFloat(formData.get('cb_person_cred_hist_length')),
            credit_score: parseFloat(formData.get('credit_score')),
            person_gender_female: gender === 'female' ? 1 : 0,
            person_gender_male: gender === 'male' ? 1 : 0,
            person_education_Associate: education === 'Associate' ? 1 : 0,
            person_education_Bachelor: education === 'Bachelor' ? 1 : 0,
            person_education_Doctorate: education === 'Doctorate' ? 1 : 0,
            'person_education_High School': education === 'High School' ? 1 : 0,
            person_education_Master: education === 'Master' ? 1 : 0,
            person_home_ownership_MORTGAGE: home_ownership === 'MORTGAGE' ? 1 : 0,
            person_home_ownership_OTHER: home_ownership === 'OTHER' ? 1 : 0,
            person_home_ownership_OWN: home_ownership === 'OWN' ? 1 : 0,
            person_home_ownership_RENT: home_ownership === 'RENT' ? 1 : 0,
            loan_intent_DEBTCONSOLIDATION: loan_intent === 'DEBTCONSOLIDATION' ? 1 : 0,
            loan_intent_EDUCATION: loan_intent === 'EDUCATION' ? 1 : 0,
            loan_intent_HOMEIMPROVEMENT: loan_intent === 'HOMEIMPROVEMENT' ? 1 : 0,
            loan_intent_MEDICAL: loan_intent === 'MEDICAL' ? 1 : 0,
            loan_intent_PERSONAL: loan_intent === 'PERSONAL' ? 1 : 0,
            loan_intent_VENTURE: loan_intent === 'VENTURE' ? 1 : 0,
            previous_loan_defaults_on_file_No: defaults === 'No' ? 1 : 0,
            previous_loan_defaults_on_file_Yes: defaults === 'Yes' ? 1 : 0
        };
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            const resultDiv = document.getElementById('predictResult');
            
            if (response.ok) {
                resultDiv.style.display = 'block';
                const statusClass = result.loan_status === 'Approved' ? 'result-approved' : 'result-rejected';
                resultDiv.innerHTML = `
                    <h3>📊 Результат:</h3>
                    <p><strong>Статус:</strong> <span class="${statusClass}">${result.loan_status}</span></p>
                    <p><strong>Вероятность одобрения:</strong> ${(result.probability * 100).toFixed(1)}%</p>
                    <p><strong>Уверенность модели:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                `;
            } else {
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<p style="color:red;">Ошибка: ${result.detail}</p>`;
            }
        } catch(e) { alert('Ошибка: ' + e.message); }
    });
    
    async function predictCSV() {
        const file = document.getElementById('csvFile').files[0];
        if (!file) { alert('Выберите CSV файл'); return; }
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/predict-csv', { method: 'POST', body: formData });
            const result = await response.json();
            const resultDiv = document.getElementById('csvResult');
            
            if (response.ok) {
                resultDiv.style.display = 'block';
                let html = `<h3>📊 Результаты:</h3>`;
                html += `<p>Всего записей: ${result.total_rows}</p>`;
                html += `<p>Одобрено: ${result.approved}</p>`;
                html += `<p>Отказано: ${result.rejected}</p>`;
                html += `<p>Средняя вероятность: ${(result.avg_probability * 100).toFixed(1)}%</p>`;
                html += `<h4>Первые 10 предсказаний:</h4>`;
                html += `<table border="1" style="width:100%; border-collapse: collapse;">`;
                html += `<tr><th>#</th><th>Статус</th><th>Вероятность</th></tr>`;
                
                result.data.slice(0, 10).forEach((item, idx) => {
                    html += `<tr>
                        <td>${idx + 1}</td>
                        <td>${item.Predicted_Status}</td>
                        <td>${(item.Probability * 100).toFixed(1)}%</td>
                    </tr>`;
                });
                html += `</table>`;
                resultDiv.innerHTML = html;
            } else {
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<p style="color:red;">Ошибка: ${result.detail}</p>`;
            }
        } catch(e) { alert('Ошибка: ' + e.message); }
    }
    
    checkStatus();
    setInterval(checkStatus, 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)