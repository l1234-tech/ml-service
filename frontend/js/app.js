// ============================================
// App - Основная логика и роутинг
// ============================================

// Состояние приложения
const state = {
    currentPage: 'model-select',
    modelLoaded: false,
    modelType: null,
    csvData: null,
    csvResults: null
};

// Конфигурация страниц
const pages = {
    'model-select': { id: 'page-model-select', title: 'Выбор модели' },
    'mode-select': { id: 'page-mode-select', title: 'Выбор режима' },
    'single-predict': { id: 'page-single-predict', title: 'Одиночное предсказание' },
    'csv-predict': { id: 'page-csv-predict', title: 'CSV предсказание' }
};

// Инициализация
async function init() {
    setupTheme();
    setupEventListeners();
    await checkModelStatus();
    navigate('model-select');
}

// === Тема ===
function setupTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeButton(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeButton(next);
}

function updateThemeButton(theme) {
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.textContent = theme === 'dark' ? '🌙' : '☀️';
        btn.title = theme === 'dark' ? 'Светлая тема' : 'Тёмная тема';
    }
}

// Роутинг
function navigate(page) {
    if (!pages[page]) return;
    
    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('active');
    });
    
    const pageElement = document.getElementById(pages[page].id);
    if (pageElement) {
        pageElement.classList.add('active');
    }
    
    state.currentPage = page;
    document.title = `${pages[page].title} - Сервис одобрения ипотеки`;
}

// Проверка статуса модели
async function checkModelStatus() {
    try {
        const status = await api.health();
        state.modelLoaded = status.model_loaded;
        state.modelType = status.model_type;
        updateModelStatusUI();
        
        // Если модель уже авто-загружена — показываем карточку с информацией
        if (status.model_loaded) {
            document.getElementById('model-type').textContent = status.model_type || 'Неизвестно';
            toggleElement('model-info-card', true);
        }
    } catch (e) {
        console.error('Ошибка проверки статуса:', e);
    }
}

// Обновление UI статуса модели
function updateModelStatusUI() {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    
    if (state.modelLoaded) {
        dot.classList.add('loaded');
        text.textContent = `Модель: ${state.modelType || 'Загружена'}`;
    } else {
        dot.classList.remove('loaded');
        text.textContent = 'Модель не загружена';
    }
}

// Настройка обработчиков событий
function setupEventListeners() {
    // === Тема ===
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    
    // === Страница 1: Выбор модели ===
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('model-file');
    const btnSelectFile = document.getElementById('btn-select-file');
    
    btnSelectFile.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    uploadArea.addEventListener('click', (e) => {
        if (e.target === uploadArea || e.target.closest('.upload-icon, .upload-text')) {
            fileInput.click();
        }
    });
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleModelFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleModelFileSelect(e.target.files[0]);
        }
    });
    
    document.getElementById('btn-upload').addEventListener('click', uploadModel);
    document.getElementById('btn-download-model').addEventListener('click', downloadModel);
    document.getElementById('btn-next-to-mode').addEventListener('click', () => {
        navigate('mode-select');
    });
    
    // Кнопка "Использовать предустановленную модель"
    document.getElementById('btn-use-default').addEventListener('click', useDefaultModel);
    
    // === Страница 2: Выбор режима ===
    document.getElementById('mode-single').addEventListener('click', () => {
        navigate('single-predict');
    });
    
    document.getElementById('mode-csv').addEventListener('click', () => {
        navigate('csv-predict');
    });
    
    document.getElementById('btn-back-to-model').addEventListener('click', () => {
        navigate('model-select');
    });
    
    // === Страница 3: Одиночное предсказание ===
    document.getElementById('single-predict-form').addEventListener('submit', handleSinglePredict);
    
    document.getElementById('btn-back-to-mode-from-single').addEventListener('click', () => {
        navigate('mode-select');
        hidePredictionResult();
    });
    
    // === Страница 4: CSV предсказание ===
    const csvUploadArea = document.getElementById('csv-upload-area');
    const csvFileInput = document.getElementById('csv-file');
    const btnSelectCsv = document.getElementById('btn-select-csv');
    
    btnSelectCsv.addEventListener('click', (e) => {
        e.stopPropagation();
        csvFileInput.click();
    });
    csvUploadArea.addEventListener('click', (e) => {
        if (e.target === csvUploadArea || e.target.closest('.upload-icon, .upload-text')) {
            csvFileInput.click();
        }
    });
    
    csvUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        csvUploadArea.classList.add('dragover');
    });
    
    csvUploadArea.addEventListener('dragleave', () => {
        csvUploadArea.classList.remove('dragover');
    });
    
    csvUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        csvUploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleCsvFileSelect(files[0]);
        }
    });
    
    csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleCsvFileSelect(e.target.files[0]);
        }
    });
    
    document.getElementById('btn-process-csv').addEventListener('click', processCsv);
    document.getElementById('btn-change-csv').addEventListener('click', changeCsvFile);
    document.getElementById('btn-download-csv').addEventListener('click', downloadResults);
    document.getElementById('btn-show-all').addEventListener('click', showAllResults);
    
    document.getElementById('btn-back-to-mode-from-csv').addEventListener('click', () => {
        navigate('mode-select');
        hideCsvResults();
    });
}

// === Обработчики страницы 1 ===

function handleModelFileSelect(file) {
    if (!file.name.endsWith('.pkl')) {
        showError('Файл должен иметь расширение .pkl');
        return;
    }
    
    state.selectedModelFile = file;
    
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    
    toggleElement('upload-area', false);
    toggleElement('upload-info', true);
}

async function uploadModel() {
    if (!state.selectedModelFile) return;
    
    setButtonLoading('btn-upload', true);
    
    try {
        const result = await api.uploadModel(state.selectedModelFile);
        
        state.modelLoaded = true;
        state.modelType = result.model_type;
        updateModelStatusUI();
        
        const statusEl = document.getElementById('status-message');
        statusEl.textContent = result.message;
        statusEl.className = 'status-message success';
        toggleElement('upload-status', true);
        
        document.getElementById('model-type').textContent = result.model_type || 'Неизвестно';
        toggleElement('model-info-card', true);
        
        showSuccess(result.message);
    } catch (e) {
        const statusEl = document.getElementById('status-message');
        statusEl.textContent = e.message;
        statusEl.className = 'status-message error';
        toggleElement('upload-status', true);
    } finally {
        setButtonLoading('btn-upload', false);
    }
}

async function useDefaultModel() {
    // Если модель ещё не загружена — пробуем перепроверить статус
    if (!state.modelLoaded) {
        await checkModelStatus();
    }
    
    if (state.modelLoaded) {
        document.getElementById('model-type').textContent = state.modelType || 'RandomForestClassifier';
        toggleElement('model-info-card', true);
        showSuccess('Используется предустановленная модель');
        navigate('mode-select');
    } else {
        showError('Предустановленная модель не найдена. Сначала обучите модель или загрузите .pkl файл.');
    }
}

function downloadModel() {
    const API_BASE = window.API_URL || 'http://localhost:8000';
    window.open(`${API_BASE}/download-model`, '_blank');
}

// === Обработчики страницы 3 ===

async function handleSinglePredict(e) {
    e.preventDefault();
    
    if (!validateForm('single-predict-form')) {
        showError('Заполните все обязательные поля');
        return;
    }
    
    const data = getFormData('single-predict-form');
    
    setButtonLoading(e.submitter, true);
    
    try {
        const result = await api.predict([data]);
        showPredictionResult(result);
    } catch (e) {
        console.error('Ошибка предсказания:', e);
    } finally {
        setButtonLoading(e.submitter, false);
    }
}

function showPredictionResult(result) {
    const resultEl = document.getElementById('prediction-result');
    const statusEl = document.getElementById('result-status');
    const probEl = document.getElementById('probability-value');
    const cardEl = document.getElementById('result-card');
    
    const prediction = result.predictions && result.predictions[0];
    if (!prediction) {
        showError('Неожиданный формат ответа от сервера');
        return;
    }
    
    const isApproved = prediction.loan_status === 'Approved';
    
    statusEl.textContent = isApproved ? '✅ Одобрено' : '❌ Отказано';
    statusEl.className = `result-status ${isApproved ? 'approved' : 'rejected'}`;
    
    probEl.textContent = formatPercent(prediction.probability);
    
    cardEl.style.borderColor = isApproved ? 'var(--success-color)' : 'var(--danger-color)';
    
    toggleElement('prediction-result', true);
    
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hidePredictionResult() {
    toggleElement('prediction-result', false);
    document.getElementById('single-predict-form').reset();
}

// === Обработчики страницы 4 ===

function handleCsvFileSelect(file) {
    if (!file.name.endsWith('.csv')) {
        showError('Файл должен иметь расширение .csv');
        return;
    }
    
    state.selectedCsvFile = file;
    
    document.getElementById('csv-file-name').textContent = file.name;
    document.getElementById('csv-file-size').textContent = formatFileSize(file.size);
    
    toggleElement('csv-upload-area', false);
    toggleElement('csv-upload-info', true);
}

async function processCsv() {
    if (!state.selectedCsvFile) return;
    
    const hasTarget = document.getElementById('has-target').checked;
    
    setButtonLoading('btn-process-csv', true);
    
    try {
        const result = await api.predictCsv(state.selectedCsvFile, hasTarget);
        state.csvResults = result;
        showCsvResults(result);
    } catch (e) {
        console.error('Ошибка обработки CSV:', e);
    } finally {
        setButtonLoading('btn-process-csv', false);
    }
}

// Смена CSV файла
function changeCsvFile() {
    toggleElement('csv-results', false);
    toggleElement('csv-upload-card', true);
    toggleElement('csv-upload-area', true);
    toggleElement('csv-upload-info', false);
    document.getElementById('csv-file').value = '';
    state.selectedCsvFile = null;
    state.csvResults = null;
}

// Динамический рендер таблицы с данными из CSV
function renderDynamicTable(result, limit) {
    const thead = document.getElementById('results-thead');
    const tbody = document.getElementById('results-tbody');
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    const predictions = result.predictions || [];
    if (!predictions.length) return;
    
    // Определяем все колонки из первой записи
    const allKeys = Object.keys(predictions[0]);
    
    // Приоритетный порядок и человеческие названия
    const columnLabels = {
        'person_age': 'Возраст',
        'person_income': 'Доход',
        'person_emp_exp': 'Стаж',
        'cb_person_cred_hist_length': 'Кред. история',
        'person_gender': 'Пол',
        'person_education': 'Образование',
        'person_home_ownership': 'Жильё',
        'loan_amnt': 'Сумма кредита',
        'loan_int_rate': 'Ставка',
        'loan_percent_income': '% дохода',
        'credit_score': 'Рейтинг',
        'loan_intent': 'Цель',
        'previous_loan_defaults_on_file': 'Просрочки',
        'predicted_loan_status': 'Статус',
        'probability': 'Вероятность'
    };
    
    const priorityOrder = [
        'person_age', 'person_income', 'person_emp_exp', 'cb_person_cred_hist_length',
        'person_gender', 'person_education', 'person_home_ownership',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score',
        'loan_intent', 'previous_loan_defaults_on_file',
        'predicted_loan_status', 'probability'
    ];
    
    // Сортируем колонки: сначала приоритетные, потом остальные
    const sortedColumns = priorityOrder.filter(k => allKeys.includes(k));
    const remaining = allKeys.filter(k => !priorityOrder.includes(k));
    const columns = [...sortedColumns, ...remaining];
    
    // Заголовки
    thead.innerHTML = columns.map(col =>
        `<th>${columnLabels[col] || col}</th>`
    ).join('');
    
    // Данные
    const count = Math.min(predictions.length, limit);
    const showData = predictions.slice(0, count);
    
    tbody.innerHTML = showData.map(row => {
        const cells = columns.map(col => {
            let val = row[col];
            
            if (col === 'probability') {
                val = formatPercent(val);
            } else if (col === 'predicted_loan_status') {
                const isApproved = val === 'Approved';
                return `<td><span class="status-badge ${isApproved ? 'approved' : 'rejected'}">${isApproved ? 'Одобрено' : 'Отказано'}</span></td>`;
            } else if (col === 'person_income' || col === 'loan_amnt') {
                val = formatNumber(val);
            }
            
            return `<td>${val !== undefined && val !== null ? val : '-'}</td>`;
        }).join('');
        return `<tr>${cells}</tr>`;
    }).join('');
    
    return { columns, total: predictions.length };
}

function showCsvResults(result) {
    const predictions = result.predictions || [];
    
    // Сводка
    document.getElementById('total-rows').textContent = formatNumber(result.total_rows || predictions.length);
    
    const approvedCount = predictions.filter(p => p.predicted_loan_status === 'Approved').length;
    const rejectedCount = predictions.filter(p => p.predicted_loan_status === 'Rejected').length;
    document.getElementById('approved-count').textContent = formatNumber(approvedCount);
    document.getElementById('rejected-count').textContent = formatNumber(rejectedCount);
    
    const avgProb = predictions.length > 0
        ? predictions.reduce((sum, p) => sum + (p.probability || 0), 0) / predictions.length
        : 0;
    document.getElementById('avg-probability').textContent = formatPercent(avgProb);
    
    // ROC-AUC
    if (result.roc_auc !== null && result.roc_auc !== undefined) {
        document.getElementById('roc-auc-value').textContent = (result.roc_auc * 100).toFixed(1) + '%';
        toggleElement('roc-auc-display', true);
        
        // ROC-кривая
        const API_BASE = window.API_URL || 'http://localhost:8000';
        const rocImg = document.getElementById('roc-curve-img');
        rocImg.src = `${API_BASE}/roc-curve?t=${Date.now()}`;
        toggleElement('roc-curve-container', true);
    } else {
        toggleElement('roc-auc-display', false);
        toggleElement('roc-curve-container', false);
    }
    
    // Таблица с динамическими колонками
    const tableInfo = renderDynamicTable(result, 50);
    
    // Кнопка "Показать все"
    const showAllBtn = document.getElementById('btn-show-all');
    if (tableInfo && tableInfo.total > 50) {
        showAllBtn.style.display = 'block';
        showAllBtn.textContent = `Показать все ${tableInfo.total} записей`;
    } else {
        showAllBtn.style.display = 'none';
    }
    
    toggleElement('csv-results', true);
    toggleElement('csv-upload-card', false);
    
    document.getElementById('csv-results').scrollIntoView({ behavior: 'smooth' });
}

function showAllResults() {
    if (state.csvResults) {
        renderDynamicTable(state.csvResults, state.csvResults.predictions.length);
        document.getElementById('btn-show-all').style.display = 'none';
    }
}

function downloadResults() {
    if (!state.csvResults) return;
    
    const predictions = state.csvResults.predictions || [];
    if (!predictions.length) return;
    
    const allKeys = Object.keys(predictions[0]);
    
    const csvContent = [
        allKeys.join(','),
        ...predictions.map(row =>
            allKeys.map(k => {
                let val = row[k];
                if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
                    val = `"${val.replace(/"/g, '""')}"`;
                }
                return val ?? '';
            }).join(',')
        )
    ].join('\n');
    
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'predictions_results.csv';
    link.click();
    URL.revokeObjectURL(link.href);
}

function hideCsvResults() {
    toggleElement('csv-results', false);
    toggleElement('csv-upload-card', true);
    toggleElement('csv-upload-area', true);
    toggleElement('csv-upload-info', false);
    document.getElementById('csv-file').value = '';
    state.selectedCsvFile = null;
    state.csvResults = null;
}

// Запуск
init();