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
function init() {
    setupEventListeners();
    checkModelStatus();
    navigate('model-select');
}

// Роутинг
function navigate(page) {
    if (!pages[page]) return;
    
    // Скрываем все страницы
    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('active');
    });
    
    // Показываем нужную страницу
    const pageElement = document.getElementById(pages[page].id);
    if (pageElement) {
        pageElement.classList.add('active');
    }
    
    state.currentPage = page;
    
    // Обновляем заголовок
    document.title = `${pages[page].title} - Сервис одобрения ипотеки`;
}

// Проверка статуса модели
async function checkModelStatus() {
    try {
        const status = await api.health();
        state.modelLoaded = status.model_loaded;
        state.modelType = status.model_type;
        updateModelStatusUI();
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
    // === Страница 1: Выбор модели ===
    
    // Выбор файла
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('model-file');
    const btnSelectFile = document.getElementById('btn-select-file');
    
    btnSelectFile.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag & Drop
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
    
    // Загрузка модели
    document.getElementById('btn-upload').addEventListener('click', uploadModel);
    
    // Далее
    document.getElementById('btn-next-to-mode').addEventListener('click', () => {
        navigate('mode-select');
    });
    
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
    
    // Выбор CSV файла
    const csvUploadArea = document.getElementById('csv-upload-area');
    const csvFileInput = document.getElementById('csv-file');
    const btnSelectCsv = document.getElementById('btn-select-csv');
    
    btnSelectCsv.addEventListener('click', () => csvFileInput.click());
    csvUploadArea.addEventListener('click', () => csvFileInput.click());
    
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
    
    // Обработка CSV
    document.getElementById('btn-process-csv').addEventListener('click', processCsv);
    
    // Скачать результаты
    document.getElementById('btn-download-csv').addEventListener('click', downloadResults);
    
    // Показать все
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
    
    // Показываем информацию о файле
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
        
        // Показываем статус
        const statusEl = document.getElementById('status-message');
        statusEl.textContent = result.message;
        statusEl.className = 'status-message success';
        toggleElement('upload-status', true);
        
        // Показываем информацию о модели
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
        const result = await api.predict(data);
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
    
    const isApproved = result.prediction === 1;
    
    statusEl.textContent = isApproved ? '✅ Approved' : '❌ Rejected';
    statusEl.className = `result-status ${isApproved ? 'approved' : 'rejected'}`;
    
    probEl.textContent = formatPercent(result.probability);
    
    cardEl.style.borderColor = isApproved ? 'var(--success-color)' : 'var(--danger-color)';
    
    toggleElement('prediction-result', true);
    
    // Прокручиваем к результату
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
    
    // Показываем информацию о файле
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

function showCsvResults(result) {
    // Сводка
    document.getElementById('total-rows').textContent = formatNumber(result.rows_processed);
    document.getElementById('approved-count').textContent = formatNumber(result.approved_count);
    document.getElementById('rejected-count').textContent = formatNumber(result.rejected_count);
    document.getElementById('avg-probability').textContent = formatPercent(result.avg_probability);
    
    // ROC-AUC
    if (result.roc_auc !== null && result.roc_auc !== undefined) {
        document.getElementById('roc-auc-value').textContent = result.roc_auc.toFixed(3);
        toggleElement('roc-auc-display', true);
    } else {
        toggleElement('roc-auc-display', false);
    }
    
    // Таблица (первые 50)
    renderResultsTable(result, 50);
    
    // Кнопка "Показать все"
    const showAllBtn = document.getElementById('btn-show-all');
    if (result.rows_processed > 50) {
        showAllBtn.style.display = 'block';
        showAllBtn.textContent = `Показать все ${result.rows_processed} записей`;
    } else {
        showAllBtn.style.display = 'none';
    }
    
    toggleElement('csv-results', true);
    
    // Прокручиваем к результатам
    document.getElementById('csv-results').scrollIntoView({ behavior: 'smooth' });
}

function renderResultsTable(result, limit) {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    const count = Math.min(result.predictions.length, limit);
    
    for (let i = 0; i < count; i++) {
        const tr = document.createElement('tr');
        
        // Номер
        const tdNum = document.createElement('td');
        tdNum.textContent = i + 1;
        tr.appendChild(tdNum);
        
        // Данные (заглушки для демо)
        const tdAge = document.createElement('td');
        tdAge.textContent = '-';
        tr.appendChild(tdAge);
        
        const tdIncome = document.createElement('td');
        tdIncome.textContent = '-';
        tr.appendChild(tdIncome);
        
        const tdAmount = document.createElement('td');
        tdAmount.textContent = '-';
        tr.appendChild(tdAmount);
        
        const tdScore = document.createElement('td');
        tdScore.textContent = '-';
        tr.appendChild(tdScore);
        
        // Статус
        const tdStatus = document.createElement('td');
        const isApproved = result.predictions[i] === 1;
        tdStatus.innerHTML = `<span class="status-badge ${isApproved ? 'approved' : 'rejected'}">
            ${isApproved ? 'Одобрено' : 'Отказано'}
        </span>`;
        tr.appendChild(tdStatus);
        
        // Вероятность
        const tdProb = document.createElement('td');
        tdProb.textContent = formatPercent(result.probabilities[i]);
        tr.appendChild(tdProb);
        
        tbody.appendChild(tr);
    }
}

function showAllResults() {
    if (state.csvResults) {
        renderResultsTable(state.csvResults, state.csvResults.rows_processed);
        document.getElementById('btn-show-all').style.display = 'none';
    }
}

function downloadResults() {
    if (!state.csvResults) return;
    
    const data = state.csvResults.predictions.map((pred, i) => ({
        '№': i + 1,
        'Prediction': pred === 1 ? 'Approved' : 'Rejected',
        'Probability': state.csvResults.probabilities[i].toFixed(4)
    }));
    
    downloadCSV(data, 'predictions.csv');
}

function hideCsvResults() {
    toggleElement('csv-results', false);
    toggleElement('csv-upload-area', true);
    toggleElement('csv-upload-info', false);
    document.getElementById('csv-file').value = '';
    state.selectedCsvFile = null;
    state.csvResults = null;
}

// Запуск
init();
