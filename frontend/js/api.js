// ============================================
// API Layer - Заглушки и реальные вызовы
// ============================================

const USE_MOCK = false; // true = заглушки, false = реальный API
const API_BASE = window.API_URL || 'http://localhost:8000';

// Вспомогательная функция для API-вызовов
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Ошибка сервера: ${response.status}`);
        }
        
        return await response.json();
    } catch (e) {
        showError(e.message);
        throw e;
    }
}

// Заглушки (Mock)
const mockApi = {
    async health() {
        await delay(500);
        return {
            status: 'ok',
            model_loaded: false,
            model_type: null,
            version: '1.0.0'
        };
    },
    
    async uploadModel(file) {
        await delay(1500);
        return {
            success: true,
            message: 'Модель успешно загружена (MOCK)',
            model_type: 'GradientBoostingClassifier'
        };
    },
    
    async predict(data) {
        await delay(1000);
        const probability = Math.random();
        return {
            predictions: [{
                loan_status: probability > 0.5 ? 'Approved' : 'Rejected',
                probability: Math.round(probability * 10000) / 10000
            }]
        };
    },
    
    async predictCsv(file, hasTarget) {
        await delay(2000);
        const count = 100;
        const predictions = Array.from({length: count}, () => Math.random() > 0.5 ? 1 : 0);
        const probabilities = Array.from({length: count}, () => Math.random());
        
        return {
            predictions: predictions.map((pred, i) => ({
                predicted_loan_status: pred === 1 ? 'Approved' : 'Rejected',
                probability: Math.round(probabilities[i] * 10000) / 10000
            })),
            total_rows: count,
            rows_processed: count,
            roc_auc: hasTarget ? 0.85 + Math.random() * 0.1 : null,
            approved_count: predictions.filter(p => p === 1).length,
            rejected_count: predictions.filter(p => p === 0).length,
            avg_probability: probabilities.reduce((a, b) => a + b, 0) / count
        };
    }
};

// Реальный API
const realApi = {
    async health() {
        return apiCall(`${API_BASE}/health`);
    },
    
    async uploadModel(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/upload-model`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Ошибка загрузки модели');
        }
        
        return await response.json();
    },
    
    async predict(data) {
        return apiCall(`${API_BASE}/predict`, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },
    
    async predictCsv(file, hasTarget) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/predict-from-csv`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Ошибка обработки CSV');
        }
        
        const result = await response.json();
        
        // Передаём все поля из ответа бэкенда как есть
        const predictions = result.predictions || [];
        const approvedCount = predictions.filter(p => p.predicted_loan_status === 'Approved').length;
        const rejectedCount = predictions.filter(p => p.predicted_loan_status === 'Rejected').length;
        const avgProb = predictions.length > 0
            ? predictions.reduce((sum, p) => sum + (p.probability || 0), 0) / predictions.length
            : 0;
        
        return {
            predictions: predictions, // передаём все поля (person_age, person_income, и т.д.)
            total_rows: result.total_rows || predictions.length,
            rows_processed: result.total_rows || predictions.length,
            roc_auc: result.roc_auc || null,
            approved_count: approvedCount,
            rejected_count: rejectedCount,
            avg_probability: avgProb
        };
    }
};

// Выбор API
const api = USE_MOCK ? mockApi : realApi;

// Вспомогательные функции
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function showError(message) {
    showNotification(message, 'error');
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'info') {
    const container = document.getElementById('notifications');
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    container.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}