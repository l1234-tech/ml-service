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
            prediction: probability > 0.5 ? 1 : 0,
            probability: probability,
            status: probability > 0.5 ? 'Approved' : 'Rejected'
        };
    },
    
    async predictCsv(file, hasTarget) {
        await delay(2000);
        const count = 100;
        const predictions = Array.from({length: count}, () => Math.random() > 0.5 ? 1 : 0);
        const probabilities = Array.from({length: count}, () => Math.random());
        
        return {
            predictions,
            probabilities,
            rows_processed: count,
            roc_auc: hasTarget ? 0.85 + Math.random() * 0.1 : null,
            confusion_matrix: hasTarget ? [[45, 5], [3, 47]] : null,
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
        formData.append('has_target', hasTarget.toString());
        
        const response = await fetch(`${API_BASE}/predict-csv`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Ошибка обработки CSV');
        }
        
        return await response.json();
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
