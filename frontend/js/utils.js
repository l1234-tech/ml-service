// ============================================
// Utilities - Вспомогательные функции
// ============================================

/**
 * Форматирует число как процент
 */
function formatPercent(value) {
    return (value * 100).toFixed(1) + '%';
}

/**
 * Форматирует число с разделителями тысяч
 */
function formatNumber(value) {
    return new Intl.NumberFormat('ru-RU').format(value);
}

/**
 * Форматирует размер файла
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Скачивает данные как CSV файл
 */
function downloadCSV(data, filename = 'results.csv') {
    const csv = convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

/**
 * Конвертирует массив объектов в CSV строку
 */
function convertToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const rows = data.map(obj => 
        headers.map(header => {
            const value = obj[header];
            // Экранируем значения с запятыми и кавычками
            if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                return `"${value.replace(/"/g, '""')}"`;
            }
            return value;
        }).join(',')
    );
    
    return [headers.join(','), ...rows].join('\n');
}

/**
 * Парсит CSV файл
 */
function parseCSV(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split('\n').filter(line => line.trim());
            const headers = lines[0].split(',').map(h => h.trim());
            
            const data = lines.slice(1).map(line => {
                const values = line.split(',');
                const obj = {};
                headers.forEach((header, index) => {
                    obj[header] = values[index]?.trim() || '';
                });
                return obj;
            });
            
            resolve({ headers, data });
        };
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

/**
 * Показывает/скрывает элемент
 */
function toggleElement(element, show) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    if (element) {
        element.style.display = show ? 'block' : 'none';
    }
}

/**
 * Устанавливает текст элемента
 */
function setText(elementId, text) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = text;
    }
}

/**
 * Устанавливает HTML элемента
 */
function setHTML(elementId, html) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = html;
    }
}

/**
 * Добавляет класс элементу
 */
function addClass(elementId, className) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add(className);
    }
}

/**
 * Удаляет класс у элемента
 */
function removeClass(elementId, className) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove(className);
    }
}

/**
 * Проверяет, есть ли класс у элемента
 */
function hasClass(elementId, className) {
    const element = document.getElementById(elementId);
    return element ? element.classList.contains(className) : false;
}

/**
 * Дебаунс функция
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Валидирует форму
 */
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;
    
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.style.borderColor = 'var(--danger-color)';
        } else {
            input.style.borderColor = '';
        }
    });
    
    return isValid;
}

/**
 * Собирает данные формы в объект
 */
function getFormData(formId) {
    const form = document.getElementById(formId);
    if (!form) return {};
    
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        // Преобразуем числовые значения
        if (!isNaN(value) && value !== '') {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }
    
    return data;
}

/**
 * Создаёт элемент таблицы
 */
function createTableRow(data, index) {
    const tr = document.createElement('tr');
    
    // Номер
    const tdIndex = document.createElement('td');
    tdIndex.textContent = index + 1;
    tr.appendChild(tdIndex);
    
    // Данные
    Object.values(data).forEach(value => {
        const td = document.createElement('td');
        td.textContent = value;
        tr.appendChild(td);
    });
    
    return tr;
}

/**
 * Очищает таблицу
 */
function clearTable(tbodyId) {
    const tbody = document.getElementById(tbodyId);
    if (tbody) {
        tbody.innerHTML = '';
    }
}

/**
 * Показывает загрузку на кнопке
 */
function setButtonLoading(buttonId, loading) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = '<span class="spinner"></span> Загрузка...';
    } else {
        button.disabled = false;
        if (button.dataset.originalText) {
            button.innerHTML = button.dataset.originalText;
        }
    }
}
