# Backend — FastAPI приложение

## Запуск
```bash
cd backend
uvicorn main:app --reload
```

## Тесты
```bash
cd backend
python -m pytest tests/ -v
```

## Обучение модели
```bash
cd backend
python -m ml.train
```

## Зависимости
Управляются через `pyproject.toml` (uv/pip).