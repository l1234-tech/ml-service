# Backend — FastAPI приложение

## Запуск
```bash
cd backend
uv run uvicorn api.main:app --reload
```

## Тесты
```bash
cd backend
python -m pytest tests/ -v
```

## Обучение модели
```bash
cd backend
uv run python ml/train.py
```

## Зависимости
Управляются через `pyproject.toml` (uv/pip)