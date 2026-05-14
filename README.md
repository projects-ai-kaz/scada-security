# SCADA/ICS Қауіпсіздік Басқару Жүйесі

**Диссертация: Әбсаматов Ә.Ә. — 2025–2026**

## 🚀 Деплой (Railway + GitHub Pages)

### 1. GitHub-қа жүктеу
```bash
git init
git add .
git commit -m "SCADA Security System"
git remote add origin https://github.com/projects-ai-kaz/scada-security.git
git push -u origin main
```

### 2. Railway (Backend + ML)
- railway.app → GitHub аккаунтымен кіру
- New Project → scada-security репозиторийін таңдау
- Settings → Root Directory: `backend`
- Deploy → автоматты іске қосылады

### 3. GitHub Pages (Frontend)
- Settings → Pages → Branch: main → /frontend → Save
- URL: `https://projects-ai-kaz.github.io/scada-security/frontend/`

## 💻 Локальды іске қосу
```bash
# Backend
cd backend
pip install -r requirements.txt
python train.py      # ML оқыту (бірінші рет)
python main.py       # Сервер іске қосу

# Frontend
cd frontend
python3 -m http.server 3000
# → http://localhost:3000
```

## 📁 Жоба құрылымы
```
scada-security/
├── backend/
│   ├── main.py          ← FastAPI API
│   ├── train.py         ← ML оқыту (9 модель)
│   ├── predict.py       ← ML болжау
│   ├── dataset.py       ← Синтетикалық деректер
│   ├── startup.sh       ← Railway іске қосу скрипті
│   └── requirements.txt
├── frontend/
│   └── index.html       ← Қазақша интерфейс
├── railway.json         ← Railway конфигурация
└── README.md
```

## 🔗 Тірі сілтемелер
- **Frontend**: https://projects-ai-kaz.github.io/scada-security/frontend/
- **API**: https://scada-security-production.up.railway.app
- **API Docs**: https://scada-security-production.up.railway.app/docs
