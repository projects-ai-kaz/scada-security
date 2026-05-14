"""
main.py — SCADA/ICS Cybersecurity Management API
Диссертация: Әбсаматов Ә.Ә., 2025–2026
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

app = FastAPI(
    title="SCADA/ICS Security Management API",
    description="Газ тасымалдау кәсіпорнының кибербезопасность жүйесі | Диссертация Әбсаматов Ә.Ә. 2025–2026",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── SCHEMAS ──────────────────────────────────────────────────────────────────
class ThreatScenario(BaseModel):
    id: int; name_ru: str; name_en: str; likelihood: int; impact: int; risk: int
    risk_level: str; target: str; consequences: str; mitigations: List[str]

class VulnerabilityGroup(BaseModel):
    id: int; name: str; description: str; priority: str; priority_score: int
    protocols: Optional[List[str]]; examples: List[str]; impact: str

class MLModelResult(BaseModel):
    id: str; name: str; category: str; accuracy: float; precision: float
    recall: float; f1_score: float; roc_auc: Optional[float]; is_best: bool; notes: str

class AuditItem(BaseModel):
    id: int; category: str; check_name: str; standard_ref: str
    status: str; description: str; risk_if_absent: Optional[str]

class ArchComp(BaseModel):
    name: str; icon: str

class ArchitectureLevel(BaseModel):
    level: int; name: str; color: str
    components: List[ArchComp]; threats: List[str]; protocols: List[str]

class DashboardStats(BaseModel):
    total_threats: int; critical_count: int; high_count: int; medium_count: int
    vulnerability_groups: int; best_model_name: str; best_model_f1: float
    best_model_auc: float; compliance_score: int; compliance_done: int
    compliance_partial: int; compliance_fail: int; generated_at: str

class PredictRequest(BaseModel):
    pressure_bar: float = Field(60.0, ge=0, le=200)
    flow_rate_m3h: float = Field(800.0, ge=0, le=5000)
    temperature_c: float = Field(15.0, ge=-50, le=150)
    valve_position_pct: float = Field(50.0, ge=0, le=100)
    rpm_compressor: float = Field(1500.0, ge=0, le=5000)
    pkt_size_bytes: float = Field(128.0, ge=20, le=1500)
    pkt_rate_pps: float = Field(10.0, ge=0, le=5000)
    inter_arrival_ms: float = Field(50.0, ge=0, le=1000)
    payload_entropy: float = Field(4.0, ge=0, le=8)
    tcp_flags_numeric: float = Field(16.0, ge=0, le=255)
    modbus_func_code: float = Field(3.0, ge=1, le=127)
    dnp3_app_ctrl: float = Field(32.0, ge=0, le=255)
    opc_tag_count: float = Field(5.0, ge=0, le=500)
    cmd_response_delta_ms: float = Field(5.0, ge=0, le=500)
    reg_write_freq: float = Field(0.05, ge=0, le=10)
    hour_of_day: float = Field(12.0, ge=0, le=23)
    src_ip_internal: float = Field(1.0, ge=0, le=1)
    dst_port_industrial: float = Field(1.0, ge=0, le=1)
    session_duration_s: float = Field(120.0, ge=0, le=3600)
    failed_auth_count: float = Field(0.0, ge=0, le=100)

class PredictResponse(BaseModel):
    prediction: int; label: str; probability: float; risk_score: float
    alert: bool; threshold_used: float; top_risk_features: List[Dict[str, Any]]
    model_used: str; timestamp: str

# ── DATA ─────────────────────────────────────────────────────────────────────
THREATS: List[ThreatScenario] = [
    ThreatScenario(id=1,name_ru="PLC жалған командалар (Command Injection)",name_en="PLC Command Injection",likelihood=4,impact=4,risk=16,risk_level="critical",target="PLC / Далалық деңгей",consequences="Қысым/ағынның қауіпті өзгеруі, авария, экологиялық зиян",mitigations=["Желіні сегменттеу","Командалық пакеттерді аутентификациялау (DNP3-SA)","ML-IDS мониторинг","Whitelist командалар"]),
    ThreatScenario(id=2,name_ru="PLC прошивкасын алмастыру (Firmware Tampering)",name_en="Firmware Tampering",likelihood=3,impact=4,risk=12,risk_level="high",target="PLC контроллерлер",consequences="Ұзақ мерзімді жасырын әсер, қорғаныс алгоритмдерін айналып өту",mitigations=["Прошивка тұтастығын тексеру","Нұсқаларды орталықтандырылған басқару","Конфигурация өзгерістерін бақылау"]),
    ThreatScenario(id=3,name_ru="Replay / Spoofing (Modbus/DNP3/OPC)",name_en="Protocol Replay/Spoofing",likelihood=3,impact=3,risk=9,risk_level="high",target="Өнеркәсіптік протоколдар",consequences="Атқарушы механизмдердің жалған қозғалысы, газ есебін бұзу",mitigations=["Пакеттерге уақыт белгілері","DNP3 Secure Authentication v5/v6","Аномалды трафик мониторингі"]),
    ThreatScenario(id=4,name_ru="SCADA/HMI-ге рұқсатсыз кіру",name_en="Unauthorized SCADA Access",likelihood=3,impact=3,risk=9,risk_level="high",target="SCADA серверлер / HMI",consequences="Басқаруды бұзу, оператордың қате шешімі",mitigations=["Көп факторлы аутентификация MFA","RBAC рөлдік модель","VPN + Zero-Trust"]),
    ThreatScenario(id=5,name_ru="MITM — телеметрияны бұрмалау",name_en="MITM Data Manipulation",likelihood=3,impact=3,risk=9,risk_level="high",target="Телеметрия арналары",consequences="Қате диагностика, ақаулықты кеш анықтау",mitigations=["TLS 1.3 / IPSec шифрлама","HMAC тұтастық тексеру","OT/IT DMZ сегментация"]),
    ThreatScenario(id=6,name_ru="Деректерді тыңдау (Eavesdropping)",name_en="Eavesdropping",likelihood=4,impact=2,risk=8,risk_level="medium",target="Барлық арналар",consequences="Деректер ашылуы, күрделі шабуылдарға дайындық",mitigations=["Барлық арналарды шифрлау","WPA3 сымсыз қорғаныс"]),
    ThreatScenario(id=7,name_ru="DoS/DDoS шабуылы",name_en="DoS/DDoS",likelihood=3,impact=2,risk=6,risk_level="medium",target="Диспетчерлік сегмент",consequences="Мониторинг тоқтауы, реакция кешігуі",mitigations=["QoS / Rate limiting","Резервтік арналар","Hot Standby SCADA"]),
    ThreatScenario(id=8,name_ru="Ransomware (инженерлік станциялар)",name_en="Ransomware",likelihood=2,impact=3,risk=6,risk_level="medium",target="Инженерлік станциялар",consequences="Деректерге қол жеткізуді жоғалту",mitigations=["Application Whitelisting","Оқшаулау","3-2-1 резервтік көшіру"]),
]

VULNERABILITIES: List[VulnerabilityGroup] = [
    VulnerabilityGroup(id=1,name="Протокол осалдықтары",description="Modbus, DNP3, OPC — шифрлау мен аутентификациясыз",priority="Критикалық",priority_score=4,protocols=["Modbus/TCP","DNP3","OPC DA/UA","IEC 104"],examples=["Аутентификация жоқ","Ашық деректер","Replay қорғанысы жоқ"],impact="Командаларды ұстап алу, жалған деректер, replay"),
    VulnerabilityGroup(id=2,name="Желілік осалдықтар",description="SCADA/OT және IT желілерінің жеткіліксіз сегментациясы",priority="Критикалық",priority_score=4,protocols=["Ethernet","TCP/IP"],examples=["DMZ жоқ","Плоская топология","Қате ACL"],impact="Рұқсатсыз кіру, желіде таралу"),
    VulnerabilityGroup(id=3,name="Сымсыз арна осалдықтары",description="Радио, 4G/5G, спутник ашық ортада",priority="Жоғары",priority_score=3,protocols=["4G/5G","Wi-Fi","Satellite","WirelessHART"],examples=["Сигнал ұстау","Jamming","Evil Twin"],impact="Байланысты блоктау, деректер ұрлануы"),
    VulnerabilityGroup(id=4,name="Аутентификация осалдықтары",description="Default немесе әлсіз тіркелгі деректері",priority="Жоғары",priority_score=3,protocols=None,examples=["Default парольдер","MFA жоқ","Артық привилегиялар"],impact="PLC/SCADA толық бақылау"),
    VulnerabilityGroup(id=5,name="Аппараттық осалдықтар (Legacy)",description="Ескі жабдық, жаңарту мүмкіндігі жоқ",priority="Жоғары",priority_score=3,protocols=None,examples=["CVE патчтары жоқ","Ескі ОЖ","Шифрлама жоқ"],impact="Белгілі осалдықтарды пайдалану"),
    VulnerabilityGroup(id=6,name="Конфигурациялық осалдықтар",description="Қате баптама",priority="Орташа",priority_score=2,protocols=None,examples=["Ашық порттар","Қате Firewall","Debug режим"],impact="DoS, шабуыл беті кеңеюі"),
    VulnerabilityGroup(id=7,name="Мониторинг жеткіліксіздігі",description="Үздіксіз бақылау жоқ",priority="Орташа",priority_score=2,protocols=None,examples=["IDS жоқ","SIEM жоқ","Журналдар нашар"],impact="Шабуылды кеш анықтау"),
    VulnerabilityGroup(id=8,name="Адам факторы",description="Қызметкер қателіктері",priority="Орташа",priority_score=2,protocols=None,examples=["Фишинг","USB қауіптері","Пароль бұзу"],impact="Жаңылыс конфигурация, деректер ашылуы"),
]

ML_MODELS: List[MLModelResult] = [
    MLModelResult(id="lr",     name="Logistic Regression",           category="Supervised",  accuracy=0.815,precision=0.984,recall=0.157,f1_score=0.271,roc_auc=0.710,is_best=False,notes="Жоғары Accuracy, өте төмен Recall"),
    MLModelResult(id="rf_base",name="Random Forest (base)",           category="Supervised",  accuracy=0.846,precision=1.000,recall=0.296,f1_score=0.457,roc_auc=0.849,is_best=False,notes="Тамаша AUC, дефолтты конфигурация"),
    MLModelResult(id="if",     name="Isolation Forest",               category="Unsupervised",accuracy=0.738,precision=0.307,recall=0.159,f1_score=0.209,roc_auc=None, is_best=False,notes="Таңбасыз деректер үшін"),
    MLModelResult(id="lr_bal", name="LR balanced",                   category="Supervised",  accuracy=0.617,precision=0.291,recall=0.524,f1_score=0.374,roc_auc=None, is_best=False,notes="Recall жақсарды"),
    MLModelResult(id="lr_bt",  name="LR balanced + tuned",           category="Supervised",  accuracy=0.489,precision=0.296,recall=0.968,f1_score=0.453,roc_auc=None, is_best=False,notes="Максималды Recall"),
    MLModelResult(id="rf_bal", name="RF balanced",                   category="Supervised",  accuracy=0.630,precision=0.363,recall=0.916,f1_score=0.520,roc_auc=0.831,is_best=False,notes="Жақсы Recall + AUC"),
    MLModelResult(id="rf_bt",  name="RF balanced + tuned",           category="Supervised",  accuracy=0.752,precision=0.454,recall=0.659,f1_score=0.537,roc_auc=0.838,is_best=False,notes="Теңгерімді нәтиже"),
    MLModelResult(id="rf_bs",  name="RF balanced_sub + tuned ★",    category="Supervised",  accuracy=0.753,precision=0.456,recall=0.665,f1_score=0.541,roc_auc=0.838,is_best=True, notes="★ Өндірістік ортаға ұсынылады"),
    MLModelResult(id="hybrid", name="Hybrid (RF+Rules+IF)",          category="Hybrid",      accuracy=0.857,precision=0.903,recall=0.385,f1_score=0.540,roc_auc=0.766,is_best=False,notes="Жоғары Precision, Recall төмен"),
]

AUDIT_ITEMS: List[AuditItem] = [
    AuditItem(id=1, category="Желілік қауіпсіздік",     check_name="OT/IT брандмауэр",                   standard_ref="IEC 62443 SR 5.1",    status="done",   description="OT және IT арасында брандмауэр бар",                    risk_if_absent=None),
    AuditItem(id=2, category="Желілік қауіпсіздік",     check_name="SCADA/Корп. желі DMZ",               standard_ref="IEC 62443 SL 3",      status="partial",description="VLAN бар, терең сүзгі жоқ",                          risk_if_absent="Боковое перемещение"),
    AuditItem(id=3, category="Желілік қауіпсіздік",     check_name="Modbus/DNP3 шифрлама",               standard_ref="IEC 62443 SR 3.1",    status="fail",   description="Протоколдар ашық жіберіледі",                           risk_if_absent="R=9-16, replay/MITM"),
    AuditItem(id=4, category="Қол жеткізу басқару",     check_name="RBAC рөлдік модель",                 standard_ref="ISO 27001 A.9.1",     status="done",   description="RBAC операторлар үшін бар",                            risk_if_absent=None),
    AuditItem(id=5, category="Қол жеткізу басқару",     check_name="PLC/RTU default парольдерін ауыстыру",standard_ref="IEC 62443 SR 1.1",   status="done",   description="Барлық парольдер ауыстырылды",                          risk_if_absent=None),
    AuditItem(id=6, category="Қол жеткізу басқару",     check_name="MFA аутентификация",                 standard_ref="NIST SP 800-82",      status="fail",   description="MFA SCADA операторлары үшін жоқ",                      risk_if_absent="R=9 рұқсатсыз кіру"),
    AuditItem(id=7, category="Мониторинг",               check_name="OT IDS жүйесі",                      standard_ref="IEC 62443 SR 6.2",    status="partial",description="Сигнатуралық IDS, ML жоқ",                           risk_if_absent="Жаңа шабуылдар өтеді"),
    AuditItem(id=8, category="Мониторинг",               check_name="ML аномалия детекторы",              standard_ref="NIST CSF DE.AE-1",    status="fail",   description="RF bs+tuned: F1=0.541, AUC=0.838",                     risk_if_absent="Шабуылдар жіберіледі"),
    AuditItem(id=9, category="Мониторинг",               check_name="SIEM журналдау",                     standard_ref="ISO 27001 A.12.4",    status="done",   description="SIEM SCADA серверлерімен интеграцияланған",             risk_if_absent=None),
    AuditItem(id=10,category="Физикалық қорғаныс",       check_name="PLC/RTU физикалық бақылау",          standard_ref="ISO 27001 A.11.1",    status="done",   description="Физикалық қауіпсіздік аймақтары бар",                  risk_if_absent=None),
    AuditItem(id=11,category="Физикалық қорғаныс",       check_name="Берудегі шифрлама",                  standard_ref="ISO 27001 A.10.1",    status="done",   description="Корпоративтік деңгей үшін VPN/TLS",                    risk_if_absent=None),
    AuditItem(id=12,category="Төзімділік",                check_name="Байланыс резервтеу",                 standard_ref="IEC 62443 SR 7.1",    status="partial",description="Резервтік арна бар, автоауысу жоқ",                   risk_if_absent="Бір сәтсіздік нүктесі"),
    AuditItem(id=13,category="Төзімділік",                check_name="ИҚ бойынша оқыту",                  standard_ref="ISO 27001 A.7.2.2",   status="partial",description="Нерегулярные тренинги",                              risk_if_absent="Фишинг тиімді"),
    AuditItem(id=14,category="Төзімділік",                check_name="IR Plan инциденттерге жауап",        standard_ref="IEC 62443 SR 6.1",    status="fail",   description="IR Plan жасалмаған",                                   risk_if_absent="Хаотикалық жауап"),
    AuditItem(id=15,category="Төзімділік",                check_name="PLC firmware patch management",      standard_ref="IEC 62443 SR 7.6",    status="fail",   description="Патч-менеджмент жүйеленбеген",                         risk_if_absent="CVE ашық қалады"),
]

ARCHITECTURE = [
    {"level":3,"name":"Корпоративтік деңгей (ERP / BI)","color":"#185FA5","components":[{"name":"ERP жүйесі"},{"name":"Аналитика"},{"name":"DWH"},{"name":"Корп. желі"}],"threats":["Ransomware (R=6)","Деректер ағып кетуі"],"protocols":["HTTPS","VPN","REST API"]},
    {"level":2,"name":"Басқару деңгейі (SCADA / HMI)","color":"#854F0B","components":[{"name":"SCADA серверлер"},{"name":"HMI станциялар"},{"name":"Пром. коммутаторлар"},{"name":"Историан"}],"threats":["HMI рұқсатсыз кіру (R=9)","DoS (R=6)","MITM (R=9)"],"protocols":["Modbus/TCP","DNP3","OPC DA"]},
    {"level":1,"name":"Далалық деңгей (PLC / RTU)","color":"#A32D2D","components":[{"name":"PLC контроллерлер"},{"name":"Датчиктер"},{"name":"Клапандар"},{"name":"RTU блоктары"}],"threats":["Жалған командалар (R=16)","Прошивка алмастыру (R=12)","Replay (R=9)"],"protocols":["Modbus RTU","HART","Profibus"]},
]

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"system":"SCADA/ICS Cybersecurity Management System","version":"1.0.0","docs":"/docs"}

@app.get("/api/dashboard", response_model=DashboardStats)
async def get_dashboard():
    done=sum(1 for a in AUDIT_ITEMS if a.status=="done")
    part=sum(1 for a in AUDIT_ITEMS if a.status=="partial")
    fail=sum(1 for a in AUDIT_ITEMS if a.status=="fail")
    score=int((done+part*0.5)/len(AUDIT_ITEMS)*100)
    best=next(m for m in ML_MODELS if m.is_best)
    return DashboardStats(total_threats=len(THREATS),critical_count=sum(1 for t in THREATS if t.risk_level=="critical"),high_count=sum(1 for t in THREATS if t.risk_level=="high"),medium_count=sum(1 for t in THREATS if t.risk_level=="medium"),vulnerability_groups=len(VULNERABILITIES),best_model_name=best.name,best_model_f1=best.f1_score,best_model_auc=best.roc_auc or 0.0,compliance_score=score,compliance_done=done,compliance_partial=part,compliance_fail=fail,generated_at=datetime.now().isoformat())

@app.get("/api/threats", response_model=List[ThreatScenario])
async def get_threats(): return THREATS

@app.get("/api/threats/{threat_id}", response_model=ThreatScenario)
async def get_threat(threat_id: int):
    for t in THREATS:
        if t.id==threat_id: return t
    raise HTTPException(404,"Not found")

@app.get("/api/vulnerabilities", response_model=List[VulnerabilityGroup])
async def get_vulns(): return VULNERABILITIES

@app.get("/api/ml-models", response_model=List[MLModelResult])
async def get_models(): return ML_MODELS

@app.get("/api/audit", response_model=List[AuditItem])
async def get_audit(): return AUDIT_ITEMS

@app.get("/api/architecture")
async def get_arch(): return ARCHITECTURE

@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        from predict import predict_single
        result = predict_single(req.model_dump())
        return PredictResponse(**result, model_used="RF balanced_subsample + tuned", timestamp=datetime.now().isoformat())
    except FileNotFoundError:
        raise HTTPException(503,"Model not trained yet. Run: python train.py")
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get("/api/predict/model-info")
async def model_info():
    try:
        from predict import get_model_info
        return get_model_info()
    except FileNotFoundError:
        return {"status":"not_trained","message":"Run: python train.py"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
