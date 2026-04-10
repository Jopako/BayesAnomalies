import sqlite3
import numpy as np
from datetime import datetime

# ---------------------------
# 1. LER DO SQLITE
# ---------------------------

DB_PATH = "historico.db"

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.execute("SELECT data, temp, press, hum FROM leituras ORDER BY data")
rows = cur.fetchall()
conn.close()

datas = []
temp  = []
press = []
hum   = []

for data_str, t, p, h in rows:
    datas.append(datetime.strptime(data_str, "%Y-%m-%d"))
    temp.append(t)
    press.append(p)
    hum.append(h)

temp  = np.array(temp)
press = np.array(press)
hum   = np.array(hum)
dia   = np.array([d.timetuple().tm_yday for d in datas])

X_bayes = np.column_stack((temp, press, hum))

# ---------------------------
# 2. LABELS SAZONAIS (z-score por época)
# ---------------------------

WINDOW_LABEL = 30
Z_THRESH     = 2.0

y = np.zeros(len(X_bayes))

for i in range(len(X_bayes)):
    viz = [j for j in range(len(X_bayes)) if j != i and abs(dia[j] - dia[i]) <= WINDOW_LABEL]
    if len(viz) < 3:
        viz = [j for j in range(len(X_bayes)) if j != i]

    z_t = abs(temp[i]  - np.mean(temp[viz]))  / (np.std(temp[viz])  + 1e-6)
    z_p = abs(press[i] - np.mean(press[viz])) / (np.std(press[viz]) + 1e-6)
    z_h = abs(hum[i]   - np.mean(hum[viz]))   / (np.std(hum[viz])   + 1e-6)

    if z_t > Z_THRESH or z_p > Z_THRESH or z_h > Z_THRESH:
        y[i] = 1

n_normais   = int((y == 0).sum())
n_anomalias = int((y == 1).sum())

# ---------------------------
# 3. NAIVE BAYES
# ---------------------------

def train_bayes(X, y):
    classes = np.unique(y)
    mean = {}; var = {}; prior = {}; const = {}
    for c in classes:
        X_c = X[y == c]
        if len(X_c) == 0:
            continue
        mean[c]  = np.mean(X_c, axis=0)
        var[c]   = np.var(X_c, axis=0) + 1e-6
        prior[c] = len(X_c) / len(X)
        const[c] = -0.5 * np.log(2 * np.pi * var[c])
    return classes, mean, var, prior, const

def predict_bayes(x, classes, mean, var, prior, const):
    scores = {}
    for c in classes:
        if c not in mean:
            continue
        ll = const[c] - ((x - mean[c]) ** 2) / (2 * var[c])
        scores[c] = np.log(prior[c]) + np.sum(ll)
    max_log    = max(scores.values())
    exp_scores = {c: np.exp(scores[c] - max_log) for c in scores}
    total      = sum(exp_scores.values())
    probs      = {c: v / total for c, v in exp_scores.items()}
    return max(probs, key=probs.get), probs, probs.get(1.0, 0.0)

# ---------------------------
# 4. SIGMOID (usado pelo z-score)
# ---------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------
# 5. NOVO DADO
# ---------------------------

nova_data = datetime(2024, 7, 18)
dia_novo  = nova_data.timetuple().tm_yday

# Apenas temp, press, hum para o Bayes
novo_dado_bayes = np.array([33.0, 1009.0, 76.0])

# ---------------------------
# 6. FILTRAR HISTÓRICO SAZONAL
# ---------------------------

WINDOW_PRED = 28

window_used = WINDOW_PRED
indices = [
    i for i, d in enumerate(datas)
    if abs(d.timetuple().tm_yday - dia_novo) <= WINDOW_PRED
]
if len(indices) < 5:
    indices = list(range(len(datas)))
    window_used = None

t_ctx = temp[indices]
p_ctx = press[indices]
h_ctx = hum[indices]

X_filtrado_bayes = X_bayes[indices, :]
y_filtrado       = y[indices]

# Se a janela sazonal não contém as duas classes, expande progressivamente
if len(np.unique(y_filtrado)) < 2:
    for extra in (30, 60, 90, 120, 180):
        indices_try = [
            i for i, d in enumerate(datas)
            if abs(d.timetuple().tm_yday - dia_novo) <= (WINDOW_PRED + extra)
        ]
        if len(indices_try) < 5:
            continue
        y_tmp = y[indices_try]
        if len(np.unique(y_tmp)) >= 2:
            window_used      = WINDOW_PRED + extra
            indices          = indices_try
            t_ctx            = temp[indices]
            p_ctx            = press[indices]
            h_ctx            = hum[indices]
            X_filtrado_bayes = X_bayes[indices, :]
            y_filtrado       = y_tmp
            break

# ---------------------------
# 7. TREINAR E PREDIZER
# ---------------------------

classes, mean, var, prior, const = train_bayes(X_filtrado_bayes, y_filtrado)
_, probs_bayes, grau_bayes = predict_bayes(novo_dado_bayes, classes, mean, var, prior, const)

# Z-scores do novo dado em relação ao contexto sazonal
z_t_novo = abs(novo_dado_bayes[0] - np.mean(t_ctx)) / (np.std(t_ctx) + 1e-6)
z_p_novo = abs(novo_dado_bayes[1] - np.mean(p_ctx)) / (np.std(p_ctx) + 1e-6)
z_h_novo = abs(novo_dado_bayes[2] - np.mean(h_ctx)) / (np.std(h_ctx) + 1e-6)
z_max    = max(z_t_novo, z_p_novo, z_h_novo)

# Grau z-score normalizado em [0, 1] (referência, não usado na decisão principal)
grau_zscore = float(sigmoid(z_max - Z_THRESH))

# ---------------------------
# 8. CLASSIFICAÇÃO FINAL
#    Bayes é soberano.
#    Z-score só entra como desempate na zona de atenção (Bayes incerto).
# ---------------------------

ZONA_NORMAL   = 0.3   # abaixo → Normal com confiança
ZONA_ANOMALIA = 0.7   # acima  → Anômalo com confiança
Z_CONFIRMA    = 3.0   # z alto na zona de atenção → confirma anomalia
Z_DESCARTA    = 1.0   # z baixo na zona de atenção → provável falso positivo

# grau_final sempre reflete o Bayes (sem distorção pelo z-score)
grau_final = float(grau_bayes)

if grau_bayes < ZONA_NORMAL:
    categoria = "Normal"
    detalhe   = "Bayes confiante (abaixo da zona de atenção)"

elif grau_bayes >= ZONA_ANOMALIA:
    categoria = "Anômalo"
    detalhe   = "Bayes confiante (acima da zona de atenção)"

else:
    # Zona de atenção: Bayes incerto → Z-score entra como desempate
    if z_max >= Z_CONFIRMA:
        categoria = "Anômalo"
        detalhe   = "Bayes incerto + Z-score alto → positivo confirmado"
    elif z_max <= Z_DESCARTA:
        categoria = "Normal"
        detalhe   = "Bayes incerto + Z-score baixo → provável falso positivo"
    else:
        categoria = "Atenção"
        detalhe   = "Bayes incerto, Z-score intermediário → revisar manualmente"

# ---------------------------
# 9. RESULTADO
# ---------------------------

print("=" * 56)
print(f"  Fonte    : SQLite → {DB_PATH}")
print(f"  Data     : {nova_data.strftime('%Y-%m-%d')}")
print(f"  Leitura  : Temp={novo_dado_bayes[0]}°C | Press={novo_dado_bayes[1]} hPa | Hum={novo_dado_bayes[2]}%")
if window_used is None:
    print(f"  Registros: {len(X_filtrado_bayes)} (toda a série histórica)")
else:
    print(f"  Registros: {len(X_filtrado_bayes)} (janela sazonal ±{window_used} dias)")
print("-" * 56)
if window_used is None:
    print("  Contexto histórico (toda a série):")
else:
    print(f"  Contexto histórico (±{window_used} dias):")
print(f"    Temp  : {np.mean(t_ctx):.1f}°C ± {np.std(t_ctx):.1f}°C")
print(f"    Press : {np.mean(p_ctx):.1f} hPa ± {np.std(p_ctx):.1f}")
print(f"    Hum   : {np.mean(h_ctx):.1f}% ± {np.std(h_ctx):.1f}%")
print("-" * 56)
print(f"  [BAYES - PRINCIPAL]")
print(f"    P(Normal)   : {probs_bayes.get(0.0, 0)*100:.1f}%")
print(f"    P(Anomalia) : {grau_bayes*100:.1f}%")
print(f"    Grau final  : {grau_final*100:.1f}%")
print("-" * 56)
print(f"  [Z-SCORE - SECUNDÁRIO]")
print(f"    T={z_t_novo:.2f} | P={z_p_novo:.2f} | H={z_h_novo:.2f}  (max={z_max:.2f})")
print(f"    Grau Z-Score : {grau_zscore*100:.1f}%")
zona_bayes = (
    "Confiante Normal"   if grau_bayes < ZONA_NORMAL   else
    "Confiante Anômalo"  if grau_bayes >= ZONA_ANOMALIA else
    "Incerto (atenção)"
)
print(f"    Zona Bayes   : {zona_bayes}")
usado = "Sim (desempate)" if ZONA_NORMAL <= grau_bayes < ZONA_ANOMALIA else "Não (Bayes confiante)"
print(f"    Z-score usado: {usado}")
print("-" * 56)
print(f"  Classificação : {categoria}")
print(f"  Motivo        : {detalhe}")
print("=" * 56)
print(f"\n  (Dataset: {n_normais} normais + {n_anomalias} anomalias históricas)")