import sqlite3
import numpy as np
from datetime import datetime

# ---------------------------
# 1. LER DO SQLITE
# ---------------------------

DB_PATH = "historico.db"   # caminho para o banco

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

sin_t = np.sin(2 * np.pi * dia / 365)
cos_t = np.cos(2 * np.pi * dia / 365)
X = np.column_stack((temp, press, hum, sin_t, cos_t))

# ---------------------------
# 2. LABELS SAZONAIS (z-score por época)
# ---------------------------

WINDOW_LABEL = 30
Z_THRESH     = 2.0

y = np.zeros(len(X))

for i in range(len(X)):
    viz = [j for j in range(len(X)) if j != i and abs(dia[j] - dia[i]) <= WINDOW_LABEL]
    if len(viz) < 3:
        viz = [j for j in range(len(X)) if j != i]

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
# 4. GRAU VIA Z-SCORE (abordagem híbrida)
# ---------------------------

def zscore_grau(valor, referencia, threshold=Z_THRESH):
    z    = abs(valor - np.mean(referencia)) / (np.std(referencia) + 1e-6)
    grau = 1 / (1 + np.exp(-(z - threshold)))
    return z, grau

# ---------------------------
# 5. NOVO DADO
# ---------------------------

nova_data = datetime(2024, 7, 18)
dia_novo  = nova_data.timetuple().tm_yday
sin_novo  = np.sin(2 * np.pi * dia_novo / 365)
cos_novo  = np.cos(2 * np.pi * dia_novo / 365)

novo_dado = np.array([21, 1015, 58, sin_novo, cos_novo])

# ---------------------------
# 6. FILTRAR HISTÓRICO SAZONAL
# ---------------------------

WINDOW_PRED = 45

indices = [
    i for i, d in enumerate(datas)
    if abs(d.timetuple().tm_yday - dia_novo) <= WINDOW_PRED
]
if len(indices) < 5:
    indices = list(range(len(datas)))

t_ctx = temp[indices]
p_ctx = press[indices]
h_ctx = hum[indices]

X_filtrado = X[indices, :3]
y_filtrado = y[indices]

if len(np.unique(y_filtrado)) < 2:
    classe_faltando = 1.0 if 0.0 in np.unique(y_filtrado) else 0.0
    idx_extra = np.where(y == classe_faltando)[0][:5]
    if len(idx_extra) > 0:
        X_filtrado = np.vstack([X_filtrado, X[idx_extra, :3]])
        y_filtrado = np.concatenate([y_filtrado, y[idx_extra]])

# ---------------------------
# 7. TREINAR E PREDIZER
# ---------------------------

classes, mean, var, prior, const = train_bayes(X_filtrado, y_filtrado)
_, probs_bayes, grau_bayes = predict_bayes(novo_dado[:3], classes, mean, var, prior, const)

z_t, g_t = zscore_grau(novo_dado[0], t_ctx)
z_p, g_p = zscore_grau(novo_dado[1], p_ctx)
z_h, g_h = zscore_grau(novo_dado[2], h_ctx)

grau_zscore = max(g_t, g_p, g_h)
grau_final  = 0.70 * grau_zscore + 0.30 * grau_bayes
classe_final = 1 if grau_final >= 0.5 else 0

# ---------------------------
# 8. RESULTADO
# ---------------------------

print("=" * 52)
print(f"  Fonte    : SQLite → {DB_PATH}")
print(f"  Data     : {nova_data.strftime('%Y-%m-%d')}")
print(f"  Leitura  : Temp={novo_dado[0]}°C | Press={novo_dado[1]} hPa | Hum={novo_dado[2]}%")
print(f"  Registros: {len(X_filtrado)} (janela ±{WINDOW_PRED} dias)")
print("-" * 52)
print(f"  Contexto histórico (±{WINDOW_PRED} dias):")
print(f"    Temp  : {np.mean(t_ctx):.1f}°C ± {np.std(t_ctx):.1f}°C   z={z_t:.2f}σ")
print(f"    Press : {np.mean(p_ctx):.1f} hPa ± {np.std(p_ctx):.1f}  z={z_p:.2f}σ")
print(f"    Hum   : {np.mean(h_ctx):.1f}% ± {np.std(h_ctx):.1f}%    z={z_h:.2f}σ")
print("-" * 52)
print(f"  Grau z-score  : {grau_zscore*100:.1f}%  (peso 70%)")
print(f"  Grau Bayes    : {grau_bayes*100:.1f}%  (peso 30%)")
print(f"  Grau final    : {grau_final*100:.1f}%")
print("-" * 52)
print(f"  Classificação : {'Anômalo' if classe_final == 1 else 'Normal'} ({classe_final})")

if grau_final < 0.3:
    print("  Status        : 🟢 Normal")
elif grau_final < 0.7:
    print("  Status        : 🟡 Atenção")
else:
    print("  Status        : 🔴 Anomalia forte")

print("=" * 52)
print(f"\n  (Dataset: {n_normais} normais + {n_anomalias} anomalias históricas)")