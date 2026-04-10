# Detecção de Anomalias Climáticas
### Naive Bayes + Z-Score Sazonal · `bayes.py` · `historico.db`

---

## Visão Geral

O script analisa leituras meteorológicas (temperatura, pressão e umidade) e classifica um novo ponto como **Normal** ou **Anômalo**. A detecção combina duas abordagens:

| Abordagem | Peso | Descrição |
|-----------|------|-----------|
| Naive Bayes Gaussiano | Principal | Classificador probabilístico treinado no subconjunto sazonal |
| Z-Score Sazonal | Trava | Se o novo ponto ficar “absurdo” (muitos desvios-padrão), força anomalia mesmo se o Bayes estiver baixo |

**Regra de decisão (simplificada):** Bayes decide; z-score só “trava” anomalias muito fora do contexto.

---

## Banco de Dados — `historico.db`

O histórico é armazenado em SQLite, substituindo o CSV original.

### Tabela `leituras`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | INTEGER | Chave primária auto-incrementada |
| `data` | TEXT | Data no formato `YYYY-MM-DD` — valor único |
| `temp` | REAL | Temperatura em °C |
| `press` | REAL | Pressão atmosférica em hPa |
| `hum` | REAL | Umidade relativa em % |

### Inserir novos registros

```python
import sqlite3

conn = sqlite3.connect("historico.db")
conn.execute(
    "INSERT OR IGNORE INTO leituras (data, temp, press, hum) VALUES (?, ?, ?, ?)",
    ("2025-03-01", 25.0, 1013, 72)
)
conn.commit()
conn.close()
```

> O `INSERT OR IGNORE` evita duplicatas automaticamente pela constraint `UNIQUE` na coluna `data`.

---

## Fluxo do Código

### Seção 1 — Leitura do SQLite
Conecta ao banco, executa `SELECT` ordenado por data e popula arrays NumPy. Substitui o antigo `csv.DictReader`.

### Seção 2 — Labels Sazonais
Para cada registro, identifica vizinhos dentro de `±WINDOW_LABEL` dias e calcula z-score de cada feature. Se qualquer z-score superar `Z_THRESH`, o ponto é marcado como anomalia (`y=1`).

> **Bug corrigido:** o critério original usava limiares fixos (`>35°C` ou `<10°C`) que nunca eram atingidos no dataset — `y` ficava sempre `0` e o Bayes jamais via a classe anomalia no treino.

### Seção 3 — Naive Bayes Gaussiano
Implementação manual. Calcula média, variância, prior e constante log-normalizada para cada classe. A predição retorna a classe de maior log-posterior e a probabilidade da classe anomalia.

### Seção 4 — Grau Híbrido
O z-score do ponto novo é convertido em grau `[0, 1]` via sigmoid centrada em `Z_THRESH`. Ele é usado como **trava**: se `z_max >= Z_FORCE`, o script força a classificação como anomalia, mesmo que a probabilidade Bayes seja baixa.

### Seção 5 — Novo Dado
Define o ponto a classificar: data, temperatura, pressão, umidade e os componentes sazonais `sin/cos` calculados a partir do dia do ano.

### Seção 6 — Filtro Sazonal
Filtra o histórico para `±WINDOW_PRED` dias do dia do ano do novo dado. O Bayes é treinado **sem** `sin/cos` — que teriam variância nula nessa janela estreita.

### Seção 7 — Treino e Predição
Treina o Bayes no subconjunto filtrado, calcula z-scores individuais e combina os graus parciais na fórmula final.

### Seção 8 — Resultado

```
grau_final < 0.3   →  🟢 Normal
grau_final < 0.7   →  🟡 Atenção
grau_final >= 0.7  →  🔴 Anomalia forte
```

---

## Por que SQLite é melhor que CSV

| Aspecto | SQLite | CSV |
|---------|--------|-----|
| **Integridade** | Constraints (`UNIQUE`, `NOT NULL`) bloqueiam dados inválidos | Aceita qualquer coisa |
| **Consultas** | Filtros e agregações via SQL sem carregar tudo na memória | Varredura total obrigatória |
| **Performance** | Índices — busca em O(log n) | Sempre O(n) |
| **Concorrência** | Leituras simultâneas; escritas transacionais | Sem controle algum |
| **Escalabilidade** | Milhões de registros sem degradação | Problemático acima de poucos MB |
| **Dependências** | `sqlite3` é nativo do Python | `csv` também nativo, mas sem nenhuma dessas garantias |
| **Extensível** | Novas colunas, tabelas e views sem quebrar código existente | Requer reescrita manual |

> Em resumo: o CSV é conveniente para dados estáticos e pequenos. O SQLite entrega tudo isso com segurança, velocidade e flexibilidade — ainda sendo um arquivo único, sem servidor.

---

## Como Usar

### Pré-requisitos

- Python 3.8+
- NumPy: `pip install numpy`
- `sqlite3` — nativo, nenhuma instalação necessária

### Execução

```bash
python3 bayes.py
```

### Alterar o ponto analisado

Edite a **Seção 5** no script:

```python
nova_data = datetime(2024, 7, 18)                          # data desejada
novo_dado = np.array([21, 1015, 58, sin_novo, cos_novo])   # temp, press, hum
```

### Consultar o banco

```bash
sqlite3 historico.db
```
```sql
.tables
SELECT * FROM leituras ORDER BY data;
SELECT COUNT(*) FROM leituras;
.quit
```

---

## Parâmetros Ajustáveis

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `WINDOW_LABEL` | `30` | Janela em dias para calcular labels de anomalia no histórico |
| `Z_THRESH` | `2.0` | Desvios padrão para classificar como anomalia |
| `WINDOW_PRED` | `45` | Janela em dias para o contexto sazonal na predição |

---

## Arquivos

```
.
├── bayes.py             # script principal
├── historico.db         # banco SQLite com as leituras históricas
└── README.md
```
