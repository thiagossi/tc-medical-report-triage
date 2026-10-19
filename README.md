# Triagem de Relatos Médicos: Tech Challenge Fase 3 (FIAP MLET)

Classificador de texto (NLP) que lê o relato de um paciente e classifica a urgência em
**normal / atenção / urgente**. O modelo é servido por uma API REST (FastAPI) em Docker,
com CI/CD no GitHub Actions, retreino orquestrado pelo Airflow e monitoramento com
Prometheus + Grafana.

> Documento em construção: cada etapa do desafio completa este README.

---

## 1. Problema

Em um pronto-atendimento, quem chega primeiro nem sempre é quem precisa de atendimento
primeiro. A proposta é um **apoio à triagem**: a partir da frase em que o paciente descreve
o que sente, o sistema sugere um nível de urgência para **ordenar a fila**. A decisão final
continua sendo de um profissional de saúde.

## 2. Dataset

- **Fonte:** [Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) (Kaggle).
- **Arquivo usado:** `overview-of-recordings.csv`, com 6.661 linhas. Colunas usadas: `phrase` (texto) e `prompt` (categoria do sintoma).
- **Atenção:** são só **703 frases únicas**, cada uma gravada cerca de 10 vezes por pessoas diferentes.

O arquivo não fica no repositório. Para rodar o treino, baixe o dataset no Kaggle e coloque
o CSV em `data/raw/overview-of-recordings.csv`.

### Das 25 categorias para 3 níveis

O dataset não traz urgência, e sim 25 categorias de sintoma. O mapeamento abaixo é uma
**decisão de projeto** (sem validação clínica), definida em `src/triage/labels.py`:

| Nível | Critério | Categorias |
|---|---|---|
| **urgente** | risco à vida ou piora rápida | Heart hurts, Hard to breath, Internal pain, Open wound, Infected wound |
| **atenção** | merece consulta logo, mas não é emergência | Feeling dizzy, Blurry vision, Body feels weak, Head ache, Stomach ache, Cough, Feeling cold, Ear ache, Injury from sports, Emotional pain |
| **normal** | queixa leve ou crônica | Back pain, Neck pain, Shoulder pain, Knee pain, Joint pain, Muscle pain, Foot ache, Skin issue, Acne, Hair falling out |

Distribuição: normal 42%, atenção 39%, urgente 19%.

## 3. Modelo

### Preparação dos dados (`src/triage/data.py`)
- **Limpeza:** 2 linhas tinham categoria diferente das outras cópias da mesma frase e foram removidas (vale a maioria).
- **Separação treino/teste por frase:** como cada frase se repete ~10 vezes, um sorteio comum colocaria cópias da mesma frase no treino e no teste, e o modelo seria avaliado em frases que já "decorou". Usamos `StratifiedGroupKFold`, que mantém todas as cópias de uma frase do mesmo lado **e** a proporção dos 3 níveis igual nos dois lados (80% treino / 20% teste).

### Pipeline (`src/triage/model.py`)
- **TF-IDF de palavras** (1 e 2 palavras) **+ TF-IDF de pedaços de letras** (3 a 5 letras), unidos com `FeatureUnion`. Os pedaços de letras ajudam o modelo a reconhecer variações como *breath / breathing*.
- **Random Forest** com `class_weight="balanced"`, porque o nível urgente tem metade das linhas dos outros.
- Todos os hiperparâmetros e a semente (`SEED = 42`) ficam em `src/triage/config.py`, o que torna o resultado reprodutível.

### Regra de decisão: "na dúvida, urgente" (`src/triage/predict.py`)
Errar um caso urgente é o pior erro. Por isso, se a chance de urgente for **≥ 25%**
(`URGENT_THRESHOLD`), a resposta é urgente, mesmo que outro nível tenha chance maior. O custo
são mais alarmes falsos, que é a troca certa numa triagem.

### Resultados (conjunto de teste, 1.334 frases nunca vistas no treino)

| Versão | Acurácia | F1 macro | Recall urgente | Urgentes classificados como normal |
|---|---|---|---|---|
| TF-IDF palavras + RF | 0.813 | 0.790 | 0.600 | 63 |
| + pedaços de letras | 0.846 | 0.836 | 0.725 | 28 |
| **+ limiar de 25% (final)** | **0.826** | **0.812** | **0.761** | **19** |

Matriz de confusão do modelo final (linhas = verdadeiro, colunas = previsto):

| | normal | atenção | urgente |
|---|---|---|---|
| **normal** | 479 | 55 | 27 |
| **atenção** | 46 | 429 | 43 |
| **urgente** | 19 | 42 | 194 |

Os 19 urgentes que ainda viram "normal" são, na verdade, **só 2 frases** repetidas:
- *"There is an injured person"*: o rótulo é ambíguo, porque a frase não indica gravidade.
- *"...I feel crushed under a huge weight and no air passing to my lungs..."*: linguagem figurada, sem as palavras comuns do treino.

Paramos de ajustar o modelo nesse ponto de propósito: continuar ajustando olhando o conjunto
de teste seria "estudar com a prova na mão".

## 4. API

| Método | Rota | O que faz |
|---|---|---|
| `GET` | `/health` | 200 se o modelo está carregado; 503 se não está |
| `POST` | `/predict` | recebe `{"text": "..."}` e devolve o nível |
| `GET` | `/docs` | documentação interativa (Swagger) |

Exemplo:

```json
// POST /predict
{"text": "I cut my hand and it is bleeding a lot"}

// resposta
{
  "label": "urgente",
  "confidence": 0.66,
  "probabilities": {"atencao": 0.08, "normal": 0.26, "urgente": 0.66},
  "model_version": "1.0.0"
}
```

- Entrada inválida (texto vazio ou com mais de 1.000 caracteres) → **422**, validada pelo Pydantic.
- O modelo é carregado **uma vez**, quando a API liga, e não a cada requisição.
- A resposta traz a **confiança**, para que um humano possa revisar os casos incertos.

## 5. Arquitetura em nuvem (AWS)

### Batch ou real-time?
**Real-time.** A triagem acontece com o paciente na recepção: a resposta precisa sair em
milissegundos, uma frase por vez. Um processamento em lote (batch), que roda de hora em
hora ou à noite, não serve para esse caso.

### Arquitetura proposta

```
Paciente/recepção ──► Application Load Balancer ──► ECS Fargate (API em Docker, auto scaling)
                                                          │  carrega o modelo
                                                          ▼
                    MWAA (Airflow) ── retreino ──►  S3 (modelos versionados)
                                                          │
                       CloudWatch (logs, métricas e alarmes) ◄── tudo
```

| Componente | Serviço AWS | Por quê |
|---|---|---|
| Imagem Docker | **ECR** | registro privado da imagem gerada pelo CI |
| API | **ECS Fargate** atrás de um **Application Load Balancer** | roda o container sem gerenciar servidor; o auto scaling acompanha o movimento do pronto-atendimento; o ALB usa o `/health` |
| Modelos | **S3** | guarda `model.joblib` + `metadata.json` por versão |
| Retreino | **MWAA** (Airflow gerenciado) | roda a mesma DAG do projeto sem manter um servidor de Airflow |
| Observabilidade | **CloudWatch** | logs da API, latência, taxa de erro e alarmes |
| Segredos | **Secrets Manager** | credenciais fora do código |
| Região | **sa-east-1 (São Paulo)** | dados de saúde são sensíveis (LGPD); mantê-los no Brasil reduz o risco regulatório |

### Alternativas consideradas

| Alternativa | Por que não foi a escolha principal |
|---|---|
| **AWS Lambda** | o *cold start* (carregar o Python, o sklearn e o modelo na primeira chamada) pode levar segundos, ruim para uma triagem em tempo real |
| **SageMaker Endpoint** | mais caro e mais complexo do que precisamos para um modelo sklearn leve (~3,5 MB) |
| **EC2** | exige gerenciar o servidor (atualizações, escalonamento, falhas) |
| **EventBridge + Step Functions** no lugar do MWAA | opção serverless e mais barata para o retreino; como o projeto já usa Airflow, o MWAA reaproveita a mesma DAG sem reescrever nada |

## 6. Como rodar

Pré-requisitos: Python 3.11, [Poetry](https://python-poetry.org/) e Docker.

### Localmente

```bash
poetry install                              # instala as dependências
poetry run python -m triage.train           # treina e salva em models/
poetry run uvicorn triage.api.main:app      # sobe a API em http://localhost:8000
```

### Com Docker

O modelo **não** vai dentro da imagem: ele entra por volume (`./models`), o mesmo diretório
que o treino gera. Rode o treino antes.

```bash
docker compose up --build -d     # constrói a imagem e sobe a API
docker compose ps                # deve mostrar "healthy"
docker compose down              # desliga
```

### Testes

Os testes usam um CSV pequeno (`tests/fixtures/sample.csv`), então não dependem do dataset real.

```bash
poetry run pytest --cov=triage
poetry run ruff check . && poetry run ruff format --check .
```

### Latência

Com a API no ar:

```bash
poetry run python scripts/benchmark_latency.py --label baseline
```

O script faz 20 requisições de aquecimento, mede outras 500 e salva o resultado em
`docs/results/latency_baseline.json`.

| Versão | p50 | p95 | p99 |
|---|---|---|---|
| Baseline (sklearn, Docker local) | _a medir_ | _a medir_ | _a medir_ |
| Otimizado (ONNX), Etapa 4 | _a medir_ | _a medir_ | _a medir_ |

## 7. Estrutura do projeto

```
src/triage/
├── config.py        # caminhos, semente, hiperparâmetros e limiar
├── labels.py        # 25 categorias → 3 níveis
├── data.py          # carregar, limpar e separar treino/teste
├── model.py         # pipeline TF-IDF + Random Forest
├── train.py         # treina, avalia e salva o modelo
├── predict.py       # carrega o modelo e aplica a regra de decisão
└── api/             # FastAPI: main.py (rotas) e schemas.py (Pydantic)
tests/               # pytest + fixtures/sample.csv
scripts/             # benchmark de latência
models/              # modelo treinado (fora do git)
data/raw/            # dataset (fora do git)
```

## 8. Limitações

- Os níveis de urgência vêm de uma regra criada para o projeto, **sem validação clínica**.
- São só 703 frases únicas, curtas e em inglês.
- Frases com palavras que o modelo nunca viu tendem a ser classificadas com pouca confiança. Por isso, a resposta mostra a confiança.
- O sistema é **apoio** à triagem: ordena a fila, mas não substitui a avaliação de um profissional.
