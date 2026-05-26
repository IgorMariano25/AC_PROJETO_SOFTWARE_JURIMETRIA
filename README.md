# Projeto AC — Jurimetria com Árvore de Decisão e API REST

Projeto da disciplina que aplica **Jurimetria** (estatística aplicada ao Direito) para prever o resultado de ações judiciais a partir de características do processo, usando uma **árvore de decisão (CART)** treinada em R e exposta via **API REST** com o pacote **Plumber**.

---

## 1. Objetivo

Construir um modelo capaz de prever, a partir de informações iniciais de uma ação judicial, qual será o **resultado** do processo:

- `Procedente`
- `Parcialmente Procedente`
- `Improcedente`

E disponibilizar esse modelo através de uma **API REST** para que outros sistemas (front-end, escritório de advocacia, dashboards, etc.) possam consumir as previsões.

---

## 2. Sobre o Dataset (`dados_jurimetria.csv`)

O dataset contém **195 processos judiciais simulados** e **7 colunas**. Cada linha representa uma ação ajuizada.

### 2.1 Variáveis preditoras

| Coluna | Tipo | Descrição | Valores observados |
|---|---|---|---|
| `area` | Categórica | Área do Direito em que a ação foi proposta | `Cível`, `Trabalhista`, `Consumidor` |
| `valor_acao` | Numérica | Valor da causa em reais | aproximadamente entre R$ 1.600 e R$ 99.900 |
| `tipo_parte_autora` | Categórica | Quem ajuizou a ação | `Pessoa Física`, `Pessoa Jurídica` |
| `ente_publico_reu` | Categórica | Se o réu é ente público (União, Estado, Município, autarquia) | `Sim`, `Não` |
| `advogado_especializado` | Categórica | Se a parte autora possui advogado especializado na área | `Sim`, `Não` |
| `foro` | Categórica | Localização da vara | `Capital`, `Interior` |

### 2.2 Variável alvo (resposta)

| Coluna | Tipo | Descrição |
|---|---|---|
| `resultado` | Categórica (3 classes) | Resultado do julgamento: `Procedente`, `Parcialmente Procedente` ou `Improcedente` |

### 2.3 Observações sobre os dados

- O dataset é **multiclasse** (3 classes no alvo), portanto a árvore de decisão é configurada com `method = "class"`.
- Há **dependência clara** de algumas variáveis com o resultado — por exemplo, ações na área `Cível` tendem fortemente a `Improcedente`, enquanto `Consumidor` e `Trabalhista` favorecem `Procedente` / `Parcialmente Procedente`. Isso torna o problema bem adequado para uma árvore de decisão, que é interpretável e captura bem regras desse tipo.
- Não há valores faltantes (`NA`) nas colunas.
- O arquivo está em **UTF-8** e contém acentos nas categorias (`Cível`, `Não`, `Pessoa Física`, etc.), o que é tratado no carregamento.

---

## 3. Arquitetura da solução

```
dados_jurimetria.csv
        │
        ▼
┌──────────────────────────────┐
│  Pré-processamento           │
│  - Renomeação de colunas     │
│  - Conversão para fator      │
│  - Split estratificado 70/30 │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Treino do modelo (rpart)    │
│  - Critério: ganho de info.  │
│  - 10-fold cross-validation  │
│  - Poda com cp ótimo         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Avaliação (caret)           │
│  - Acurácia, Kappa, IC95%    │
│  - Matriz de confusão        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  API REST (plumber)          │
│  GET /, /info, /metricas,    │
│  /importancia, /prever       │
└──────────────────────────────┘
```

---

## 4. O que foi feito no código (`api_rest_arvore_jurimetria.R`)

### 4.1 Carga e preparação dos dados
- Leitura do CSV via caminho **relativo** (`dados_jurimetria.csv`), para funcionar em qualquer máquina.
- Renomeação das colunas removendo acentos/espaços, facilitando o uso em fórmulas e na API.
- Conversão explícita de cada coluna categórica para `factor` e de `valor_acao` para `numeric`.
- Armazenamento dos **níveis válidos** de cada fator em `niveis_validos`, que é usado depois para validar a entrada da API.

### 4.2 Divisão treino/teste estratificada
- Uso de `caret::createDataPartition(p = 0.7)` em vez de `sample()` simples — isso garante que **a proporção das 3 classes do alvo seja mantida** tanto no treino quanto no teste, evitando avaliações enviesadas.
- `set.seed(2025)` garante reprodutibilidade.

### 4.3 Treino do modelo
- Algoritmo: **CART** via `rpart::rpart`.
- **Critério de divisão**: `split = "information"` (ganho de informação / entropia), em vez do Gini padrão.
- **Controle**: `minsplit = 10`, `minbucket = 4`, `cp = 0.001`, `xval = 10` (10-fold cross-validation interna).
- **Poda automática**: após treinar uma árvore "grande", o código identifica o `cp` que **minimiza o erro de validação cruzada** (`xerror`) na `cptable` e realiza `rpart::prune` com esse valor. Isso reduz overfitting e produz uma árvore mais generalizável.

### 4.4 Avaliação
- Previsões no conjunto de teste comparadas com o real via `caret::confusionMatrix`.
- Impressão no console de: `cp` ótimo, acurácia, kappa e matriz de confusão.

### 4.5 Visualização
- A árvore é exportada como imagem **`arvore_decisao.png`** (1400×900) usando `rpart.plot` com paleta `RdYlGn`. Isso evita abrir janela gráfica ao subir a API (importante para execução em servidor/headless).

### 4.6 Validação de entrada
- Funções utilitárias `validar_fator()` e `validar_numerico()` checam:
  - Campo presente e não vazio
  - Para categóricos: valor pertencente aos níveis aceitos
  - Para `valor_acao`: numérico e ≥ 0
- Em caso de erro, a API responde **HTTP 400** com a lista de problemas.

### 4.7 Endpoints da API (Plumber)

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Health-check — confirma que a API está no ar |
| `GET` | `/info` | Metadados do modelo: algoritmo, `cp` ótimo, tamanhos de treino/teste, classes do alvo e **níveis aceitos** em cada campo |
| `GET` | `/metricas` | Acurácia, Kappa, IC 95%, métricas por classe e matriz de confusão |
| `GET` | `/importancia` | Importância percentual das variáveis na árvore |
| `GET`/`POST` | `/prever` | Recebe as características do processo e retorna a previsão + probabilidades + confiança |

A API também expõe documentação automática no **Swagger** em `http://localhost:8000/__docs__/`.

---

## 5. Como executar

### 5.1 Pré-requisitos
- **R ≥ 4.0** instalado
- Pacotes (instalados automaticamente pelo script se ausentes):
  `rpart`, `rpart.plot`, `caret`, `plumber`, `jsonlite`

### 5.2 Subir a API

No R ou RStudio:

```r
setwd("E:/AC_Projeto_ES_Jurimetria")

library(plumber)
pr <- plumb("api_rest_arvore_jurimetria.R")
pr$run(host = "0.0.0.0", port = 8000)
```

A documentação interativa Swagger ficará em:

```
http://localhost:8000/__docs__/
```

---

## 6. Exemplos de uso da API

### 6.1 Health-check
```
GET http://localhost:8000/
```
Resposta:
```json
{
  "status": "ok",
  "mensagem": "API de Jurimetria no ar",
  "horario": "2026-05-26 10:00:00",
  "endpoints": ["/", "/info", "/metricas", "/importancia", "/prever"]
}
```

### 6.2 Previsão (GET via query string)
```
GET http://localhost:8000/prever?area=Consumidor&valor_acao=15000&tipo_parte_autora=Pessoa%20F%C3%ADsica&ente_publico_reu=N%C3%A3o&advogado_especializado=Sim&foro=Capital
```

### 6.3 Previsão (POST via curl)
```bash
curl -X POST "http://localhost:8000/prever" \
  -d "area=Consumidor" \
  -d "valor_acao=15000" \
  -d "tipo_parte_autora=Pessoa Física" \
  -d "ente_publico_reu=Não" \
  -d "advogado_especializado=Sim" \
  -d "foro=Capital"
```

Resposta esperada (formato):
```json
{
  "previsao": "Procedente",
  "confianca": 0.83,
  "probabilidades": {
    "Improcedente": 0.05,
    "Parcialmente Procedente": 0.12,
    "Procedente": 0.83
  },
  "entrada_recebida": { "...": "..." },
  "horario": "2026-05-26 10:00:00"
}
```

### 6.4 Métricas do modelo
```
GET http://localhost:8000/metricas
```

### 6.5 Importância das variáveis
```
GET http://localhost:8000/importancia
```

---

## 7. Decisões técnicas que diferenciam este projeto

- **Split estratificado** (`caret::createDataPartition`) em vez de amostragem aleatória simples.
- **Tuning de `cp` por validação cruzada interna** e poda automática, em vez de usar a árvore default.
- **Critério de divisão por ganho de informação** em vez do Gini padrão.
- **Validação de entrada** com mensagens de erro detalhadas e código HTTP correto (`400 Bad Request`).
- **Suporte a `GET` e `POST`** no mesmo endpoint `/prever`.
- **Múltiplos endpoints** (`/info`, `/metricas`, `/importancia`) que enriquecem a API além da previsão.
- **Plot salvo em PNG**, permitindo execução headless da API.
- Documentação automática via **Swagger** com `@apiTitle`, `@apiDescription` e `@apiVersion`.

---

## 8. Estrutura do projeto

```
AC_Projeto_ES_Jurimetria/
├── api_rest_arvore_jurimetria.R   # Script principal: treina o modelo e define a API
├── dados_jurimetria.csv           # Dataset de 195 processos
├── arvore_decisao.png             # (gerado) Imagem da árvore podada
└── README.md                      # Este arquivo
```
