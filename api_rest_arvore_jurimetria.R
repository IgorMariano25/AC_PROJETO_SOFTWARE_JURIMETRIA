# =============================================================
# API REST - Arvore de Decisao para Jurimetria
# Dataset: dados_jurimetria.csv
# Modelo : rpart (CART) com poda via validacao cruzada (cp otimo)
# API    : plumber
#
# Como executar:
#   1) No R/RStudio, ajuste o working directory para a pasta do projeto:
#        setwd("E:/AC_Projeto_ES_Jurimetria")
#   2) Suba a API:
#        library(plumber)
#        pr <- plumb("api_rest_arvore_jurimetria.R")
#        pr$run(host = "0.0.0.0", port = 8000)
#   3) Acesse a documentacao Swagger em:
#        http://localhost:8000/__docs__/
# =============================================================


# -------------------------------------------------------------
# 1. PACOTES
# -------------------------------------------------------------
pacotes <- c("rpart", "rpart.plot", "caret", "plumber", "jsonlite")
for (p in pacotes) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(rpart)
  library(rpart.plot)
  library(caret)
  library(plumber)
})


# -------------------------------------------------------------
# 2. CARGA E PREPARACAO DOS DADOS
# -------------------------------------------------------------
caminho_csv <- "dados_jurimetria.csv"
if (!file.exists(caminho_csv)) {
  stop(
    "Arquivo '", caminho_csv, "' nao encontrado no working directory: ",
    getwd()
  )
}

dados <- read.csv(
  caminho_csv,
  stringsAsFactors = TRUE,
  fileEncoding = "UTF-8"
)

# Renomeia colunas para nomes sem acento / espacos
names(dados) <- c(
  "area",
  "valor_acao",
  "tipo_parte_autora",
  "ente_publico_reu",
  "advogado_especializado",
  "foro",
  "resultado"
)

# Garante tipos corretos
colunas_fator <- c(
  "area", "tipo_parte_autora", "ente_publico_reu",
  "advogado_especializado", "foro", "resultado"
)
dados[colunas_fator] <- lapply(dados[colunas_fator], as.factor)
dados$valor_acao    <- as.numeric(dados$valor_acao)

# Guarda os niveis validos para validacao de entrada na API
niveis_validos <- lapply(dados[, colunas_fator], levels)


# -------------------------------------------------------------
# 3. DIVISAO TREINO / TESTE (ESTRATIFICADA PELO ALVO)
# -------------------------------------------------------------
set.seed(2025)

idx_treino <- caret::createDataPartition(
  y = dados$resultado,
  p = 0.7,
  list = FALSE
)
treino <- dados[idx_treino, ]
teste  <- dados[-idx_treino, ]


# -------------------------------------------------------------
# 4. TREINO DO MODELO + PODA AUTOMATICA (cp otimo via xerror)
# -------------------------------------------------------------
controle <- rpart.control(
  minsplit  = 10,
  minbucket = 4,
  cp        = 0.001,
  xval      = 10
)

modelo_full <- rpart(
  resultado ~ .,
  data    = treino,
  method  = "class",
  control = controle,
  parms   = list(split = "information")  # ganho de informacao em vez de Gini
)

# Escolhe o cp que minimiza o erro de validacao cruzada (xerror)
tabela_cp <- modelo_full$cptable
cp_otimo  <- tabela_cp[which.min(tabela_cp[, "xerror"]), "CP"]

modelo_arvore <- rpart::prune(modelo_full, cp = cp_otimo)


# -------------------------------------------------------------
# 5. AVALIACAO NO CONJUNTO DE TESTE
# -------------------------------------------------------------
previsoes_teste <- predict(modelo_arvore, newdata = teste, type = "class")
matriz_conf     <- caret::confusionMatrix(previsoes_teste, teste$resultado)

cat("\n========== AVALIACAO DO MODELO ==========\n")
cat("cp otimo escolhido:", round(cp_otimo, 5), "\n")
cat("Acuracia (teste) :", round(matriz_conf$overall["Accuracy"], 4), "\n")
cat("Kappa            :", round(matriz_conf$overall["Kappa"], 4), "\n\n")
print(matriz_conf$table)


# -------------------------------------------------------------
# 6. SALVA O PLOT DA ARVORE EM PNG (nao abre janela ao subir API)
# -------------------------------------------------------------
arquivo_plot <- "arvore_decisao.png"
tryCatch({
  png(arquivo_plot, width = 1400, height = 900, res = 130)
  rpart.plot(
    modelo_arvore,
    type          = 2,
    extra         = 104,
    fallen.leaves = TRUE,
    box.palette   = "RdYlGn",
    shadow.col    = "gray",
    main          = "Arvore de Decisao - Jurimetria"
  )
  dev.off()
  cat("Plot da arvore salvo em:", normalizePath(arquivo_plot), "\n\n")
}, error = function(e) {
  message("Falha ao salvar plot: ", e$message)
})


# -------------------------------------------------------------
# 7. FUNCOES AUXILIARES DE VALIDACAO
# -------------------------------------------------------------
validar_fator <- function(valor, niveis, nome_campo) {
  if (is.null(valor) || is.na(valor) || !nzchar(as.character(valor))) {
    return(paste0("Campo '", nome_campo, "' e obrigatorio."))
  }
  if (!(as.character(valor) %in% niveis)) {
    return(paste0(
      "Valor invalido para '", nome_campo, "': '", valor,
      "'. Valores aceitos: ", paste(niveis, collapse = ", "), "."
    ))
  }
  NULL
}

validar_numerico <- function(valor, nome_campo, minimo = 0) {
  if (is.null(valor) || is.na(valor) || !nzchar(as.character(valor))) {
    return(paste0("Campo '", nome_campo, "' e obrigatorio."))
  }
  v <- suppressWarnings(as.numeric(valor))
  if (is.na(v)) {
    return(paste0("Campo '", nome_campo, "' deve ser numerico."))
  }
  if (v < minimo) {
    return(paste0("Campo '", nome_campo, "' deve ser >= ", minimo, "."))
  }
  NULL
}


# =============================================================
#                       ENDPOINTS PLUMBER
# =============================================================

#* @apiTitle API de Jurimetria - Arvore de Decisao
#* @apiDescription Prediz o resultado de uma acao judicial (Procedente /
#*   Parcialmente Procedente / Improcedente) usando uma arvore de decisao
#*   CART treinada sobre o dataset dados_jurimetria.csv.
#* @apiVersion 1.0.0


#* Health-check da API
#* @get /
function() {
  list(
    status    = "ok",
    mensagem  = "API de Jurimetria no ar",
    horario   = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    endpoints = c("/", "/info", "/metricas", "/importancia", "/prever")
  )
}


#* Informacoes do modelo e niveis aceitos nos campos categoricos
#* @get /info
function() {
  list(
    algoritmo        = "CART (rpart) com poda por cp otimo",
    cp_otimo         = unname(cp_otimo),
    n_treino         = nrow(treino),
    n_teste          = nrow(teste),
    variavel_alvo    = "resultado",
    classes_alvo     = levels(dados$resultado),
    niveis_aceitos   = niveis_validos,
    range_valor_acao = list(
      min = min(dados$valor_acao),
      max = max(dados$valor_acao)
    )
  )
}


#* Metricas de desempenho no conjunto de teste
#* @get /metricas
function() {
  list(
    acuracia       = unname(matriz_conf$overall["Accuracy"]),
    kappa          = unname(matriz_conf$overall["Kappa"]),
    acuracia_ic_95 = c(
      inferior = unname(matriz_conf$overall["AccuracyLower"]),
      superior = unname(matriz_conf$overall["AccuracyUpper"])
    ),
    metricas_por_classe = as.data.frame(matriz_conf$byClass),
    matriz_confusao     = as.data.frame.matrix(matriz_conf$table)
  )
}


#* Importancia das variaveis na arvore
#* @get /importancia
function() {
  imp <- modelo_arvore$variable.importance
  if (is.null(imp)) {
    return(list(mensagem = "Modelo sem importancia calculada."))
  }
  imp_norm <- round(100 * imp / sum(imp), 2)
  data.frame(
    variavel               = names(imp_norm),
    importancia_percentual = as.numeric(imp_norm),
    row.names              = NULL
  )
}


#* Faz uma previsao com base nas caracteristicas do processo
#* @param area Area do direito (ex.: Civel, Trabalhista, Consumidor)
#* @param valor_acao Valor da acao (numerico, >= 0)
#* @param tipo_parte_autora "Pessoa Fisica" ou "Pessoa Juridica"
#* @param ente_publico_reu "Sim" ou "Nao"
#* @param advogado_especializado "Sim" ou "Nao"
#* @param foro "Capital" ou "Interior"
#* @get /prever
#* @post /prever
function(
  area = NULL,
  valor_acao = NULL,
  tipo_parte_autora = NULL,
  ente_publico_reu = NULL,
  advogado_especializado = NULL,
  foro = NULL,
  res
) {
  erros <- c(
    validar_fator(area,                   niveis_validos$area,                   "area"),
    validar_numerico(valor_acao, "valor_acao", minimo = 0),
    validar_fator(tipo_parte_autora,      niveis_validos$tipo_parte_autora,      "tipo_parte_autora"),
    validar_fator(ente_publico_reu,       niveis_validos$ente_publico_reu,       "ente_publico_reu"),
    validar_fator(advogado_especializado, niveis_validos$advogado_especializado, "advogado_especializado"),
    validar_fator(foro,                   niveis_validos$foro,                   "foro")
  )

  if (length(erros) > 0) {
    res$status <- 400
    return(list(erro = "Requisicao invalida", detalhes = erros))
  }

  novo <- data.frame(
    area = factor(area, levels = niveis_validos$area),
    valor_acao = as.numeric(valor_acao),
    tipo_parte_autora = factor(
      tipo_parte_autora,
      levels = niveis_validos$tipo_parte_autora
    ),
    ente_publico_reu = factor(
      ente_publico_reu,
      levels = niveis_validos$ente_publico_reu
    ),
    advogado_especializado = factor(
      advogado_especializado,
      levels = niveis_validos$advogado_especializado
    ),
    foro = factor(foro, levels = niveis_validos$foro)
  )

  classe_prevista <- predict(modelo_arvore, newdata = novo, type = "class")
  probabilidades  <- predict(modelo_arvore, newdata = novo, type = "prob")
  probs           <- probabilidades[1, ]
  confianca       <- max(probs)

  list(
    previsao         = as.character(classe_prevista),
    confianca        = round(unname(confianca), 4),
    probabilidades   = as.list(round(probs, 4)),
    entrada_recebida = as.list(novo[1, ]),
    horario          = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
}
