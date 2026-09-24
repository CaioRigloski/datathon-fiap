# Datathon FIAP — Adaptive Experimentation Platform

Plataforma de experimentação adaptativa para recomendação de ofertas em canais digitais de uma instituição financeira.

O projeto utiliza um modelo de **propensity** para estimar a probabilidade base de conversão de um cliente e um algoritmo de **Thompson Sampling** para realizar a seleção adaptativa entre diferentes estratégias de oferta.

> **Importante:** o dataset utilizado, baseado no Bank Marketing, não possui histórico de múltiplas ofertas ou braços de experimentação. Portanto, as ofertas A, B e C e seus efeitos são utilizados como uma **simulação controlada para demonstrar o funcionamento de uma plataforma de experimentação adaptativa**. Os efeitos simulados não representam resultados históricos observados no dataset.

---

## 1. Objetivo

O objetivo do projeto é desenvolver uma solução capaz de:

* estimar a propensão de conversão de um cliente;
* avaliar diferentes estratégias de oferta;
* utilizar exploração e exploração adaptativa por meio de Multi-Armed Bandits;
* aprender com os resultados das interações;
* disponibilizar a recomendação por meio de uma API;
* registrar experimentos e métricas utilizando MLflow;
* propor uma arquitetura cloud para evolução do sistema.

A solução foi desenvolvida considerando o cenário de uma instituição financeira que precisa escolher, para cada cliente elegível, qual estratégia de comunicação ou oferta deve ser apresentada.

---

# 2. Dataset

Foi utilizado o dataset **Bank Marketing**, disponibilizado publicamente no Kaggle e baseado em campanhas de marketing de uma instituição financeira.

O dataset contém informações como:

* idade;
* profissão;
* estado civil;
* escolaridade;
* situação de crédito;
* financiamento habitacional;
* empréstimos;
* canal de contato;
* mês e dia da semana;
* quantidade de contatos anteriores;
* resultado de contatos anteriores;
* indicadores econômicos;
* resultado da campanha.

A variável `y` representa a resposta à campanha:

```text
yes = conversão
no  = não conversão
```

Durante a preparação dos dados, essa informação foi convertida para a variável numérica `converted`:

```text
yes → 1
no  → 0
```

---

# 3. Preparação dos dados

Foi realizada uma etapa de preparação e análise exploratória dos dados.

## Remoção de vazamento temporal

A variável `duration` foi removida do modelo.

Essa variável representa a duração da ligação e só é conhecida após a interação com o cliente. Utilizá-la para decidir qual oferta apresentar criaria vazamento de informação, pois a variável contém informação posterior ao início da interação.

Além disso, foram utilizadas técnicas de pré-processamento diferentes para variáveis categóricas e numéricas.

### Variáveis categóricas

Foi utilizado `OneHotEncoder` para transformação das variáveis categóricas.

### Variáveis numéricas

As variáveis numéricas foram mantidas em sua representação numérica.

---

# 4. Modelo de Propensity

Foi utilizado um modelo de **Regressão Logística** para estimar a probabilidade base de conversão de cada cliente.

O modelo utiliza as características disponíveis antes da interação para produzir uma estimativa:

```text
Cliente
   ↓
Características
   ↓
Pré-processamento
   ↓
Regressão Logística
   ↓
Probabilidade base de conversão
```

As métricas obtidas no conjunto de teste foram:

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |    0.8987 |
| Precision |    0.6577 |
| Recall    |    0.2112 |
| ROC-AUC   |    0.7974 |

O **ROC-AUC** foi utilizado como uma das principais métricas para avaliar a capacidade do modelo de ordenar clientes de acordo com sua propensão estimada.

---

# 5. Experimentação Adaptativa

## Estratégias

Foram definidas três estratégias simuladas:

| Braço | Estratégia                         |
| ----: | ---------------------------------- |
|     0 | Oferta A — estratégia padrão       |
|     1 | Oferta B — abordagem personalizada |
|     2 | Oferta C — abordagem alternativa   |

As estratégias foram utilizadas para criar um ambiente controlado de experimentação.

### Oferta A

Estratégia padrão, sem efeito adicional:

```text
efeito = 0%
```

### Oferta B

Abordagem personalizada:

```text
efeito base = +3%

education = tertiary → +2%
housing = no → +1%
```

### Oferta C

Abordagem alternativa:

```text
efeito base = +2%

poutcome = success → +3%
previous > 0 → +1%
```

A probabilidade final é limitada ao intervalo `[0, 1]`.

Esses efeitos são **hipóteses simuladas**, criadas exclusivamente para demonstrar o funcionamento do algoritmo adaptativo.

---

# 6. Thompson Sampling

O algoritmo utilizado para a experimentação adaptativa foi o **Thompson Sampling**.

O algoritmo mantém uma distribuição Beta para cada braço:

```text
Oferta A → Beta(alpha_A, beta_A)
Oferta B → Beta(alpha_B, beta_B)
Oferta C → Beta(alpha_C, beta_C)
```

Inicialmente:

```text
Alpha = [1, 1, 1]
Beta  = [1, 1, 1]
```

A cada interação:

* `reward = 1` atualiza o `alpha` do braço escolhido;
* `reward = 0` atualiza o `beta` do braço escolhido.

Dessa forma, o algoritmo consegue equilibrar:

* **exploração**, testando estratégias ainda pouco conhecidas;
* **exploração**, utilizando estratégias que apresentam melhores resultados observados.

---

# 7. Avaliação do experimento

Para avaliar a estabilidade do experimento, foram realizadas **30 simulações com diferentes seeds**.

Os resultados médios observados foram:

| Estratégia        | Conversão média |
| ----------------- | --------------: |
| Baseline          |          11,16% |
| Thompson Sampling |          14,33% |

A diferença média observada foi de aproximadamente:

```text
+3,17 pontos percentuais
```

A melhoria relativa média foi de:

```text
28,38%
```

Esses resultados são referentes ao **ambiente simulado** utilizado no projeto e não devem ser interpretados como evidência de que as ofertas A, B ou C possuem esses efeitos em uma campanha real.

---

# 8. Golden Set

Foi criado um conjunto de cinco clientes para demonstrar o funcionamento da recomendação.

A recomendação é calculada a partir da probabilidade base e dos efeitos simulados de cada estratégia.

Exemplos de clientes avaliados incluem:

* clientes com baixa probabilidade base;
* clientes com histórico anterior de interação;
* clientes com resultado anterior positivo;
* diferentes níveis de escolaridade;
* diferentes situações de financiamento habitacional.

A recomendação pode ser explicada a partir das características utilizadas pelo ambiente simulado.

Por exemplo, um cliente com:

```text
poutcome = success
previous > 0
```

possui um efeito adicional associado à Oferta C no ambiente simulado.

---

# 9. Serviço — FastAPI

A solução possui uma API desenvolvida com **FastAPI**.

A API recebe os dados de um cliente e retorna:

* probabilidade base de conversão;
* probabilidade estimada para cada oferta;
* melhor oferta segundo a estimativa contextual;
* oferta selecionada pelo Thompson Sampling;
* estado atual do algoritmo.

## Endpoint de recomendação

```text
POST /recommend
```

### Exemplo de payload

```json
{
  "age": 67,
  "job": "housemaid",
  "marital": "divorced",
  "education": "professional.course",
  "default": "no",
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "month": "aug",
  "day_of_week": "thu",
  "campaign": 1,
  "pdays": 6,
  "previous": 2,
  "poutcome": "success",
  "emp_var_rate": -1.7,
  "cons_price_idx": 94.027,
  "cons_conf_idx": -38.3,
  "euribor3m": 0.899,
  "nr_employed": 4991.6
}
```

A resposta possui, entre outros campos:

```json
{
  "best_estimated_offer": "Oferta C",
  "thompson_sampling_offer": "Oferta B",
  "selected_arm": 1,
  "base_probability": 0.7566,
  "offer_probabilities": {
    "Oferta A": 0.7566,
    "Oferta B": 0.7966,
    "Oferta C": 0.8166
  }
}
```

`best_estimated_offer` representa a estratégia com maior probabilidade estimada pelo modelo contextual.

`thompson_sampling_offer` representa a estratégia selecionada pelo algoritmo de Thompson Sampling, que pode realizar exploração mesmo quando outra estratégia apresenta a maior probabilidade estimada.

---

# 10. Feedback e aprendizado

A API também possui o endpoint:

```text
POST /feedback
```

Ele recebe o braço selecionado e o resultado da interação:

```json
{
  "arm": 1,
  "reward": 1
}
```

Onde:

```text
reward = 1 → cliente converteu
reward = 0 → cliente não converteu
```

O Thompson Sampling atualiza sua distribuição após cada feedback.

O estado é persistido em:

```text
src/bandit_state.json
```

Dessa forma, o aprendizado não é perdido quando a API é reiniciada.

Fluxo:

```text
Dados do cliente
       ↓
   /recommend
       ↓
Oferta selecionada
       ↓
Interação com cliente
       ↓
    /feedback
       ↓
Atualização Alpha/Beta
       ↓
Próxima recomendação
```

---

# 11. MLOps com MLflow

O projeto utiliza **MLflow** para rastrear experimentos de Machine Learning.

O experimento do modelo de propensity registra parâmetros como:

```text
model_type
max_iter
test_size
random_state
features_count
removed_feature
```

E métricas como:

```text
accuracy
precision
recall
roc_auc
```

O modelo treinado também é registrado como artefato no MLflow.

O experimento adaptativo registra:

```text
baseline_conversion
ts_conversion
improvement
```

Isso permite acompanhar os experimentos e comparar diferentes execuções.

Para iniciar a interface do MLflow:

```bash
mlflow ui
```

A interface pode ser acessada localmente em:

```text
http://127.0.0.1:5000
```

---

# 12. Arquitetura Cloud

A arquitetura proposta utiliza a AWS para disponibilizar o serviço de recomendação adaptativa em um ambiente escalável. Os canais digitais enviam as informações do cliente por meio do Amazon API Gateway, que encaminha as requisições para uma API FastAPI executada em containers no Amazon ECS com AWS Fargate. A API utiliza o modelo de propensity para estimar a probabilidade base de conversão e o Thompson Sampling para selecionar a estratégia de oferta. Os resultados das interações e os feedbacks dos usuários podem ser armazenados no Amazon S3, permitindo a criação de um histórico para análises e futuras atualizações dos modelos.

Para observabilidade e operação, o Amazon CloudWatch pode ser utilizado para centralizar logs e métricas da aplicação, enquanto o MLflow registra parâmetros, métricas e versões dos experimentos de Machine Learning. O Amazon S3 também pode servir como armazenamento dos datasets e artefatos dos modelos. Em uma evolução do projeto, os dados de feedback poderiam alimentar novos ciclos de treinamento e avaliação, permitindo monitorar mudanças no comportamento das estratégias e manter o processo de experimentação adaptativa de forma controlada.

### Arquitetura proposta

```text
                    Canais Digitais
                          │
                          ▼
                  ┌───────────────┐
                  │ API Gateway   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ ECS / Fargate │
                  │   FastAPI     │
                  └───────┬───────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       ┌──────────────┐       ┌───────────────┐
       │ Propensity   │       │   Thompson    │
       │ Model        │       │   Sampling     │
       └──────────────┘       └───────┬───────┘
                                      │
                                      ▼
                                  Oferta
                                      │
                                      ▼
                                   Feedback
                                      │
                                      ▼
                                ┌───────────┐
                                │    S3     │
                                │ Dados e   │
                                │ artefatos │
                                └───────────┘

              ┌─────────────────────────────┐
              │ CloudWatch + MLflow         │
              │ Logs, métricas e experimentos│
              └─────────────────────────────┘
```

---

# 13. Estrutura do projeto

```text
datathon/
│
├── app/
│   └── main.py
│
├── data/
│   ├── bank_marketing.csv
│   └── bank_marketing_clean.csv
│
├── notebooks/
│   └── eda_model.ipynb
│
├── src/
│   ├── bandit.py
│   ├── train_model.py
│   ├── mlflow_experiment.py
│   ├── propensity_model.pkl
│   └── bandit_state.json
│
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

# 14. Como executar

## Clonar o projeto

```bash
git clone https://github.com/CaioRigloski/datathon-fiap.git
```

Entrar na pasta:

```bash
cd datathon-fiap
```

## Criar ambiente virtual

No Windows:

```bash
py -m venv .venv
```

Ativar:

```bash
source .venv/Scripts/activate
```

## Instalar Poetry

```bash
pip install poetry
```

## Executar o notebook

O notebook contém a análise exploratória, preparação dos dados, treinamento do modelo, experimentação com Thompson Sampling e avaliação dos resultados.

```bash
jupyter nbconvert --to notebook --execute notebooks/eda_model.ipynb --output eda_model_executed.ipynb
```

## Executar a API

```bash
uvicorn app.main:app --reload
```

A documentação interativa da API estará disponível em:

```text
http://127.0.0.1:8000/docs
```

A partir do Swagger é possível testar os endpoints `/recommend` e `/feedback`.

---

# 15. Etapas do desafio

| Etapa                               | Implementação                                             |
| ----------------------------------- | --------------------------------------------------------- |
| 1 — Dataset + EDA                   | Dataset Bank Marketing e notebook de análise exploratória |
| 2 — Preparação da base              | Tratamento, encoding e remoção de `duration`              |
| 3 — Baseline + algoritmo adaptativo | Logistic Regression + Thompson Sampling                   |
| 4 — Avaliação + Golden Set          | 30 simulações e conjunto de cinco clientes                |
| 5 — Serviço/interface               | API FastAPI com `/recommend` e `/feedback`                |
| 6 — Arquitetura Cloud               | Arquitetura proposta utilizando AWS                       |
| 7 — MLOps                           | MLflow para parâmetros, métricas e artefatos              |

---

# 16. Governança e limitações

O projeto utiliza dados públicos e não utiliza identificadores pessoais reais.

As variáveis utilizadas têm como objetivo demonstrar o funcionamento de um sistema de recomendação adaptativa e não devem ser utilizadas, sem avaliação adicional, para decisões reais de crédito, concessão de produtos financeiros ou outras decisões sensíveis.

A solução proposta deve manter supervisão humana, finalidade definida, minimização de dados e políticas adequadas de retenção e governança caso seja evoluída para um ambiente produtivo.

Também é importante destacar que os efeitos atribuídos às ofertas A, B e C são **simulados**. O dataset original não contém experimentos históricos com diferentes ofertas, portanto os resultados do Thompson Sampling representam a avaliação do algoritmo dentro do ambiente simulado criado para o projeto.

---

# 17. Tecnologias

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Pydantic
* Joblib
* Matplotlib
* Jupyter Notebook
* MLflow
* AWS — arquitetura proposta
* Thompson Sampling
* Multi-Armed Bandits
* Logistic Regression
