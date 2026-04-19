# 🤖 Hedge Fund Agent Team

Time de agentes de IA para análise de investimentos usando **LangGraph** com padrão supervisor.

## Como funciona

Um agente supervisor coordena 3 analistas especializados que trabalham em paralelo. Ao final, um gestor de portfólio sintetiza tudo em um relatório com recomendação de investimento.

```
START → Supervisor → [Analista Fundamental]  ─┐
                   → [Analista Técnico]       ─┼→ Portfolio Manager → END
                   → [Analista de Sentimento] ─┘
```

## Agentes

| Agente | Especialidade | Ferramentas |
|--------|--------------|-------------|
| 📊 Analista Fundamental | Balanços, receita, fluxo de caixa | Financial Datasets API |
| 📈 Analista Técnico | Preços históricos e cotação atual | Financial Datasets API |
| 📰 Analista de Sentimento | Insider trading, opções e notícias | Financial Datasets API + Tavily |
| 🧠 Portfolio Manager | Síntese e recomendação final | — |

## Stack

- **LLM:** GPT-4o Mini (OpenAI)
- **Orquestração:** LangGraph
- **Dados financeiros:** [financialdatasets.ai](https://financialdatasets.ai)
- **Busca de notícias:** [Tavily](https://tavily.com)
- **Interface:** Gradio

## Como usar

### 1. Instalar dependências

```bash
pip install langgraph langchain langchain_openai langchain_community langsmith gradio pandas rich
```

### 2. Configurar as APIs

Você vai precisar de 3 chaves de API:

| Chave | Onde obter |
|-------|-----------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `FINANCIAL_DATASETS_API_KEY` | [financialdatasets.ai](https://financialdatasets.ai) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |

### 3. Executar no Google Colab

Abra o notebook `Hedge-fund-agent-team.ipynb` no Google Colab, insira suas chaves quando solicitado e execute todas as células.

### 4. Interface Gradio

A última célula sobe uma interface web onde você digita o ticker e a pergunta:

![Interface](https://i.imgur.com/placeholder.png)

**Exemplos de uso:**
- `AAPL` — "Qual o preço atual, últimas notícias e receita?"
- `MSFT` — "Como está a saúde financeira da empresa?"
- `NVDA` — "Vale a pena investir agora?"

## Estrutura do projeto

```
hedge-fund-agent-team/
├── Hedge-fund-agent-team.ipynb   # Notebook principal
└── .gitignore
```

## Aviso

Este projeto é apenas para fins educacionais. Não constitui recomendação de investimento.
