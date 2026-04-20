import os
import getpass
import asyncio
from typing import Annotated, Sequence, Union, Optional
import operator
import functools

import requests
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

import chainlit as cl

# ── API Keys ──────────────────────────────────────────────────────────────────
# Coloque suas chaves aqui ou use variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FINANCIAL_KEY  = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
TAVILY_KEY     = os.environ.get("TAVILY_API_KEY", "")

os.environ["OPENAI_API_KEY"]              = OPENAI_API_KEY
os.environ["FINANCIAL_DATASETS_API_KEY"]  = FINANCIAL_KEY
os.environ["TAVILY_API_KEY"]              = TAVILY_KEY

# ── Tools ─────────────────────────────────────────────────────────────────────
class GetIncomeStatementsInput(BaseModel):
    ticker: str = Field(...); period: str = Field(default="ttm"); limit: int = Field(default=10)

@tool("get_income_statements", args_schema=GetIncomeStatementsInput)
def get_income_statements(ticker: str, period: str = "ttm", limit: int = 10) -> Union[dict, str]:
    "Get income statements for a ticker."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    url = f"https://api.financialdatasets.ai/financials/income-statements?ticker={ticker}&period={period}&limit={limit}"
    try: return requests.get(url, headers={"X-API-Key": api_key}).json()
    except Exception as e: return {"error": str(e)}

class GetBalanceSheetsInput(BaseModel):
    ticker: str = Field(...); period: str = Field(default="ttm"); limit: int = Field(default=10)

@tool("get_balance_sheets", args_schema=GetBalanceSheetsInput)
def get_balance_sheets(ticker: str, period: str = "ttm", limit: int = 10) -> Union[dict, str]:
    "Get balance sheets for a ticker."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    url = f"https://api.financialdatasets.ai/financials/balance-sheets?ticker={ticker}&period={period}&limit={limit}"
    try: return requests.get(url, headers={"X-API-Key": api_key}).json()
    except Exception as e: return {"error": str(e)}

class GetCashFlowInput(BaseModel):
    ticker: str = Field(...); period: str = Field(default="ttm"); limit: int = Field(default=10)

@tool("get_cash_flow_statements", args_schema=GetCashFlowInput)
def get_cash_flow_statements(ticker: str, period: str = "ttm", limit: int = 10) -> Union[dict, str]:
    "Get cash flow statements for a ticker."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    url = f"https://api.financialdatasets.ai/financials/cash-flow-statements?ticker={ticker}&period={period}&limit={limit}"
    try: return requests.get(url, headers={"X-API-Key": api_key}).json()
    except Exception as e: return {"error": str(e)}

class GetPricesInput(BaseModel):
    ticker: str = Field(...); start_date: str = Field(...); end_date: str = Field(...)
    interval: str = Field(default="day"); interval_multiplier: int = Field(default=1); limit: int = Field(default=5000)

@tool("get_stock_prices", args_schema=GetPricesInput)
def get_stock_prices(ticker: str, start_date: str, end_date: str, interval: str = "day", interval_multiplier: int = 1, limit: int = 5000) -> Union[dict, str]:
    "Get historical stock prices."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    url = f"https://api.financialdatasets.ai/prices?ticker={ticker}&start_date={start_date}&end_date={end_date}&interval={interval}&interval_multiplier={interval_multiplier}&limit={limit}"
    try: return requests.get(url, headers={"X-API-Key": api_key}).json()
    except Exception as e: return {"error": str(e)}

class GetCurrentPriceInput(BaseModel):
    ticker: str = Field(...)

@tool("get_current_stock_price", args_schema=GetCurrentPriceInput)
def get_current_stock_price(ticker: str) -> Union[dict, str]:
    "Get current stock price."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    url = f"https://api.financialdatasets.ai/prices/snapshot?ticker={ticker}"
    try: return requests.get(url, headers={"X-API-Key": api_key}).json()
    except Exception as e: return {"error": str(e)}

class GetOptionsChainInput(BaseModel):
    ticker: str = Field(...); limit: int = Field(default=10)
    strike_price: Optional[float] = Field(default=None); option_type: Optional[str] = Field(default=None)

@tool("get_options_chain", args_schema=GetOptionsChainInput)
def get_options_chain(ticker: str, limit: int = 10, strike_price=None, option_type=None) -> Union[dict, str]:
    "Get options chain."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    params = {"ticker": ticker, "limit": limit}
    if strike_price: params["strike_price"] = strike_price
    if option_type: params["option_type"] = option_type
    try: return requests.get("https://api.financialdatasets.ai/options/chain", headers={"X-API-Key": api_key}, params=params).json()
    except Exception as e: return {"error": str(e)}

class GetInsiderTradesInput(BaseModel):
    ticker: str = Field(...); limit: int = Field(default=10)

@tool("get_insider_trades", args_schema=GetInsiderTradesInput)
def get_insider_trades(ticker: str, limit: int = 10) -> Union[dict, str]:
    "Get insider trades."
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    url = f"https://api.financialdatasets.ai/insider-transactions?ticker={ticker}&limit={limit}"
    try: return requests.get(url, headers={"X-API-Key": api_key}).json()
    except Exception as e: return {"error": str(e)}

get_news_tool = TavilySearchResults(max_results=5)

fundamental_tools = [get_income_statements, get_balance_sheets, get_cash_flow_statements]
technical_tools   = [get_stock_prices, get_current_stock_price]
sentiment_tools   = [get_options_chain, get_insider_trades, get_news_tool]

# ── Graph ─────────────────────────────────────────────────────────────────────
def _to_str(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content) if content is not None else ""

def agent_node(state, agent, name):
    result = agent.invoke(state)
    content = _to_str(result["messages"][-1].content)
    return {"messages": [HumanMessage(content=content, name=name)]}

summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Voce e um gestor de portfolio. Sintetize os relatorios dos analistas em: "
     "1. Metricas financeiras 2. Analise tecnica 3. Sentimento 4. Recomendacao. "
     "Responda em portugues do Brasil."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Forneca o resumo e recomendacao de investimento."),
])

llm = ChatOpenAI(model="gpt-4o-mini", max_retries=3)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def supervisor_agent(state):
    return {"messages": state["messages"] + [HumanMessage(content="Prossiga com a analise.", name="supervisor")]}

def final_summary_agent(state):
    result = (summary_prompt | llm).invoke(state)
    return {"messages": [HumanMessage(content=result.content, name="portfolio_manager")]}

workflow = StateGraph(AgentState)

fund_agent = create_react_agent(llm, tools=fundamental_tools)
tech_agent = create_react_agent(llm, tools=technical_tools)
sent_agent = create_react_agent(llm, tools=sentiment_tools)

workflow.add_node("fundamental_analyst", functools.partial(agent_node, agent=fund_agent, name="fundamental_analyst"))
workflow.add_node("technical_analyst",   functools.partial(agent_node, agent=tech_agent, name="technical_analyst"))
workflow.add_node("sentiment_analyst",   functools.partial(agent_node, agent=sent_agent, name="sentiment_analyst"))
workflow.add_node("supervisor",     supervisor_agent)
workflow.add_node("final_summary",  final_summary_agent)

for m in ["fundamental_analyst", "technical_analyst", "sentiment_analyst"]:
    workflow.add_edge("supervisor", m)
    workflow.add_edge(m, "final_summary")

workflow.add_edge(START, "supervisor")
workflow.add_edge("final_summary", END)

graph = workflow.compile()

# ── Chainlit ──────────────────────────────────────────────────────────────────
AGENTES = {
    "fundamental_analyst": "📊 Analista Fundamental",
    "technical_analyst":   "📈 Analista Tecnico",
    "sentiment_analyst":   "📰 Analista de Sentimento",
    "portfolio_manager":   "🧠 Portfolio Manager",
}

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "👋 Bem-vindo ao **Hedge Fund Agent Team**!\n\n"
            "Digite o ticker e sua pergunta. Exemplos:\n"
            "- `AAPL - Qual o preco atual e receita?`\n"
            "- `MSFT - Vale a pena investir?`\n"
            "- `NVDA - Como esta a saude financeira?`"
        )
    ).send()

@cl.on_message
async def main(message: cl.Message):
    text = message.content.strip()

    if "-" in text:
        parts = text.split("-", 1)
        ticker  = parts[0].strip().upper()
        pergunta = parts[1].strip()
    else:
        ticker  = text.upper()
        pergunta = "Qual o preco atual, ultimas noticias e receita"

    await cl.Message(content=f"Analisando **{ticker}**... aguarde ⏳").send()

    input_data = {"messages": [HumanMessage(content=f"{pergunta} para {ticker}")]}

    try:
        steps_done = {}
        state = graph.invoke(input_data, {"recursion_limit": 25})

        for msg in state.get("messages", []):
            nome = getattr(msg, "name", None)
            if nome not in AGENTES or nome in steps_done:
                continue
            titulo  = AGENTES[nome]
            content = _to_str(msg.content)
            steps_done[nome] = True

            if nome == "portfolio_manager":
                await cl.Message(content=f"## {titulo}\n\n{content}").send()
            else:
                async with cl.Step(name=titulo, type="tool") as step:
                    step.output = content

    except Exception as e:
        import traceback
        await cl.Message(content=f"❌ Erro:\n```\n{traceback.format_exc()}\n```").send()
