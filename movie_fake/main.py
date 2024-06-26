from agents import create_agent, agent_node, AgentState
from tools import db_search, web_search, tool_node
import functools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
from langchain.schema import HumanMessage
# .env 파일의 경로 설정 (기본적으로 현재 디렉토리에서 .env 파일을 찾음)
load_dotenv()


def router(state):
    # 상태 정보를 기반으로 다음 단계를 결정하는 라우터 함수
    print('\n\n\n---\nrouter 수행!\n')
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # 이전 에이전트가 도구를 호출함
        print('라우터: call_tool!')
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # 어느 에이전트든 작업이 끝났다고 결정함
        print('라우터: end!')
        return "end"
    print('라우터: continue!')
    return "continue"

def agents(llm_model):
    """
    llm_model: select llm model to use
    """
    llm = ChatOpenAI(model=llm_model)
    # Research agent and node
    research_agent = create_agent(
        llm,
        [db_search],
        system_message="""
        * You are a researcher to find info from DB
        * You should retrieve information from DB and provide accurate data for making answer of target question
        """,
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")


    # web searcher
    web_search_agent = create_agent(
        llm,
        [web_search],
        system_message="""
        * You are a web searcher to find external info from web.
        * You should provide accurate data for making final answer by web searching
        * Summarize only the essential information concisely but comprehensively.
        """,
    )
    web_search_node = functools.partial(agent_node, agent=web_search_agent, name="Websearcher")
    return research_node, web_search_node

def graph_setting():
    research_node, web_search_node = agents('gpt-4o')

    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", research_node)
    workflow.add_node("Websearcher", web_search_node)
    workflow.add_node("call_tool", tool_node)


    workflow.add_conditional_edges(
        "Researcher",
        router,
        {"continue": "Websearcher", "call_tool": "call_tool", "end": END},
    )
    workflow.add_conditional_edges(
        "Websearcher",
        router,
        {"continue": "Researcher", "call_tool": "call_tool", "end": END},
    )
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            "Websearcher": "Websearcher",
        },
    )
    workflow.set_entry_point("Researcher")
    graph = workflow.compile()
    return graph


def main(query):
    graph = graph_setting()
    for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="범죄도시4 감독의 다른 영화를 추천해줘"
            )
        ],
    },
    # 그래프에서 수행할 최대 단계 수
    {"recursion_limit": 200},
):
        key, = s.keys()
        print(s[key])
        if 'messages' in s[key].keys():
            print(f"{key}: {s[key]['messages'][-1].additional_kwargs}")
        if 'sender' in s[key].keys():
            print(f"{key}: {s[key]['sender']}")
        print("----")

if __name__=="__main__":
    main(input("영화 관련 추천 질의를 해보세요:"))