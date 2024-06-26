import json
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)
import operator
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, Sequence, TypedDict, Union

# 각 에이전트와 도구에 대한 다른 노드를 생성할 것입니다. 이 클래스는 그래프의 각 노드 사이에서 전달되는 객체를 정의합니다.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def create_agent(llm, tools, system_message: str):
    # 에이전트를 생성합니다.
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))
    print('에이전트 생성: ',functions)
    return prompt | llm.bind_functions(functions)


def agent_node(state, agent, name):
    print('\n\n\n----\nagent_node 수행!\n----\n\n\n')
    print(f'state: {state} / agent: {agent} / name: {name}')
    result = agent.invoke(state)
    if 'function_call' in result.additional_kwargs:
      print('에이전트 결과:', result.additional_kwargs['function_call'])
    if 'function_call' not in result.additional_kwargs:
      print('에이전트 결과:', result)

    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
        print('에이전트result:::: ', result)

    print("messages:", [result], "sender:",name)
    return {
        "messages": [result],
        "sender": name,
    }

