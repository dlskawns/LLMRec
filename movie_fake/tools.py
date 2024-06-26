from typing import Annotated
from langchain_core.messages import FunctionMessage
from langchain_core.tools import tool
import json
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

@tool
def db_search(
    query: Annotated[str, "Query to find information from internal DB"]
    ):
  """
  Use this to search info from DB. If you need to find any information in the internal database, you can use this tool.
  """

  return '원하는 정보가 없습니다.'


@tool
def web_search(
    query: Annotated[str, "Query to find information from web browser"]
    ):
  """
  Use this to search info from web. If you need to find any information using an external web browser, you can use this tool.
  """

  return '범죄도시 감독은 허명행이며, 다른 작품으로는 황야, 신세계 등이 있다.'



def tool_node(state):
    print('\n\n\n----\ntool_node 수행!\n')
    # 그래프에서 도구를 실행하는 함수입니다.
    # 에이전트 액션을 입력받아 해당 도구를 호출하고 결과를 반환합니다.
    messages = state["messages"]
    # 계속 조건에 따라 마지막 메시지가 함수 호출을 포함하고 있음을 알 수 있습니다.
    last_message = messages[-1]
    # ToolInvocation을 함수 호출로부터 구성합니다.
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    print(f'tool input: {tool_input}')
    # 단일 인자 입력은 값으로 직접 전달할 수 있습니다.
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    print(f'tool name: {tool_name}')
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # 도구 실행자를 호출하고 응답을 받습니다.
    response = tool_executor.invoke(action)
    # 응답을 사용하여 FunctionMessage를 생성합니다.
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    print('function messages:', function_message)
    print('\ntool_node 수행완료!\n----\n\n\n')
    # 기존 리스트에 추가될 리스트를 반환합니다.
    return {"messages": [function_message]}

tools = [db_search, web_search]
tool_executor = ToolExecutor(tools)