import openai
import json
from googlesearch import search  #upm package(googlesearch-python)

__all__ = ['set_client', 'tools_table', 'get_reply', 'chat', 
           'set_backtrace', 'empty_history']

_client = openai
def set_client(client):
    global _client
    _client = client
    
def google_res(user_msg, num_results=5, verbose=False):
    content = "以下為已發生的事實：\n"                # 強調資料可信度
    for res in search(user_msg, advanced=True,    # 一一串接搜尋結果
                      num_results=num_results,
                      lang='zh-TW'):
        content += f"標題：{res.title}\n" \
                    f"摘要：{res.description}\n\n"
    content += "請依照上述事實回答以下問題：\n"        # 下達明確指令
    if verbose:
        print('------------')
        print(content)
        print('------------')
    return content

tools_table = [             # 可用工具表
    {                       # 每個元素代表一個工具
        "chain": True,      # 工具執行結果是否要再傳回給 API
        "func": google_res, # 工具對應的函式
        "spec": {           # function calling 需要的工具規格
            "type": "function",
            "function": {
                "name": "google_res",
                "description": "取得 Google 搜尋結果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_msg": {
                            "type": "string",
                            "description": "要搜尋的關鍵字",
                        }
                    },
                    "required": ["user_msg"],
                },
            }
        }
    }
]

def call_tools(tool_calls, tools_table):
    res = ''
    msg = []
    for tool_call in tool_calls:
        func = tool_call.function
        func_name = func.name
        args = json.loads(func.arguments)
        for f in tools_table:  # 找出包含此函式的項目
            if func_name == f['spec']['function']['name']:
                print(f"嘗試叫用：{func_name}(**{args})")
                val = f['func'](**args)
                if f['chain']: # 要將結果送回模型
                    msg.append({
                        'tool_call_id': tool_call.id,
                        'role': 'tool',
                        'name': 'func_name',
                        'content': val
                    })
                else:
                    res += str(val)
                break
    return msg, res

def _get_tool_calls(messages, stream=False, tools_table=None,
                  **kwargs):
    model = 'gpt-3.5-turbo-1106' # 設定模型
    if 'model' in kwargs: model = kwargs['model']

    tools = {}
    if tools_table: # 加入工具表
        tools = {'tools':[tool['spec'] for tool in tools_table]}

    response = _client.chat.completions.create(
        model = model,
        messages = messages,
        stream = stream,
        **tools
    )

    if not stream: # 非串流模式
        msg = response.choices[0].message
        if msg.content == None: # function calling 的回覆
            return msg.tool_calls, None # 取出叫用資訊
        return None, response # 一般回覆

    tool_calls = [] # 要呼叫的函式清單
    prev = None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content != None: # 一般回覆 (非 function calling)
            return None, response # 直接返回結果
        if delta.tool_calls:      # 不是頭/尾的 chunk
            curr = delta.tool_calls[0]
            if curr.function.name:       # 單一 call 開始
                prev = curr              # 取得工具名稱
                tool_calls.append(curr)  # 加入串列
            else: # 串接引數內容
                prev.function.arguments += curr.function.arguments
    return tool_calls, None

def get_reply(messages, stream=False, tools_table=None, **kwargs):
    try:
        tool_calls, response = _get_tool_calls(
            messages, stream, tools_table, **kwargs)
        if tool_calls:
            tool_messages, res = call_tools(tool_calls, tools_table)
            tool_calls_messeges = []
            for tool_call in tool_calls:
                tool_calls_messeges.append(tool_call.model_dump())
            if tool_messages:  # 如果需要將函式執行結果送回給 AI 再回覆
                messages += [ # 必須傳回原本 function_calling 的內容
                    {
                        "role": "assistant", "content": None,
                        "tool_calls": tool_calls_messeges
                    }]
                messages += tool_messages
                # pprint(messages)
                yield from get_reply(messages, stream,
                                       tools_table, **kwargs)
            else:      # chain 為 False, 以函式叫用結果當成模型生成內容
                yield res
        elif stream:   # 不需叫用函式但使用串流模式
            for chunk in response:
                yield chunk.choices[0].delta.content or ''
        else:          # 不需叫用函式也沒有使用串流模式
            yield response.choices[0].message.content
    except openai.APIError as err:
        reply = f"發生錯誤\n{err.message}"
        print(reply)
        yield reply


_hist = []       # 歷史對話紀錄
_backtrace = 2   # 記錄幾組對話

def chat(sys_msg, user_msg, stream=False, tools_table=tools_table, **kwargs):
    global _hist

    replies = get_reply(    # 使用函式功能版的函式
        _hist                  # 先提供歷史紀錄
        + [{"role": "user", "content": user_msg}]
        + [{"role": "system", "content": sys_msg}],
        stream, tools_table, **kwargs)
    reply_full = ''
    for reply in replies:
        reply_full += reply
        yield reply

    _hist += [{"role":"user", "content":user_msg},
             {"role":"assistant", "content":reply_full}]
    _hist = _hist[-2 * _backtrace:] # 留下最新的對話

def empty_history():
    _hist.clear()

def set_backtrace(backtrace=0):
    global _backtrace
    if backtrace > 0:
        _backtrace = backtrace
    return _backtrace

# 測試用的主程式
if __name__ == '__main__':
    print('測試預設使用 openai 的方式------')
    sys_msg = input('請輸入系統訊息：')
    if not sys_msg: sys_msg = '繁體中文小助理'
    while True:
        user_msg = input('請輸入使用者訊息：')
        if not user_msg: break
        for reply in chat(sys_msg, user_msg, stream=True):
            print(reply, end='')
        print()
        
    print('測試使用自訂的 openai client------')
    client = openai.OpenAI()
    set_client(client)
    while True:
        user_msg = input('請輸入使用者訊息：')
        if not user_msg: break
        for reply in chat(sys_msg, user_msg, stream=True):
            print(reply, end='')
        print()
