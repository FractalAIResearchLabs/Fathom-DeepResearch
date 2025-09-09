import re
import json
import requests
import time
from typing import List
from functools import wraps
from together import Together          # pip install together
from datetime import datetime          # only needed for retries / logging



#     return decorator
def retry(max: int = 10, sleep: int = 1, fallback=None):
    """
    Retry `max` times and, if still failing, return `fallback`
    instead of raising.  This keeps outer loops alive.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] attempt {i+1}/{max} failed: {e}")
                    if i == max - 1:                 # last try exhausted
                        print(f"[retry] giving up – returning {fallback!r}")
                        return fallback              # ← swallow the error
                    if sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

class ReCall():
    sys_prompt = """
    You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.
    In this environment you have access to a set of tools you can use to assist with the user query. 
    You may perform multiple rounds of function calls. In each round, you can call one or more functions.

    As you proceed, adhere to the following principles:

    1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

    2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

    3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.


    Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

    In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
    The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
    The results of the function calls will be given back to you after execution, \
    and you can continue to call functions until you get the final answer for the user's question. \
    Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
    i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": <function-name>, "arguments": <args-json-object>}}
    </tool_call>
    """
    


    def __init__(self, executor_url):
        self.executor_url = executor_url
        
    def init_prompt(self, func_schemas, question):
        system_prompt = f"<|im_start|>system\n{self.sys_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix


    def _strip_old_tool_responses(self, prompt: str) -> str:
        TOOL_RESPONSE_RE = re.compile(r"<tool_response>.*?</tool_response>\s*", re.DOTALL)
        """Remove every existing <tool_response> … </tool_response> block."""
        return TOOL_RESPONSE_RE.sub("", prompt)

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"
    
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\n"
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"
    
    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        # print(tool_calls)
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            # print(call_str)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=60)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)
        
        results = []
        for tool_call in tool_calls:
            result = exe_tool_call(env, tool_call)
            results.append(result)
        return results
    
    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
        
    
    @retry(max=5, sleep=1, fallback={"score": 0})     
    def run(
        self, 
        env: str, 
        func_schemas: str,
        question: str,
        tokenizer,
        model_url="http://0.0.0.0:1214",
        temperature: float = 0.0,
        max_new_tokens: int = 40960,
        ):
        curr_prompt = self.init_prompt(func_schemas, question)
        all_tool_calls = []

    
        for i in range(64):
            prompt_tokens = tokenizer(curr_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
            max_tokens_left = max_new_tokens - len(prompt_tokens) - 100
   
            response = requests.post(
                f'{model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens_left,
                        "repetition_penalty": 1.05
                    },

                }
            ).json()
            print("="*100)
            print("Thinking ....")
            print("<think>"+response['text'])
            print("="*100)

            if "error" in response.keys():
                print("resp",response)
            curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])

            tool_calls: List[str] = self.extract_tool_calls(response['text'])
            all_tool_calls += tool_calls

            if len(tool_calls) == 0:
                break
            else:
                results: List[str] = self.execute_tool_calls(env, tool_calls)
                curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt, all_tool_calls

