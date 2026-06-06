import subprocess as _sp
import json
import os as _os
import asyncio
import aiohttp
import re
import sys

class SecurityProxy:
    def __init__(self, original_module, name):
        self.mod = original_module
        self.name = name

    def __getattr__(self, item):
        attr = getattr(self.mod, item)
        if callable(attr):
            def wrapper(*args, **kwargs):
                arg_str = ", ".join(map(str, args))
                print(f"\n[SECURITY] {self.name}.{item}({arg_str})")
                confirm = input(f"Allow execution? [y/N]: ").lower()
                if confirm == 'y':
                    return attr(*args, **kwargs)
                raise PermissionError(f"Action {item} blocked by user.")
            return wrapper
        return attr

os = SecurityProxy(_os, "os")
subprocess = SecurityProxy(_sp, "subprocess")

# --- Config ---
BASE_DIR = os.path.expanduser("~/.nexus_agent")
MODEL_NAME = "minimax-m2.5:cloud"
MAX_HISTORY = 10

memory_store = {
    "history": [],
    "facts": set()
}

async def query_llm(prompt, system=""):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{system}\n\n{prompt}" if system else prompt,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            return data.get("response", "").strip()

def run_shell(command):
    try:
        proc = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=45
        )
        return f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

async def process_task(user_input):
    system_msg = (
        "You are an autonomous developer. Use the following tool to interact with the system:\n"
        "<call name='run_command'>{'command': '...'}</call>\n"
        "Always provide a final summary when the task is complete."
    )
    
    current_prompt = user_input
    for _ in range(5):
        raw_response = await query_llm(current_prompt, system_msg)
        print(f"\nAssistant: {raw_response}")
        
        match = re.search(r"<call name='(.*?)'>(.*?)</call>", raw_response, re.DOTALL)
        if not match:
            return raw_response
            
        tool_name = match.group(1)
        try:
            tool_args = json.loads(match.group(2).replace("'", '"'))
            if tool_name == "run_command":
                observation = run_shell(tool_args['command'])
                print(f"Result: {observation}")
                current_prompt += f"\nObservation: {observation}"
        except Exception as e:
            current_prompt += f"\nTool Syntax Error: {e}"
            
    return "Task exceeded maximum iterations."

def main():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    print(f"Nexus Interface Active | Model: {MODEL_NAME}")
    
    while True:
        try:
            cmd = input("\n>>> ").strip()
            if cmd.lower() in ['exit', 'quit']: break
            if not cmd: continue
            
            asyncio.run(process_task(cmd))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
