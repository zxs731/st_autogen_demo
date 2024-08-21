from queue import Queue
import threading 
import streamlit as st
import time
import autogen
from typing import Any, Dict, List, Optional, Union
from streamlit_extras.let_it_rain import rain 

import random

avatars={"chat_manager":"🤦","Boss":"👮","Boss_Assistant":"🤖","Senior_Python_Engineer":"🦹","Product_Manager":"🧖","Code_Reviewer":"🧟"}
st.sidebar.write("# ♥️Microsoft Autogen♥️")
st.sidebar.write("# 智能体应用Group Chat RAG(1)")

for key in avatars:  
    st.sidebar.write("## "+avatars[key]+" : "+key)


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": {
            "gpt-35-turbo-16k-deploy",
        }
    }
)

llm_config = {
    "request_timeout": 60,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

@st.cache_resource
def iniHis():
    history=[]
    return history

his=iniHis()

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json,Agent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb

if 'input_msg' not in st.session_state:
    st.session_state.input_msg = None

        
class ST_Print_Message(Agent):
    def __init__(self, agent):
        self.agent=agent
        
    def _print_received_message(self,message: Union[Dict, str], sender: Agent):
        all_msg=[]
        def st_write(msg):
            st.write(msg)
            all_msg.append(msg)
            
        with st.chat_message(sender.name,avatar=avatars[sender.name]):
            s=sender.name+" (to "+self.agent.name+"):\n"
            st_write(s)
            if message.get("role") == "function":
                func_print = f"***** Response from calling function \"{message['name']}\" *****"
                st_write(func_print)
                st_write(message["content"])
                st_write("*" * len(func_print))
            else:
                content = message.get("content")
                if content is not None:
                    
                    if message == '':
                        content="💚Auto reply...💚"
                    st_write(content)
                
                    
                if "function_call" in message:
                    func_print = f"***** Suggested function Call: {message['function_call'].get('name', '(No function name found)')} *****"
                    st_write(func_print)
                    st_write(
                        "Arguments: \n"+
                        message["function_call"].get("arguments", "(No arguments found)")
                    )
                    st_write("*" * len(func_print))
            st_write("\n"+ "-" * 80)
            newline = "\n"
            curMsg=newline.join(all_msg)
            his.append({'role':sender.name,'content':curMsg})
    
class StreamlitGroupChatManager(autogen.GroupChatManager):
    def _print_received_message(self, message: Union[Dict, str], sender: Agent):
        ST_Print_Message(self)._print_received_message(message,sender)
        super()._print_received_message(message,sender)
        
class StreamliAssistantAgent(AssistantAgent):
    def _print_received_message(self, message: Union[Dict, str], sender: Agent):
        ST_Print_Message(self)._print_received_message(message,sender)
        super()._print_received_message(message,sender)
          
class StreamlitUserAgent(UserProxyAgent):
    def _print_received_message(self, message: Union[Dict, str], sender: Agent):
        ST_Print_Message(self)._print_received_message(message,sender)
        super()._print_received_message(message,sender)
    
    def get_human_input(self, prompt: str) -> str:
        print("get_human_input")

        time.sleep(5)
        print("AI handle input automatically...")
        user_input=None
        return user_input
    
class StreamlitRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def _print_received_message(self, message: Union[Dict, str], sender: Agent):
        ST_Print_Message(self)._print_received_message(message,sender)
        super()._print_received_message(message,sender)
            

def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()

@st.cache_resource
def initialAgents():
    print('initialAgents')
    assistant = StreamliAssistantAgent(
                name="assistant",
                llm_config={
                    "seed": 42,  # seed for caching and reproducibility
                    "config_list": config_list,  # a list of OpenAI API configurations
                    "temperature": 0,  # temperature for sampling
                },  # configuration for flaml.oai, an enhanced inference API compatible with OpenAI API
            )
        # create a UserProxyAgent instance named "user_proxy"
    user_proxy = StreamlitUserAgent(
        name="user",
        human_input_mode="ALWAYS",#"NEVER",#
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": "python:3",  # set to True or image name like "python:3" to use docker
        },
    )
    autogen.ChatCompletion.start_logging()
    termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

    boss = StreamlitUserAgent(
        name="Boss",
        is_termination_msg=termination_msg,
        human_input_mode="ALWAYS",
        system_message="The boss who ask questions and give tasks.",
        code_execution_config=False,  # we don't want to execute code in this case.
        
    )
    import chromadb
    client = chromadb.PersistentClient(path="./chromadb")
    boss_aid = StreamlitRetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "code",
            "docs_path": './README.md',#"https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "client":client,
            "collection_name": "groupchat",
            "get_or_create": True,
        },
        code_execution_config=False,  # we don't want to execute code in this case.
    )

    coder = StreamliAssistantAgent(
        name="Senior_Python_Engineer",
        is_termination_msg=termination_msg,
        system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    pm = StreamliAssistantAgent(
        name="Product_Manager",
        is_termination_msg=termination_msg,
        system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    reviewer = StreamliAssistantAgent(
        name="Code_Reviewer",
        is_termination_msg=termination_msg,
        system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()
    groupchat = autogen.GroupChat(
        agents=[boss, coder, pm, reviewer], messages=[], max_round=12
    )
    manager = StreamlitGroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return boss,boss_aid,coder,pm,reviewer,manager
    

boss,boss_aid,coder,pm,reviewer,manager= initialAgents()


PROBLEM = "How to use spark for parallel training in FLAML? Give me sample code."


def rag_chat(PROBLEM):
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss_aid, coder, pm, reviewer], messages=[], max_round=12
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        problem=PROBLEM,
        n_results=3,
    )

def norag_chat(PROBLEM):
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss, coder, pm, reviewer], messages=[], max_round=12
    )
    manager = StreamlitGroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )

def call_rag_chat():
    _reset_agents()
    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(message, n_results=3):
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message
    
    boss_aid.human_input_mode = "NEVER" # Disable human input for boss_aid since it only retrieves content.
    
    llm_config = {
        "functions": [
            {
                "name": "retrieve_content",
                "description": "retrieve content for code generation and question answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                        }
                    },
                    "required": ["message"],
                },
            },
        ],
        "config_list": config_list,
        "request_timeout": 60,
        "seed": 42,
    }

    for agent in [coder, pm, reviewer]:
        # update llm_config for assistant agents.
        agent.llm_config.update(llm_config)

    for agent in [boss, coder, pm, reviewer]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
            }
        )

    groupchat = autogen.GroupChat(
        agents=[boss, coder, pm, reviewer], messages=[], max_round=12
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )
if 'count' not in st.session_state:
    st.session_state.count = 0 

user_input = st.chat_input('Enter input')
#his=getHis()

input_user_name="chat_manager"
if st.session_state.count == 0:
    if user_input is None:
        with st.chat_message(input_user_name,avatar=avatars[input_user_name]): 
            st.write("chat_manager:")
            st.write("有何可为您效劳？")
            st.write("eg: How to use spark for parallel training in FLAML? Give me sample code.")
            st.write("eg: write a simple demo with autogen")
       

    else:    
        st.session_state.count += 1
        rag_chat(user_input)
else:
    #print(user_input is None)
    if user_input is None:  
        with st.chat_message(input_user_name,avatar=avatars[input_user_name]): 
            st.write("chat_manager:")
            st.write("请提出您的问题！")
    else:    
        print(his)
        for h in his:
            curMsg=h['content']
            with st.chat_message(h["role"],avatar=avatars[h["role"]]):
                st.write(curMsg[0])
                st.markdown(curMsg[1])  

        boss.send({'content':user_input},manager)


