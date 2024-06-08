import logging
from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

import os
os.environ["OPENAI_API_KEY"] = "Your_api"

llm = ChatGroq(temperature=0, groq_api_key="Your_api", model_name="llama3-70b-8192")


# Create the Python REPL tool
python_repl = PythonREPL()
python_repl_tool = Tool(
    name="python_repl",
    description="This tool can execute python code and shell commands (pip commands to modules installation) Use with caution",
    func=python_repl.run,
)

# Create the DuckDuckGo search tool
duckduckgo_search_tool = Tool(
    name="duckduckgo_search",
    description="A wrapper around DuckDuckGo Search.",
    func=DuckDuckGoSearchRun().run,
)

coderAgent = Agent(
    role='Senior Software engineer and developer',
    goal='Write production grade bug free code on this user prompt :- {topic}',
    verbose=True,
    memory=True,
    backstory="You are an experienced developer in big tech companies",
    max_iter=3,
    llm = llm,
    max_rpm=10,
    tools=[duckduckgo_search_tool],
    allow_delegation=False
)

DebuggerAgent = Agent(
    role='Code Debugger and bug solving agent',
    goal='You debug the code line by line and solve bugs and errors in the code by using Python_repl tool. You also have Access to search tool which can assist you for searching that bug',
    verbose=True,
    memory=True,
    backstory="You are a debugger agent with access to a python interpreter",
    tools=[duckduckgo_search_tool, python_repl_tool],
    max_iter=3,
    llm = llm,
    max_rpm=10,
    allow_delegation=True
)

coding_task = Task(
    description="Write code in this {topic}.",
    expected_output='A Bug-free and production-grade code on {topic}',
    tools=[duckduckgo_search_tool],
    agent=coderAgent,
)

debug_task = Task(
    description="You should run the python code given by the CoderAgent and Check for bugs and errors",
    expected_output='you should communicate to CoderAgent and give feedback on the code if the code got error while execution',
    tools=[duckduckgo_search_tool, python_repl_tool],
    agent=DebuggerAgent,
    output_file='temp.py'
)

Final_check = Task(
    description="You fill finalize the Code which is verified by debugger agent Which is error free no bugs",
    expected_output="You should communicate to DebuggerAgent and if the code is bug free and executed Without errors then return the code to user",
    agent=coderAgent,
    tools=[duckduckgo_search_tool]
)

crew = Crew(
    agents=[coderAgent, DebuggerAgent],
    tasks=[coding_task, debug_task, Final_check],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=25,
    share_crew=True
)

while True:
    topic = input("Enter the topic: ")
    if topic.lower() == 'quit':
        break
    result = crew.kickoff(inputs={'topic': topic})
    print(result)
