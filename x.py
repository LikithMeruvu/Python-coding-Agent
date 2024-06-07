import logging
from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Ensure correct API key and model are set
llm = ChatGroq(temperature=0, api_key="GROQ_API", model="llama3-70b-8192")

# Create the Python REPL tool
python_repl = PythonREPL()
python_repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Create the DuckDuckGo search tool
duckduckgo_search = DuckDuckGoSearchRun()
duckduckgo_search_tool = Tool(
    name="duckduckgo_search",
    description="A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.",
    func=duckduckgo_search.run,
)

coderAgent = Agent(
    role='Senior Software engineer and developer',
    goal='Write production grade bug free code on this user prompt :- {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "You are an experienced developer in big tech companies"
        "You have a record of writing bug-free python code all the time and delivering extraordinary programming logic"
        "You are extremely good at python programming "
    ),
    llm=llm,  # Optional
    max_iter=10,  # Optional
    max_rpm=10,
    tools=[duckduckgo_search_tool],
    allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
DebuggerAgent = Agent(
    role='Code Debugger and bug solving agent',
    goal='You debug the code line by line and solve bugs and errors in the code by using Python_repl tool which can execute python code and give feedback',
    verbose=True,
    memory=True,
    backstory=(
        "You are a debugger agent you have access to a python interpreter which can run python code and give feedback"
        "You also have internet searching capabilities if you are unable to solve the bug you can search on the internet to solve that bug"
    ),
    tools=[duckduckgo_search_tool, python_repl_tool],
    llm=llm, 
    max_iter=10, 
    max_rpm=10,
    allow_delegation=True
)

# Research task
coding_task = Task(
    description=(
        "Write code in this {topic}."
        "Focus on writing bug-free and production-grade code all the time"
        "You are extremely good in python programming language"
        "You should only return code"
    ),
    expected_output='A Bug-free and production-grade code on {topic}',
    tools=[duckduckgo_search_tool],
    llm=llm,
    agent=coderAgent,
)

# Writing task with language model configuration
debug_task = Task(
    description=(
        "You should run the python code given by the CoderAgent and Check for bugs and errors"
        "If you find any bugs or errors then give feedback to the coderAgent to write code again this is the bug"
        "Always delegate the work if the executed python code gives error"
    ),
    expected_output='you should communicate to CoderAgent and give feedback on the code if the code got error while execution',
    tools=[duckduckgo_search_tool, python_repl_tool],
    agent=DebuggerAgent,
    llm=llm,
    # output_file='temp.py'  # Example of output customization
)

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[coderAgent, DebuggerAgent],
    tasks=[coding_task, debug_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=25,
    share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'Write me code for maximum subarray sum in an Array using python and Dont you any helper methods to solve this just pure python'})
print(result)
