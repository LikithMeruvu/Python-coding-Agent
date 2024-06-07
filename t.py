from langchain.agents import initialize_agent,create_structured_chat_agent
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

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

# Create the list of tools
tools = [python_repl_tool, duckduckgo_search_tool]

# Initialize the LLM
llm = ChatGroq(temperature=0, groq_api_key="groq_api", model_name="llama3-70b-8192")

# Initialize the agent
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")


# Run the agent
while True:
    user_input = input("Enter a command or search query (or 'quit' to stop): ")
    if user_input.lower() == 'quit':
        break
    result = agent.run(user_input)
    print(result)
