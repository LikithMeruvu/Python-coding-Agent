from langchain.agents import initialize_agent
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import FileManagementToolkit
# from tempfile import TemporaryDirectory
import os

# Create the temporary directory for file operations
working_directory = os.getcwd()

# Create the file management toolkit
file_management_toolkit = FileManagementToolkit(root_dir=str(working_directory))
file_tools = file_management_toolkit.get_tools()

# Extract individual file operation tools
read_tool, write_tool, list_tool, copy_tool, delete_tool, move_tool, search_tool = file_tools

# Wrap the file operation tools with Tool class
read_file_tool = Tool(
    name="read_file",
    description="Read a file from the file system.",
    func=read_tool.invoke,
)

write_file_tool = Tool(
    name="write_file",
    description="Write a file to the file system.",
    func=write_tool.invoke,
)

list_directory_tool = Tool(
    name="list_directory",
    description="List files in a directory.",
    func=list_tool.invoke,
)

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
tools = [python_repl_tool, duckduckgo_search_tool, read_file_tool, write_file_tool, list_directory_tool]

# Initialize the LLM
llm = ChatGroq(temperature=0, groq_api_key="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", model_name="llama3-8b-8192")

# Initialize the agent
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# Run the agent
while True:
    user_input = input("Enter a command or search query (or 'quit' to stop): ")
    if user_input.lower() == 'quit':
        break
    result = agent.run(user_input)
    print(result)
