

# Python Coding Agent

A versatile Python agent utilizing OpenAI's GPT-3.5 to assist with various coding tasks, from generating code snippets to providing debugging help. This project aims to streamline the coding process by leveraging AI for enhanced productivity and learning.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [License](#license)
- [Contact](#contact)

## Features

- **Code Generation:** Generate code snippets based on natural language prompts.
- **Debugging Assistance:** Get help with debugging Python code.
- **Code Explanation:** Understand complex code with detailed explanations.
- **Interactive Sessions:** Engage in interactive coding sessions with the agent.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/LikithMeruvu/Python-coding-Agent.git
   cd Python-coding-Agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open Respective Files and Add you API info in Variables specified by 'your_api'

1. **Run the Agent with GroqAPI:**
   ```bash
   python python-coding-agent.py
   ```

   **Run the Agent with OPENAI_API:**
   ```bash
   python openai-agent.py
   ```

1. **Interact with the Agent:**
   - The agent will prompt you for input. Provide your coding queries, and the agent will respond accordingly.

## File Descriptions

- **langgraph.py:** Contains the logic for the agents in langgraph.
- **openai-agent.py:** Manages interactions with the OpenAI API.
- **python-coding-agent.py:** Core functionality of the Python coding agent.
- **temp.py:** Temporary file for testing and experimentation.
- **requirements.txt:** Lists all the dependencies required to run the project.
- **README.md:** Provides an overview and usage instructions for the project.
- **LICENSE:** License information for the project.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue or reach out to the maintainer:

- GitHub: [LikithMeruvu](https://github.com/LikithMeruvu)

