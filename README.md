# Research Automation Script with Self-Improving Agents

This project leverages a combination of AI tools to automate the process of research, technical writing, data gathering, fact-checking, and information synthesis. The script orchestrates a group of specialized agents to perform these tasks autonomously with the ability to self-improve based on performance metrics.

## Features

- Automatically loads and configures necessary API keys and environment variables.
- Six distinct agents are created to handle different research tasks such as data gathering, research, writing, fact-checking, and more.
- Agents evaluate their performance and adjust their goals to improve over time.
- Tasks are defined and executed in a sequential process managed by the Crew library.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/geeknik/self-improving-agents.git
    cd self-improving-agents
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory of the project and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the script:

    ```sh
    python main.py
    ```

2. Enter the research topic when prompted.

3. The script will generate output by coordinating between multiple agents.

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Contact

If you have any questions or issues, please contact via X: [@geeknik](https://x.com/geeknik)
