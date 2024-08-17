import os
import warnings
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent as CrewAgent, Task, Process, Crew

# Suppress Pydantic warning
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic._internal._config')

# Configuration and Setup
def setup_environment() -> str:
    """Load environment variables and initialize API keys."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return api_key

def initialize_llm(api_key: str) -> ChatOpenAI:
    """Set up the language model with the provided API key."""
    return ChatOpenAI(model="chatgpt-4o-latest", temperature=0.78, top_p=0.95, api_key=api_key)

class FileReaderTool:
    def read_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            return file.read()

# Create a dictionary to store performance logs for each agent
agent_performance_logs: Dict[str, List[Dict[str, Any]]] = {}

def create_agent(role: str, goal: str, backstory: str, tools: List[Any], verbose: bool, llm: ChatOpenAI, allow_delegation: bool) -> CrewAgent:
    agent = CrewAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        llm=llm,
        allow_delegation=allow_delegation
    )
    agent_performance_logs[agent.role] = []
    return agent

def self_improve(agent: CrewAgent, result: Dict[str, Any]) -> None:
    agent_performance_logs[agent.role].append(result)
    if agent.verbose:
        print(f"{agent.role.capitalize()} is self-improving based on the result: {result}")

    # Implement self-improvement logic
    if len(agent_performance_logs[agent.role]) >= 3:
        recent_performances = agent_performance_logs[agent.role][-3:]
        avg_performance = sum(p.get('quality_score', 0) for p in recent_performances) / 3

        if avg_performance < 0.7:  # Assuming a score out of 1
            # Adjust the agent's behavior
            agent.goal += " with increased focus on accuracy and relevance"
            if agent.verbose:
                print(f"{agent.role.capitalize()} has updated its goal to improve performance.")

def create_agents(llm: ChatOpenAI) -> List[CrewAgent]:
    """Create and return various specialized agents."""
    search_tool = DuckDuckGoSearchRun()
    file_reader_tool = FileReaderTool()

    agents = [
        create_agent(
            role='researcher',
            goal='Conduct thorough research on the specified topic using diverse and reliable sources',
            backstory='An AI research assistant with expertise in academic literature review',
            tools=[search_tool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ),
        create_agent(
            role='technical writer',
            goal='Write clear, concise, and technically accurate research papers on the specified topic',
            backstory='An AI writer with experience in scientific and technical documentation',
            tools=[],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ),
        create_agent(
            role='data gatherer',
            goal='Collect comprehensive and relevant data from various sources including academic databases and reputable websites',
            backstory='An AI designed to efficiently collect and organize data from diverse sources',
            tools=[search_tool, file_reader_tool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ),
        create_agent(
            role='fact checker',
            goal='Rigorously verify the accuracy and reliability of gathered information',
            backstory='An AI dedicated to ensuring the veracity of data with a strong background in critical analysis',
            tools=[search_tool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ),
        create_agent(
            role='information correlator',
            goal='Synthesize information from multiple sources to identify patterns, trends, and potential breakthroughs',
            backstory='An AI expert in data analysis and pattern recognition across diverse fields',
            tools=[search_tool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ),
        create_agent(
            role='autonomous research assistant',
            goal='Coordinate research efforts and assist in producing high-quality research papers autonomously',
            backstory='An AI capable of managing complex research projects and collaborating with human researchers',
            tools=[search_tool, file_reader_tool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        )
    ]

    return agents

def define_tasks(agents: List[CrewAgent], topic: str) -> List[Task]:
    """Create tasks for the agents based on the specified topic."""
    return [
        Task(description=f'Conduct a comprehensive literature review on {topic}', agent=agents[0], expected_output=f'A detailed report on the current state of research on {topic}'),
        Task(description=f'Identify and analyze key methodologies related to {topic}', agent=agents[0], expected_output=f'A comparative analysis of methodologies in {topic}'),
        Task(description=f'Gather and organize data from recent studies on {topic}', agent=agents[2], expected_output=f'A structured dataset of recent research findings on {topic}'),
        Task(description=f'Verify and validate the collected data and findings on {topic}', agent=agents[3], expected_output=f'A comprehensive verification report with confidence scores for each piece of information'),
        Task(description=f'Analyze correlations and identify emerging trends in {topic}', agent=agents[4], expected_output=f'An in-depth analysis report highlighting key trends and potential breakthroughs in {topic}'),
        Task(description=f'Draft a technical research paper on {topic} incorporating all findings', agent=agents[1], expected_output=f'A well-structured draft of a technical paper on {topic} with proper citations'),
        Task(description=f'Review, refine, and finalize the research paper on {topic}', agent=agents[5], expected_output=f'A polished, publication-ready research paper on {topic}')
    ]

def evaluate_performance(crew_output: Any) -> Dict[str, Any]:
    """Evaluate the performance of the crew based on the final output."""
    # Check if crew_output is a string or has a specific attribute for the final result
    if isinstance(crew_output, str):
        result = crew_output
    elif hasattr(crew_output, 'final_output'):
        result = crew_output.final_output
    elif hasattr(crew_output, 'result'):
        result = crew_output.result
    else:
        result = str(crew_output)  # Convert to string as a fallback

    # Implement a more sophisticated evaluation metric
    word_count = len(result.split())
    quality_score = min(word_count / 1000, 1)  # Assume 1000 words is a good length

    return {
        "word_count": word_count,
        "quality_score": quality_score,
        "areas_for_improvement": ["Expand on methodology", "Include more recent sources"] if quality_score < 0.8 else []
    }

def main():
    """Main function to orchestrate the script's execution."""
    try:
        api_key = setup_environment()
        llm = initialize_llm(api_key)
        agents = create_agents(llm)

        # Get the research topic from the user
        topic = input("Enter the research topic: ")

        tasks = define_tasks(agents, topic)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        print("\n" + "#" * 20)
        print(result)

        # Evaluate performance
        performance = evaluate_performance(result)
        print(f"\nPerformance Evaluation: {performance}")

        # Implement self-recursive and self-improving mechanisms
        for agent in agents:
            self_improve(agent, performance)

        # Optionally, re-run the process if the performance is below a certain threshold
        if performance['quality_score'] < 0.7:
            print("\nRe-running the research process with improved agents...")
            result = crew.kickoff()
            print("\n" + "#" * 20)
            print(result)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
