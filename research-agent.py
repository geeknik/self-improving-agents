import os
import warnings
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent as CrewAgent, Task, Process, Crew
from langchain.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress Pydantic warning
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic._internal._config')

def setup_environment() -> Optional[str]:
    """Load environment variables and initialize API key."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file")
        return None
    logging.info("Environment setup completed successfully")
    return api_key

def initialize_llm(api_key: str) -> ChatOpenAI:
    """Set up the language model with the provided API key."""
    return ChatOpenAI(model="gpt-4", temperature=0.2, api_key=api_key)

class FileReaderTool:
    @staticmethod
    def read_file(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            return file.read()

# Global dictionary to store performance logs for each agent
agent_performance_logs: Dict[str, List[Dict[str, Any]]] = {}

def create_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: List[Tool],
    verbose: bool,
    llm: ChatOpenAI,
    allow_delegation: bool
) -> CrewAgent:
    agent = CrewAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        llm=llm,
        allow_delegation=allow_delegation,
        tools=tools
    )
    agent_performance_logs[agent.role] = []
    return agent

def self_improve(agent: CrewAgent, performance: Dict[str, Any]) -> None:
    agent_performance_logs[agent.role].append(performance)
    if agent.verbose:
        logging.info(f"{agent.role.capitalize()} is self-improving based on the performance: {performance}")

    if len(agent_performance_logs[agent.role]) >= 3:
        recent_performances = agent_performance_logs[agent.role][-3:]
        avg_quality = sum(p.get('quality_score', 0) for p in recent_performances) / 3
        avg_content = sum(p.get('content_score', 0) for p in recent_performances) / 3

        improvements = []
        if avg_quality < 0.7:
            improvements.append("increase overall quality")
        if avg_content < 0.3:
            improvements.append("focus on relevant content")
        
        for area in performance.get('areas_for_improvement', []):
            improvements.append(area.lower())

        if improvements:
            improvement_str = ", ".join(improvements)
            agent.goal += f" with emphasis on: {improvement_str}"
            if agent.verbose:
                logging.info(f"{agent.role.capitalize()} has updated its goal to: {agent.goal}")

def create_agents(llm: ChatOpenAI) -> List[CrewAgent]:
    """Create and return various specialized agents."""
    search_tool = Tool(
        name="DuckDuckGo Search",
        func=DuckDuckGoSearchRun().run,
        description="Useful for searching the internet for information."
    )
    file_reader_tool = Tool(
        name="File Reader",
        func=FileReaderTool.read_file,
        description="Useful for reading content from files."
    )

    agent_configs = [
        {
            'role': 'researcher',
            'goal': 'Conduct thorough research on the specified topic using diverse and reliable sources',
            'backstory': 'An AI research assistant with expertise in academic literature review',
            'tools': [search_tool],
        },
        {
            'role': 'technical writer',
            'goal': 'Write clear, concise, and technically accurate research papers on the specified topic',
            'backstory': 'An AI writer with experience in scientific and technical documentation',
            'tools': [],
        },
        {
            'role': 'data gatherer',
            'goal': 'Collect comprehensive and relevant data from various sources including academic databases and reputable websites',
            'backstory': 'An AI designed to efficiently collect and organize data from diverse sources',
            'tools': [search_tool, file_reader_tool],
        },
        {
            'role': 'fact checker',
            'goal': 'Rigorously verify the accuracy and reliability of gathered information',
            'backstory': 'An AI dedicated to ensuring the veracity of data with a strong background in critical analysis',
            'tools': [search_tool],
        },
        {
            'role': 'information correlator',
            'goal': 'Synthesize information from multiple sources to identify patterns, trends, and potential breakthroughs',
            'backstory': 'An AI expert in data analysis and pattern recognition across diverse fields',
            'tools': [search_tool],
        },
        {
            'role': 'autonomous research assistant',
            'goal': 'Coordinate research efforts and assist in producing high-quality research papers autonomously',
            'backstory': 'An AI capable of managing complex research projects and collaborating with human researchers',
            'tools': [search_tool, file_reader_tool],
        }
    ]

    return [
        create_agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=config['tools'],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ) for config in agent_configs
    ]

def define_tasks(agents: List[CrewAgent], topic: str) -> List[Task]:
    """Create tasks for the agents based on the specified topic."""
    task_configs = [
        ('Conduct a comprehensive literature review on {topic}', 'researcher', 'A detailed report on the current state of research on {topic}'),
        ('Identify and analyze key methodologies related to {topic}', 'researcher', 'A comparative analysis of methodologies in {topic}'),
        ('Gather and organize data from recent studies on {topic}', 'data gatherer', 'A structured dataset of recent research findings on {topic}'),
        ('Verify and validate the collected data and findings on {topic}', 'fact checker', 'A comprehensive verification report with confidence scores for each piece of information'),
        ('Analyze correlations and identify emerging trends in {topic}', 'information correlator', 'An in-depth analysis report highlighting key trends and potential breakthroughs in {topic}'),
        ('Draft a technical research paper on {topic} incorporating all findings', 'technical writer', 'A well-structured draft of a technical paper on {topic} with proper citations'),
        ('Review, refine, and finalize the research paper on {topic}', 'autonomous research assistant', 'A polished, publication-ready research paper on {topic}')
    ]

    return [
        Task(
            description=desc.format(topic=topic),
            agent=next(agent for agent in agents if agent.role == role),
            expected_output=output.format(topic=topic)
        ) for desc, role, output in task_configs
    ]

def evaluate_performance(crew_output: str) -> Dict[str, Any]:
    """Evaluate the performance of the crew based on the final output."""
    word_count = len(crew_output.split())
    
    # More sophisticated quality scoring
    quality_score = min(word_count / 1000, 1)  # Base score on length
    
    # Check for key phrases that indicate quality
    quality_indicators = ['methodology', 'analysis', 'conclusion', 'references', 'findings', 'data', 'evidence']
    for indicator in quality_indicators:
        if indicator in crew_output.lower():
            quality_score += 0.1  # Boost score for each quality indicator
    
    quality_score = min(quality_score, 1)  # Cap at 1.0

    areas_for_improvement = []
    if 'methodology' not in crew_output.lower():
        areas_for_improvement.append("Expand on methodology")
    if 'recent' not in crew_output.lower():
        areas_for_improvement.append("Include more recent sources")
    if word_count < 500:
        areas_for_improvement.append("Increase content length")
    if 'limitation' not in crew_output.lower():
        areas_for_improvement.append("Discuss limitations of the research")
    if 'future' not in crew_output.lower():
        areas_for_improvement.append("Suggest future research directions")

    content_score = sum(crew_output.lower().count(word) for word in ['data', 'analysis', 'result', 'conclusion']) / word_count

    logging.info(f"Performance evaluation completed. Quality score: {quality_score}, Content score: {content_score}")

    return {
        "word_count": word_count,
        "quality_score": quality_score,
        "content_score": content_score,
        "areas_for_improvement": areas_for_improvement
    }

def run_research_process(crew: Crew) -> Any:
    """Run the research process and return the result."""
    result = crew.kickoff()
    print("\n" + "#" * 20)
    print(result)
    return result

def main():
    """Main function to orchestrate the script's execution."""
    try:
        api_key = setup_environment()
        if not api_key:
            logging.error("Failed to set up environment. Exiting.")
            return

        llm = initialize_llm(api_key)
        agents = create_agents(llm)

        topic = input("Enter the research topic: ")
        tasks = define_tasks(agents, topic)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )

        result = run_research_process(crew)
        if not result:
            logging.error("Research process failed to produce a result.")
            return

        performance = evaluate_performance(result)
        logging.info(f"Initial Performance Evaluation: {performance}")

        for agent in agents:
            self_improve(agent, performance)

        if performance['quality_score'] < 0.7:
            logging.info("Re-running the research process with improved agents...")
            result = run_research_process(crew)
            if not result:
                logging.error("Second research process failed to produce a result.")
                return
            final_performance = evaluate_performance(result)
            logging.info(f"Final Performance Evaluation: {final_performance}")
        
        print("\nFinal Research Result:")
        print(result)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
