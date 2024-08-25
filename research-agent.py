import os
import warnings
import logging
import json
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

# Load configuration
with open('config.json', 'r') as config_file:
    CONFIG = json.load(config_file)

# Global variable to track progress
PROGRESS = {"completed_tasks": 0, "total_tasks": 0}

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
    return ChatOpenAI(model=CONFIG['openai_model'], temperature=CONFIG['temperature'], api_key=api_key)

class FileReaderTool:
    @staticmethod
    def read_file(file_path: str) -> str:
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"The file {file_path} does not exist.")
            return f"Error: File {file_path} not found."
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            return f"Error reading file: {str(e)}"

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
        avg_creativity = sum(p.get('creativity_score', 0) for p in recent_performances) / 3

        improvements = []
        if avg_quality < 0.7:
            improvements.append("increase overall quality")
        if avg_content < 0.3:
            improvements.append("focus on relevant content")
        if avg_creativity < 0.5:
            improvements.append("enhance creative thinking")
        
        for area in performance.get('areas_for_improvement', []):
            improvements.append(area.lower())

        if improvements:
            improvement_str = ", ".join(improvements)
            agent.goal += f" with emphasis on: {improvement_str}"
            if agent.verbose:
                logging.info(f"{agent.role.capitalize()} has updated its goal to: {agent.goal}")
        
        # Implement adaptive learning rate
        learning_rate = 0.1 * (1 - avg_quality)  # Adjust learning rate based on performance
        
        # Update agent's skills or knowledge base
        agent.update_skills(improvements, learning_rate)
        
        # Reflect on past performance and generate insights
        insights = agent.reflect_on_performance(recent_performances)
        if agent.verbose:
            logging.info(f"{agent.role.capitalize()} insights: {insights}")

def update_skills(self, improvements: List[str], learning_rate: float) -> None:
    if not hasattr(self, 'skills'):
        self.skills = {}
    for improvement in improvements:
        if improvement not in self.skills:
            self.skills[improvement] = 0
        self.skills[improvement] += learning_rate
    if self.verbose:
        logging.info(f"{self.role.capitalize()} updated skills: {self.skills}")

def reflect_on_performance(self, performances: List[Dict[str, Any]]) -> str:
    # Analyze trends and patterns in performance
    trend = "improving" if performances[-1]['quality_score'] > performances[0]['quality_score'] else "declining"
    strengths = [area for area, score in performances[-1].items() if score > 0.7]
    weaknesses = [area for area, score in performances[-1].items() if score < 0.3]
    
    insights = f"Performance trend is {trend}. Strengths: {', '.join(strengths)}. Areas for improvement: {', '.join(weaknesses)}."
    return insights

CrewAgent.update_skills = update_skills
CrewAgent.reflect_on_performance = reflect_on_performance

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

    tool_map = {
        "DuckDuckGo Search": search_tool,
        "File Reader": file_reader_tool
    }

    return [
        create_agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=[tool_map[tool] for tool in config['tools']],
            verbose=True,
            llm=llm,
            allow_delegation=True
        ) for config in CONFIG['agent_configs']
    ]

def define_tasks(agents: List[CrewAgent], topic: str) -> List[Task]:
    """Create tasks for the agents based on the specified topic."""
    return [
        Task(
            description=config['description'].format(topic=topic),
            agent=next(agent for agent in agents if agent.role == config['role']),
            expected_output=config['expected_output'].format(topic=topic)
        ) for config in CONFIG['task_configs']
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
    creativity_score = len(set(crew_output.split())) / word_count  # Unique words ratio as a simple creativity metric

    logging.info(f"Performance evaluation completed. Quality score: {quality_score}, Content score: {content_score}, Creativity score: {creativity_score}")

    return {
        "word_count": word_count,
        "quality_score": quality_score,
        "content_score": content_score,
        "creativity_score": creativity_score,
        "areas_for_improvement": areas_for_improvement
    }

def run_research_process(crew: Crew) -> Optional[str]:
    """Run the research process and return the result."""
    try:
        result = crew.kickoff()
        print("\n" + "#" * 20)
        print(result)
        if isinstance(result, str):
            return result
        elif hasattr(result, 'final_output'):
            return result.final_output
        elif hasattr(result, 'result'):
            return result.result
        else:
            logging.warning(f"Unexpected result type: {type(result)}")
            return str(result)
    except Exception as e:
        logging.error(f"Error during research process: {e}", exc_info=True)
        return None

def update_progress(completed_tasks: int, total_tasks: int):
    """Update the global progress tracker."""
    global PROGRESS
    PROGRESS["completed_tasks"] = completed_tasks
    PROGRESS["total_tasks"] = total_tasks
    progress_percentage = (completed_tasks / total_tasks) * 100
    logging.info(f"Research Progress: {progress_percentage:.2f}% ({completed_tasks}/{total_tasks} tasks completed)")

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

        # Set up progress tracking
        update_progress(0, len(tasks))

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

        # Final progress update
        update_progress(len(tasks), len(tasks))

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
