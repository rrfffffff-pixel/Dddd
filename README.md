# Coding Agent

Open-source, local-first multi-agent coding platform that competes with Cursor.

## Features

- **Multi-Agent Architecture** - Specialized agents for coding, testing, shell, and review
- **Local-First** - Runs entirely on your machine via Ollama
- **Autonomous Coding Loops** - Write, test, debug, iterate without human intervention
- **Tool Framework** - Extensible tool system for file ops, shell, search
- **Task Orchestration** - Automatic task decomposition and agent coordination

## Quick Start

```bash
# Install
pip install -e .

# Run a coding task
coding-agent run "add a health check endpoint to this Flask app"

# Quick code edit
coding-agent code "rename the main function to start"

# Run tests
coding-agent test

# Show config
coding-agent info
```

## Requirements

- Python 3.11+
- Ollama running locally (for local models)
- Or set `OPENAI_API_KEY` for cloud models

## Configuration

Create `coding-agent.yaml` in your project root:

```yaml
model:
  provider: ollama
  model: qwen3:1.7b
  base_url: http://localhost:11434
```

## Architecture

```
User → CLI → Orchestrator → Agent (code/test/shell/review) → Tools (file/shell/search)
                                                    ↕
                                            LLM Provider (Ollama/OpenAI)
```

## Agent Types

- **Code Agent** - Reads, writes, and edits source files
- **Test Agent** - Runs test suites and reports failures
- **Shell Agent** - Executes terminal commands
- **Review Agent** - Code review and security checks
- **Browser Agent** - Web automation and visual testing (coming soon)

## License

MIT
