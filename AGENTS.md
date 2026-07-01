# Coding Agent - Self Architecture

This is the Coding Agent's own source code. Use this reference when modifying it.

## Project Structure

```
src/coding_agent/
├── main.py              # CLI entry point (Click commands: run, code, test, info, benchmark)
├── agents/              # Specialized agent implementations
│   ├── code_agent.py    # Reads, writes, edits source code (max_iterations=15)
│   ├── test_agent.py    # Runs test suites, reports failures (max_iterations=5)
│   ├── shell_agent.py   # Executes shell commands (max_iterations=10)
│   ├── review_agent.py  # Code review & security analysis (max_iterations=5)
│   └── browser_agent.py # Web automation via Playwright
├── core/                # Core framework
│   ├── agent.py         # Base Agent class with LLM loop, caching, metrics
│   ├── config.py        # YAML-based configuration
│   ├── context.py       # SharedContext for inter-agent state
│   ├── message.py       # Message/MessageBus for agent communication
│   └── tool.py          # Tool/ToolRegistry system
├── intelligence/        # Preprocessing & optimization
│   ├── preprocessor.py  # CacheLayer, TaskClassifier, PromptCompressor, LexicalAnalyzer
│   ├── ast_index.py     # Python AST symbol indexer (CodeIndex)
│   └── semantic_search.py # TF-IDF semantic search (SemanticIndex)
├── models/              # LLM providers
│   └── provider.py      # Ollama, OpenAI, Anthropic, DeepSeek, GitHub, Mock
├── orchestrator/        # Task planning & coordination
│   ├── task_router.py   # TaskOrchestrator - decomposes tasks, routes to agents
│   └── workflow.py      # WorkflowEngine - DAG-based parallel execution
├── tools/               # Tool implementations
│   ├── file_ops.py      # read_file, write_file, edit_file, delete_file, move_file, list_files, search_files
│   ├── shell.py         # run_command (with destructive command blocking)
│   ├── search.py        # grep (with context lines, binary skip)
│   └── browser_ops.py   # Playwright browser automation
├── benchmark/           # Evaluation framework
│   └── runner.py        # BenchmarkTask, BenchmarkSuite, BenchmarkResult
├── ui/                  # TUI placeholder
└── plugins/             # Plugin placeholder
tests/                   # 167 tests (pytest)
config/default.yaml      # Default configuration
pyproject.toml           # Build & dependency config
AGENTS.md                # This file - architecture reference
```

## Key Architecture Patterns

### Agent Lifecycle
1. User invokes via CLI (`main.py`)
2. `TaskOrchestrator.decompose_task()` breaks task into subtasks via LLM
3. Each subtask is dispatched to appropriate agent
4. Agent runs LLM loop: reason -> call tools -> observe -> repeat
5. Results aggregated and printed as summary

### Base Agent Loop (`core/agent.py`)
```
for iteration in range(max_iterations):
    trim_messages()
    summarize_old_messages()
    check_token_budget()
    llm_response = provider.chat(messages, tools)
    if tool_calls:
        for each tool_call:
            result = execute_tool_with_retry(name, args)
            messages.append(result)
    else:
        return llm_response.content
```

### Tools System (`core/tool.py`)
Tools are `Tool` dataclasses with name, description, parameters, handler.
Registered in `ToolRegistry` during CLI startup.
Tools validate parameters, check path traversal safety, support caching.

### LLM Providers (`models/provider.py`)
Each provider implements `chat(messages, tools) -> LLMResponse`.
Current providers: Ollama (default), OpenAI, Anthropic, DeepSeek, GitHub, Mock.

### Configuration (`config/default.yaml`)
```yaml
model:
  provider: ollama
  model: qwen3:1.7b
  base_url: http://localhost:11434
  temperature: 0.1
```

## When modifying this codebase:
1. Read the relevant file(s) first before editing
2. Run `pytest tests/` to verify changes
3. Run `ruff check src/coding_agent/` for lint
4. New tools need to be registered in `main.py`
5. New agents need entry in `TaskOrchestrator` agent dict
6. Follow existing patterns (dataclasses, type hints, docstrings)
7. Tests mirror source structure (test_agents/, test_core/, test_tools/, etc.)
