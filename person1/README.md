PERSON 1 — AI AGENT / ORCHESTRATOR

WORK:
Build the main AI brain of SatQuery.

The agent receives:
- User's question
- Uploaded image information
- Available sensor/type information

It decides:
- What task the user is asking for
- Which specialist AI/tool should be used
- What order the tools should run
- How to combine their results

Examples:

"What is in this image?"
→ Vision AI

"What changed between these images?"
→ Change Detection

"Compare optical and SAR."
→ Optical + SAR analysis

TOOLS:
- Python
- Qwen3-VL
- LangGraph
- FastAPI
- Pydantic
- Hugging Face

FOLDERS:

agent/
Main agent/orchestration work.

tools/
Interfaces that allow the agent to call other AI tools.

tests/
Test whether the agent selects the correct tool for different questions.

FINAL OUTPUT:
Agent should return:
- Selected task
- Selected tools/models
- Tool execution order
- Results received
- Final combined result