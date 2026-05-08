# =============================================================
# FILE 2: agentic_example.py
# =============================================================
# WHAT IS THIS?
#   An AGENT — the LLM decides which tools to call and when.
#   The LLM reads the goal, picks the right tools, and stops
#   when it thinks the analysis is complete.
#
# TASK: Same as File 1 — Analyze a DNA sequence
#   But now the LLM decides:
#     - WHICH tools to call
#     - IN WHAT ORDER to call them
#     - WHEN to stop
#
# COMPARE TO FILE 1 (Pipeline):
#   Pipeline  → Python always runs steps 1→2→3→4 in order
#   Agent     → LLM chooses: "I'll call gc_content first,
#               then start_codon — I don't need nucleotide
#               counts for this short sequence."
#
# THE KEY DIFFERENCE:
#   In File 1, Python controls the flow.
#   In File 2, the LLM controls the flow.
#
# INSTALL:
#   pip install langgraph langchain-openai python-dotenv
#
# SETUP:
#   Create a file called .env in the same folder:
#   OPENAI_API_KEY=your_key_here
#
# RUN:
#   python agentic_example.py
# =============================================================

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

load_dotenv()

# ── The DNA sequence we want to analyze ──────────────────────
DNA_SEQUENCE = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


# =============================================================
# TOOLS — the LLM can call any of these, in any order
# =============================================================

@tool
def count_nucleotides(sequence: str) -> str:
    """Count how many A, T, G, C nucleotides are in a DNA sequence.

    Args:
        sequence: DNA sequence string (e.g. 'ATGGCC')
    """
    print("   🔧 Tool called: count_nucleotides")
    counts = {
        "A": sequence.count("A"),
        "T": sequence.count("T"),
        "G": sequence.count("G"),
        "C": sequence.count("C"),
        "total_length": len(sequence),
    }
    return str(counts)


@tool
def calculate_gc_content(sequence: str) -> str:
    """Calculate the GC content percentage of a DNA sequence.
    High GC content (>60%) means thermally stable. Low (<40%) means AT-rich.

    Args:
        sequence: DNA sequence string
    """
    print("   🔧 Tool called: calculate_gc_content")
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence) * 100
    return f"GC content = {round(gc, 2)}%"


@tool
def find_start_codon(sequence: str) -> str:
    """Check if the DNA sequence has an ATG start codon (protein-coding gene).
    Also finds where it appears in the sequence.

    Args:
        sequence: DNA sequence string
    """
    print("   🔧 Tool called: find_start_codon")
    positions = []
    pos = 0
    while True:
        idx = sequence.find("ATG", pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + 1
    if positions:
        return f"ATG start codon found at positions: {positions}. Sequence is likely protein-coding."
    return "No ATG start codon found. Sequence may be non-coding."


@tool
def find_stop_codons(sequence: str) -> str:
    """Find stop codons (TAA, TAG, TGA) in the DNA sequence.
    Stop codons mark the end of a protein-coding region.

    Args:
        sequence: DNA sequence string
    """
    print("   🔧 Tool called: find_stop_codons")
    stop_codons = ["TAA", "TAG", "TGA"]
    found = {}
    for codon in stop_codons:
        positions = []
        pos = 0
        while True:
            idx = sequence.find(codon, pos)
            if idx == -1:
                break
            positions.append(idx)
            pos = idx + 1
        if positions:
            found[codon] = positions
    if found:
        return f"Stop codons found: {found}"
    return "No stop codons found in this reading frame."


# =============================================================
# AGENT STATE
# All information the agent carries between steps
# =============================================================

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # conversation history


# =============================================================
# LLM + TOOLS SETUP
# The LLM can call ANY of the 4 tools above
# =============================================================

ALL_TOOLS      = [count_nucleotides, calculate_gc_content,
                  find_start_codon, find_stop_codons]

llm = ChatOpenAI(model="pt-5.5-mini", temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)    # LLM now knows about the tools
tool_executor  = ToolNode(ALL_TOOLS)           # executes whatever tool LLM picks


# =============================================================
# GRAPH NODES
# =============================================================

def agent_node(state: AgentState) -> AgentState:
    """The LLM reasons and decides which tool to call next."""
    print("\n⚡ Agent thinking...")
    response = llm_with_tools.invoke(state["messages"])

    # Show what the LLM decided to do
    if hasattr(response, "tool_calls") and response.tool_calls:
        tools_chosen = [tc["name"] for tc in response.tool_calls]
        print(f"   LLM chose to call: {tools_chosen}")
    else:
        print("   LLM decided: analysis is complete, no more tools needed.")

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Check if the LLM wants to call more tools or is done."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"    # LLM wants to call a tool → go to tool_executor
    return "done"         # LLM is finished → end the graph


# =============================================================
# BUILD THE AGENT GRAPH
# =============================================================

def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent",  agent_node)
    graph.add_node("tools",  tool_executor)

    # Set starting point
    graph.set_entry_point("agent")

    # Conditional routing: LLM decides → tools or done
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",    # if LLM picked a tool → run it
            "done":  END,        # if LLM is done → stop
        }
    )

    # After tools run → go back to agent so LLM can decide next step
    graph.add_edge("tools", "agent")

    return graph.compile()


# =============================================================
# MAIN
# =============================================================

def main():
    from langchain_core.messages import SystemMessage, HumanMessage

    print("=" * 55)
    print("  AGENTIC EXAMPLE — LLM Decides What to Do")
    print("  Task: DNA Sequence Analysis")
    print("=" * 55)
    print(f"\nInput sequence: {DNA_SEQUENCE}")
    print(f"Length: {len(DNA_SEQUENCE)} bp")
    print("\nAvailable tools the LLM can choose from:")
    for t in ALL_TOOLS:
        print(f"  - {t.name}")

    # Initial messages sent to the LLM
    initial_messages = [
        SystemMessage(content=(
            "You are a bioinformatics assistant. "
            "Analyze the given DNA sequence using the available tools. "
            "Choose the tools that are most useful for the analysis. "
            "You do NOT need to call all tools — only call what makes sense."
        )),
        HumanMessage(content=(
            f"Please analyze this DNA sequence: {DNA_SEQUENCE}\n\n"
            "Use the tools available to you to understand this sequence. "
            "When you have enough information, provide a clear biological summary."
        )),
    ]

    # Build and run the agent
    agent = build_agent()
    result = agent.invoke({"messages": initial_messages})

    # Print the final response
    final_message = result["messages"][-1]
    print("\n" + "─" * 55)
    print("FINAL ANALYSIS (decided and written by LLM):")
    print("─" * 55)
    print(final_message.content)
    print("─" * 55)
    print("\n✅ Agent complete.")
    print("\nNOTE FOR STUDENTS:")
    print("  The LLM decided WHICH tools to call.")
    print("  The LLM decided the ORDER of tool calls.")
    print("  The LLM decided WHEN to stop.")
    print("  Python only provided the tools — not the logic.")


if __name__ == "__main__":
    main()
