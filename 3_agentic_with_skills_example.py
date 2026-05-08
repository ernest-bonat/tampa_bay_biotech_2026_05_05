# =============================================================
# FILE 3: agentic_with_skills_example.py
# =============================================================
# WHAT IS THIS?
#   An AGENT WITH SKILLS — same as File 2, but now the tools
#   are organized into SKILL GROUPS. Each skill is a focused
#   set of related tools with a clear purpose.
#
#   A SKILL is just a named group of tools + a system prompt
#   that tells the LLM how to use them correctly.
#
# TASK: Same as Files 1 and 2 — Analyze a DNA sequence
#   But now skills are organized as:
#     Skill 1 → COMPOSITION_SKILL  (nucleotides, GC content)
#     Skill 2 → STRUCTURE_SKILL    (start codons, stop codons)
#     Skill 3 → INTERPRETATION_SKILL (classify the sequence)
#
# WHY USE SKILLS?
#   - Easier to maintain: each skill does one thing well
#   - Safer: LLM gets focused instructions per skill area
#   - Reusable: drop a skill into any other agent project
#   - Readable: new students immediately understand the purpose
#
# COMPARE TO FILE 2 (Agent without skills):
#   File 2 → 4 tools dumped together, one generic system prompt
#   File 3 → tools organized into 3 named skills, each with
#             its own description and usage instructions
#
# INSTALL:
#   pip install langgraph langchain-openai python-dotenv
#
# SETUP:
#   Create a file called .env in the same folder:
#   OPENAI_API_KEY=your_key_here
#
# RUN:
#   python agentic_with_skills_example.py
# =============================================================

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from dataclasses import dataclass
import operator

load_dotenv()

# ── The DNA sequence we want to analyze ──────────────────────
DNA_SEQUENCE = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


# =============================================================
# SKILL DEFINITION
# A Skill groups related tools together with instructions.
# Think of it like a specialist: "call the composition
# specialist when you need nucleotide or GC analysis."
# =============================================================

@dataclass
class Skill:
    """A named group of related tools with a description."""
    name:        str
    description: str       # tells the LLM when to use this skill
    tools:       list      # the actual tool functions


# =============================================================
# SKILL 1: COMPOSITION SKILL
# Purpose: analyze the basic chemical composition of the sequence
# =============================================================

@tool
def count_nucleotides(sequence: str) -> str:
    """Count how many A, T, G, C nucleotides are in a DNA sequence.

    Args:
        sequence: DNA sequence string (e.g. 'ATGGCC')
    """
    print("   🔧 [COMPOSITION_SKILL] count_nucleotides called")
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
    High GC (>60%) = thermally stable. Low GC (<40%) = AT-rich region.

    Args:
        sequence: DNA sequence string
    """
    print("   🔧 [COMPOSITION_SKILL] calculate_gc_content called")
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence) * 100
    category = (
        "High GC (thermally stable)" if gc > 60 else
        "Low GC (AT-rich region)"    if gc < 40 else
        "Moderate GC content"
    )
    return f"GC content = {round(gc, 2)}%  →  {category}"


COMPOSITION_SKILL = Skill(
    name        = "COMPOSITION_SKILL",
    description = "Use for nucleotide counting and GC content analysis.",
    tools       = [count_nucleotides, calculate_gc_content],
)


# =============================================================
# SKILL 2: STRUCTURE SKILL
# Purpose: find functional elements like codons
# =============================================================

@tool
def find_start_codon(sequence: str) -> str:
    """Check if the DNA sequence has ATG start codons (marks protein-coding start).
    Returns positions of all ATG codons found.

    Args:
        sequence: DNA sequence string
    """
    print("   🔧 [STRUCTURE_SKILL] find_start_codon called")
    positions, pos = [], 0
    while True:
        idx = sequence.find("ATG", pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + 1
    if positions:
        return f"ATG found at positions {positions} → likely protein-coding."
    return "No ATG start codon found → likely non-coding."


@tool
def find_stop_codons(sequence: str) -> str:
    """Find stop codons (TAA, TAG, TGA) that mark the end of a coding region.

    Args:
        sequence: DNA sequence string
    """
    print("   🔧 [STRUCTURE_SKILL] find_stop_codons called")
    found = {}
    for codon in ["TAA", "TAG", "TGA"]:
        positions, pos = [], 0
        while True:
            idx = sequence.find(codon, pos)
            if idx == -1:
                break
            positions.append(idx)
            pos = idx + 1
        if positions:
            found[codon] = positions
    if found:
        return f"Stop codons found: {found} → open reading frame may be present."
    return "No stop codons found."


STRUCTURE_SKILL = Skill(
    name        = "STRUCTURE_SKILL",
    description = "Use for finding start/stop codons and reading frame analysis.",
    tools       = [find_start_codon, find_stop_codons],
)


# =============================================================
# SKILL 3: INTERPRETATION SKILL
# Purpose: classify what kind of sequence this is
# =============================================================

@tool
def classify_sequence(
    gc_content:   float,
    has_start:    bool,
    has_stop:     bool,
    length:       int,
) -> str:
    """Classify the biological type of a DNA sequence based on analysis results.

    Args:
        gc_content: GC percentage (0-100)
        has_start:  True if ATG start codon was found
        has_stop:   True if stop codon was found
        length:     Length of the sequence in base pairs
    """
    print("   🔧 [INTERPRETATION_SKILL] classify_sequence called")
    flags = []

    if has_start and has_stop:
        flags.append("✅ Contains complete ORF (Open Reading Frame) → protein-coding gene")
    elif has_start and not has_stop:
        flags.append("⚠️  Has start codon but no stop → may be truncated gene")
    else:
        flags.append("❌ No start codon → likely non-coding or regulatory region")

    if gc_content > 60:
        flags.append("🌡️  High GC — may be CpG island (gene promoter region)")
    elif gc_content < 40:
        flags.append("📋  Low GC — AT-rich region (often intergenic or heterochromatin)")

    if length < 100:
        flags.append("📏  Short sequence — could be a primer, probe, or motif")
    elif length < 500:
        flags.append("📏  Medium length — could be an exon or short gene")
    else:
        flags.append("📏  Long sequence — could be a full gene or multi-exon region")

    return "\n".join(flags)


INTERPRETATION_SKILL = Skill(
    name        = "INTERPRETATION_SKILL",
    description = "Use after composition and structure analysis to classify the sequence type.",
    tools       = [classify_sequence],
)


# =============================================================
# REGISTER ALL SKILLS
# The agent has access to all skills.
# The LLM decides which skill tools to use.
# =============================================================

ALL_SKILLS = [COMPOSITION_SKILL, STRUCTURE_SKILL, INTERPRETATION_SKILL]
ALL_TOOLS  = [t for skill in ALL_SKILLS for t in skill.tools]

def build_skill_context() -> str:
    """Build a description of all available skills for the LLM system prompt."""
    lines = ["You have access to the following skills:\n"]
    for skill in ALL_SKILLS:
        lines.append(f"  📦 {skill.name}")
        lines.append(f"     When to use: {skill.description}")
        tool_names = [t.name for t in skill.tools]
        lines.append(f"     Tools: {', '.join(tool_names)}\n")
    return "\n".join(lines)


# =============================================================
# AGENT STATE
# =============================================================

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# =============================================================
# LLM + TOOLS SETUP
# =============================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
tool_executor  = ToolNode(ALL_TOOLS)


# =============================================================
# GRAPH NODES
# =============================================================

def agent_node(state: AgentState) -> AgentState:
    """The LLM reasons about which skill tools to call next."""
    print("\n⚡ Agent thinking (with skills)...")
    response = llm_with_tools.invoke(state["messages"])

    if hasattr(response, "tool_calls") and response.tool_calls:
        tools_chosen = [tc["name"] for tc in response.tool_calls]
        # Show which SKILL each chosen tool belongs to
        for tc in tools_chosen:
            skill_name = next(
                (s.name for s in ALL_SKILLS if tc in [t.name for t in s.tools]),
                "UNKNOWN"
            )
            print(f"   LLM chose: {tc}  [from {skill_name}]")
    else:
        print("   LLM decided: all skills used, analysis complete.")

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Check if the LLM wants to call more skill tools or is done."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "done"


# =============================================================
# BUILD THE AGENT GRAPH
# =============================================================

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent",  agent_node)
    graph.add_node("tools",  tool_executor)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "done": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("  AGENTIC + SKILLS EXAMPLE — Organized Tool Groups")
    print("  Task: DNA Sequence Analysis")
    print("=" * 60)
    print(f"\nInput sequence: {DNA_SEQUENCE}")
    print(f"Length: {len(DNA_SEQUENCE)} bp")

    print("\nSkills available to the agent:")
    for skill in ALL_SKILLS:
        tool_names = [t.name for t in skill.tools]
        print(f"  📦 {skill.name}")
        print(f"       Tools: {', '.join(tool_names)}")
        print(f"       Use:   {skill.description}")

    # System prompt tells the LLM about skills, not just tools
    skill_context = build_skill_context()

    initial_messages = [
        SystemMessage(content=(
            "You are a bioinformatics assistant with specialized skills.\n\n"
            + skill_context
            + "\nInstructions:\n"
            "1. Start with COMPOSITION_SKILL to understand the sequence makeup.\n"
            "2. Use STRUCTURE_SKILL to find coding elements.\n"
            "3. Use INTERPRETATION_SKILL to classify the sequence.\n"
            "4. Write a final biological interpretation when done.\n"
            "Only call tools that make sense for this sequence."
        )),
        HumanMessage(content=(
            f"Please analyze this DNA sequence using your skills: {DNA_SEQUENCE}\n\n"
            "Apply each relevant skill and then give a final biological classification."
        )),
    ]

    agent  = build_agent()
    result = agent.invoke({"messages": initial_messages})

    final_message = result["messages"][-1]
    print("\n" + "─" * 60)
    print("FINAL ANALYSIS (skills used, written by LLM):")
    print("─" * 60)
    print(final_message.content)
    print("─" * 60)
    print("\n✅ Agent with skills complete.")

    print("\n" + "=" * 60)
    print("SUMMARY — THREE APPROACHES COMPARED")
    print("=" * 60)
    print("""
┌───────────────────┬────────────────────────────────────────┐
│ Approach          │ Who Controls the Flow?                 │
├───────────────────┼────────────────────────────────────────┤
│ 1. Pipeline       │ Python — fixed steps, always same order│
│ 2. Agent          │ LLM — picks tools freely, no structure │
│ 3. Agent + Skills │ LLM — picks tools from organized groups│
└───────────────────┴────────────────────────────────────────┘

KEY DIFFERENCES:
  Pipeline       → Fast, predictable, easy to debug
                   Bad: inflexible, same steps every time

  Agent          → LLM picks tools freely and adaptively
                   Bad: can get confused with many tools

  Agent + Skills → LLM picks from organized tool groups
                   Good: flexible AND organized AND maintainable
                   Best for real bioinformatics projects!

WHEN TO USE EACH:
  Pipeline       → Simple, well-defined analysis (QC checks)
  Agent          → Small set of tools (< 5), simple tasks
  Agent + Skills → Complex domain with many tools (> 5)
                   Multiple analysis areas (genomics, clinical)
                   Team projects where organization matters
    """)


if __name__ == "__main__":
    main()
