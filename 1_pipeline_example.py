# =============================================================
# FILE 1: pipeline_example.py
# =============================================================
# WHAT IS THIS?
#   A PIPELINE — a fixed, predetermined sequence of steps.
#   Each step always runs in the same order, no matter what.
#   The LLM is used only at the END to summarize the results.
#   Python controls EVERYTHING — the LLM just writes text.
#
# TASK: Analyze a DNA sequence
#   Step 1 → Count nucleotides (A, T, G, C)
#   Step 2 → Calculate GC content %
#   Step 3 → Find if it is protein-coding (starts with ATG)
#   Step 4 → LLM summarizes the findings (text only, no decisions)
#
# KEY POINT FOR STUDENTS:
#   This is NOT agentic. The LLM does NOT decide what to do.
#   Python decides every step. The LLM just writes the summary.
#   Same steps run for EVERY sequence, even if unnecessary.
#
# INSTALL:
#   pip install langchain-openai python-dotenv
#
# SETUP:
#   Create a file called .env in the same folder:
#   OPENAI_API_KEY=your_key_here
#
# RUN:
#   python pipeline_example.py
# =============================================================

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ── The DNA sequence we want to analyze ──────────────────────
DNA_SEQUENCE = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


# ── STEP 1: Count nucleotides ─────────────────────────────────
def step1_count_nucleotides(sequence: str) -> dict:
    print("\n[Step 1] Counting nucleotides...")
    counts = {
        "A": sequence.count("A"),
        "T": sequence.count("T"),
        "G": sequence.count("G"),
        "C": sequence.count("C"),
        "total": len(sequence),
    }
    print(f"         A={counts['A']}  T={counts['T']}  G={counts['G']}  C={counts['C']}")
    return counts


# ── STEP 2: Calculate GC content ─────────────────────────────
def step2_gc_content(counts: dict) -> float:
    print("\n[Step 2] Calculating GC content...")
    gc = round((counts["G"] + counts["C"]) / counts["total"] * 100, 2)
    print(f"         GC content = {gc}%")
    return gc


# ── STEP 3: Check for start codon ────────────────────────────
def step3_check_start_codon(sequence: str) -> bool:
    print("\n[Step 3] Checking for start codon (ATG)...")
    has_start = sequence.startswith("ATG")
    print(f"         Has ATG start codon = {has_start}")
    return has_start


# ── STEP 4: LLM summarizes (text only — no decisions) ─────────
def step4_llm_summary(sequence: str, counts: dict, gc: float, has_start: bool) -> str:
    print("\n[Step 4] LLM writing summary (text only)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""You are a bioinformatics assistant. Summarize these DNA analysis results:
        Sequence: {sequence}
        Nucleotide counts: A={counts['A']}, T={counts['T']}, G={counts['G']}, C={counts['C']}
        Total length: {counts['total']} bp
        GC content: {gc}%
        Has ATG start codon: {has_start}
        Write 2-3 sentences summarizing what this sequence looks like biologically."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── MAIN — runs all steps in fixed order, every time ──────────
def main():
    print("=" * 55)
    print("  PIPELINE EXAMPLE — Fixed Sequential Steps")
    print("  Task: DNA Sequence Analysis")
    print("=" * 55)
    print(f"\nInput sequence: {DNA_SEQUENCE}")
    print(f"Length: {len(DNA_SEQUENCE)} bp")

    # These 4 steps ALWAYS run in this EXACT order.
    # Python decides the order — not the LLM.
    counts    = step1_count_nucleotides(DNA_SEQUENCE)
    gc        = step2_gc_content(counts)
    has_start = step3_check_start_codon(DNA_SEQUENCE)
    summary   = step4_llm_summary(DNA_SEQUENCE, counts, gc, has_start)

    print("\n" + "─" * 55)
    print("FINAL SUMMARY (written by LLM):")
    print("─" * 55)
    print(summary)
    print("─" * 55)
    print("\n✅ Pipeline complete.")
    print("\nNOTE FOR STUDENTS:")
    print("  The LLM only wrote the summary text.")
    print("  It did NOT decide which steps to run.")
    print("  Python controlled every step.")


if __name__ == "__main__":
    main()
