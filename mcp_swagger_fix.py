# import pandas as pd
# import re
# from collections import defaultdict
# from typing import List, Dict, Optional

# # ── Sample Data ────────────────────────────────────────────────────────────────
# data = {
#     "id":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     "qid": ["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"],
#     "question": [
#         "What is machine learning?",
#         "How does deep learning differ from machine learning?",
#         "What is supervised learning?",
#         "Explain unsupervised learning with examples.",
#         "What is natural language processing?",
#         "How do neural networks work?",
#         "What is reinforcement learning?",
#         "How is deep learning used in computer vision?",
#         "What are transformers in NLP?",
#         "How does BERT work in natural language processing?",
#     ]
# }
# df = pd.DataFrame(data)


# # ── PageIndex ──────────────────────────────────────────────────────────────────
# class PageIndex:
#     """
#     Vectorless full-text index over a DataFrame.

#     Strategy:
#       1. Split the DataFrame into fixed-size pages.
#       2. For each page, build an inverted index: token → set of row positions.
#       3. At query time, tokenize the query, intersect (AND) or union (OR)
#          candidate row sets across pages, then return matching rows.
#     """

#     def __init__(self, df: pd.DataFrame, text_col: str, page_size: int = 3):
#         self.df        = df.reset_index(drop=True)
#         self.text_col  = text_col
#         self.page_size = page_size
#         self.pages: List[Dict] = []   # list of {page_id, row_range, index}
#         self._build()

#     # ── tokeniser ─────────────────────────────────────────────────────────────
#     @staticmethod
#     def _tokenize(text: str) -> List[str]:
#         return re.findall(r"\b\w+\b", text.lower())

#     # ── build inverted index per page ─────────────────────────────────────────
#     def _build(self):
#         n = len(self.df)
#         for page_id, start in enumerate(range(0, n, self.page_size)):
#             end      = min(start + self.page_size, n)
#             page_df  = self.df.iloc[start:end]
#             inv_idx: Dict[str, set] = defaultdict(set)

#             for local_pos, (abs_idx, row) in enumerate(page_df.iterrows()):
#                 for token in self._tokenize(str(row[self.text_col])):
#                     inv_idx[token].add(abs_idx)   # store absolute df index

#             self.pages.append({
#                 "page_id"  : page_id,
#                 "row_range": (start, end - 1),
#                 "index"    : inv_idx,
#             })

#         print(f"[PageIndex] Built {len(self.pages)} pages "
#               f"(page_size={self.page_size}) over {n} rows.")

#     # ── query ─────────────────────────────────────────────────────────────────
#     def query(self, query_str: str, mode: str = "AND") -> pd.DataFrame:
#         """
#         Parameters
#         ----------
#         query_str : str   – free-text query
#         mode      : str   – "AND" (all terms must match) | "OR" (any term matches)

#         Returns
#         -------
#         pd.DataFrame of matching rows.
#         """
#         tokens = self._tokenize(query_str)
#         if not tokens:
#             return pd.DataFrame(columns=self.df.columns)

#         print(f"\n[Query] '{query_str}'  mode={mode}  tokens={tokens}")

#         candidate_rows: Optional[set] = None

#         for page in self.pages:
#             inv_idx = page["index"]

#             # collect matching row-sets for each token in this page
#             token_sets = [inv_idx.get(tok, set()) for tok in tokens]

#             if mode == "AND":
#                 page_matches = token_sets[0].copy()
#                 for s in token_sets[1:]:
#                     page_matches &= s
#             else:  # OR
#                 page_matches = set()
#                 for s in token_sets:
#                     page_matches |= s

#             if candidate_rows is None:
#                 candidate_rows = page_matches
#             else:
#                 candidate_rows |= page_matches   # union across pages

#         if not candidate_rows:
#             print("[Query] No matches found.")
#             return pd.DataFrame(columns=self.df.columns)

#         result = self.df.loc[sorted(candidate_rows)]
#         print(f"[Query] {len(result)} row(s) found.")
#         return result


# # ── Demo ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     idx = PageIndex(df, text_col="question", page_size=3)

#     print("\n" + "="*60)
#     print("SEARCH 1 — AND: 'machine learning'")
#     print("="*60)
#     print(idx.query("machine learning", mode="AND").to_string(index=False))

#     print("\n" + "="*60)
#     print("SEARCH 2 — OR: 'deep learning'")
#     print("="*60)
#     print(idx.query("deep learning", mode="OR").to_string(index=False))

#     print("\n" + "="*60)
#     print("SEARCH 3 — AND: 'natural language processing'")
#     print("="*60)
#     print(idx.query("natural language processing", mode="AND").to_string(index=False))

#     print("\n" + "="*60)
#     print("SEARCH 4 — AND: 'neural networks'")
#     print("="*60)
#     print(idx.query("neural networks", mode="AND").to_string(index=False))












"""
PageIndex Semantic Search over a DataFrame
==========================================
Inspired by: https://github.com/VectifyAI/PageIndex

Core idea (from VectifyAI):
  - NO vector DB, NO embeddings
  - Build a hierarchical tree index (page summaries → LLM reasoning)
  - Use LLM to REASON over page summaries to find relevant pages
  - Retrieve rows only from relevant pages

Pipeline:
  Step 1: Split DataFrame into pages
  Step 2: LLM generates a semantic summary for each page
  Step 3: On query, LLM reasons over all summaries → picks relevant page IDs
  Step 4: Return rows from those pages
"""

import os
import json
import re
import pandas as pd
from anthropic import Anthropic

# ── Config ─────────────────────────────────────────────────────────────────────
PAGE_SIZE   = 3          # rows per page
MODEL       = "claude-sonnet-4-20250514"
MAX_TOKENS  = 1024

client = Anthropic()

# ── Sample DataFrame ───────────────────────────────────────────────────────────
data = {
    "id":  list(range(1, 16)),
    "qid": [f"q{i}" for i in range(1, 16)],
    "question": [
        "What is machine learning?",
        "How does deep learning differ from machine learning?",
        "What is supervised learning?",
        "Explain unsupervised learning with examples.",
        "What is natural language processing?",
        "How do neural networks work?",
        "What is reinforcement learning?",
        "How is deep learning used in computer vision?",
        "What are transformers in NLP?",
        "How does BERT work in natural language processing?",
        "What is gradient descent optimization?",
        "Explain the concept of overfitting in ML models.",
        "What is transfer learning and when is it useful?",
        "How do convolutional neural networks process images?",
        "What is the attention mechanism in transformers?",
    ]
}
df = pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 + 2: Build PageIndex tree — split into pages, summarize each with LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_page_index(df: pd.DataFrame, text_col: str, page_size: int = PAGE_SIZE) -> list[dict]:
    """
    Splits DataFrame into pages and asks the LLM to generate a semantic
    summary for each page — the 'table of contents' tree structure.
    """
    df = df.reset_index(drop=True)
    pages = []

    print(f"[PageIndex] Building index for {len(df)} rows (page_size={page_size})...")

    for page_id, start in enumerate(range(0, len(df), page_size)):
        end      = min(start + page_size, len(df))
        page_df  = df.iloc[start:end]
        rows_txt = "\n".join(
            f"  Row {row['id']}: {row[text_col]}"
            for _, row in page_df.iterrows()
        )

        # Ask LLM to summarize this page semantically
        prompt = f"""You are indexing rows from a dataset for semantic search.

Below are rows from page {page_id} of the dataset:
{rows_txt}

Write a concise semantic summary (2-3 sentences) describing the TOPICS and CONCEPTS
covered by these rows. This summary will be used by an LLM to decide if this page
is relevant to a user's query. Be specific about topics, not generic.

Respond with ONLY the summary text, nothing else."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response.content[0].text.strip()

        pages.append({
            "page_id"  : page_id,
            "row_range": (start, end - 1),
            "row_ids"  : list(page_df["id"]),
            "summary"  : summary,
        })

        print(f"  Page {page_id} (rows {start}-{end-1}): {summary[:80]}...")

    print(f"[PageIndex] Index built — {len(pages)} pages.\n")
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Tree Search — LLM reasons over page summaries to find relevant pages
# ══════════════════════════════════════════════════════════════════════════════

def tree_search(query: str, pages: list[dict]) -> list[int]:
    """
    Presents all page summaries to the LLM and asks it to reason about
    which pages are relevant to the query. Returns list of relevant page IDs.
    """
    summaries_txt = "\n\n".join(
        f"Page {p['page_id']} (rows {p['row_range'][0]}-{p['row_range'][1]}):\n  {p['summary']}"
        for p in pages
    )

    prompt = f"""You are a retrieval reasoning engine using the PageIndex approach.

A user has this query:
"{query}"

Below are semantic summaries of each page in the dataset index:

{summaries_txt}

Your task: reason step-by-step about which pages likely contain rows that are
RELEVANT to the user's query. Consider semantic meaning, not just keywords.
For example, a query about "ML algorithms" is relevant to pages about
"machine learning methods" even if exact words differ.

Respond ONLY with a JSON object in this exact format:
{{
  "reasoning": "brief explanation of your decision",
  "relevant_page_ids": [list of integer page IDs]
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    result = json.loads(raw)

    print(f"[TreeSearch] Reasoning: {result['reasoning']}")
    print(f"[TreeSearch] Relevant pages: {result['relevant_page_ids']}")
    return result["relevant_page_ids"]


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Retrieve rows from relevant pages
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, df: pd.DataFrame, pages: list[dict]) -> pd.DataFrame:
    """Full PageIndex pipeline: tree search → retrieve rows."""
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print(f"{'='*60}")

    relevant_page_ids = tree_search(query, pages)

    if not relevant_page_ids:
        print("[Retrieve] No relevant pages found.")
        return pd.DataFrame(columns=df.columns)

    # Collect all row IDs from relevant pages
    relevant_row_ids = []
    for pid in relevant_page_ids:
        page = next((p for p in pages if p["page_id"] == pid), None)
        if page:
            relevant_row_ids.extend(page["row_ids"])

    result = df[df["id"].isin(relevant_row_ids)].reset_index(drop=True)
    print(f"[Retrieve] {len(result)} row(s) returned.\n")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build the PageIndex tree (only once — cache this in production)
    pages = build_page_index(df, text_col="question", page_size=PAGE_SIZE)

    # Run semantic queries
    queries = [
        "ML algorithms and training methods",         # semantic: doesn't say 'machine learning'
        "how AI understands text and language",       # semantic: NLP-related
        "image recognition and visual processing",    # semantic: CV/CNN
        "preventing models from memorizing training data",  # semantic: overfitting
    ]

    for q in queries:
        result = retrieve(q, df, pages)
        print(result.to_string(index=False))
        print()
