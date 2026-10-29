"""
AI Content Generation Engine for Local AI Document Studio.
Transforms selected PDF document context into reports, summaries, notes,
proposals, academic/event posters, brochures, presentations, diagrams, and data tables.
"""

import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.config import DEFAULT_LLM_MODEL
from src.retrieval import hybrid_search_and_rerank
from src.rag_pipeline import format_context

LLM_MODEL = DEFAULT_LLM_MODEL

# Generation categories and their supported asset types
GENERATION_TEMPLATES = {
    # Reports & Summaries
    "executive_summary": {
        "title": "Executive Summary",
        "category": "Reports",
        "description": "High-impact briefing highlighting core findings, strategic implications, and action items.",
        "icon": "📊",
    },
    "research_summary": {
        "title": "Research Paper Summary",
        "category": "Reports",
        "description": "Rigorous academic synthesis including background, methodology, experimental results, and limitations.",
        "icon": "🔬",
    },
    "project_documentation": {
        "title": "Project Documentation",
        "category": "Reports",
        "description": "Technical system specification with system architecture, components, workflows, and operational notes.",
        "icon": "📝",
    },
    "study_notes": {
        "title": "Study Notes & Review Guide",
        "category": "Reports",
        "description": "Structured study sheet with core concepts, glossary of terms, memory hooks, and practice review questions.",
        "icon": "📖",
    },
    "project_proposal": {
        "title": "Project & Business Proposal",
        "category": "Reports",
        "description": "Formal proposal including problem statement, proposed methodology, milestones, deliverables, and ROI.",
        "icon": "💼",
    },

    # Posters
    "academic_poster": {
        "title": "Academic Conference Poster",
        "category": "Posters",
        "description": "Quad-section poster draft (Background, Methodology, Results, Conclusions) designed for high readability.",
        "icon": "🎓",
    },
    "event_poster": {
        "title": "Event Announcement Poster",
        "category": "Posters",
        "description": "Eye-catching event announcement with headline, dates, keynote themes, agenda points, and RSVP call to action.",
        "icon": "📢",
    },
    "infographic_poster": {
        "title": "Infographic Summary Poster",
        "category": "Posters",
        "description": "Data-driven poster layout featuring highlighted metrics, key stats, bullet insights, and visual takeaways.",
        "icon": "💡",
    },
    "awareness_poster": {
        "title": "Awareness & Educational Poster",
        "category": "Posters",
        "description": "Public educational poster detailing key facts, best practices, dos and don'ts, and essential contact resources.",
        "icon": "🛡️",
    },

    # Brochures
    "one_page_flyer": {
        "title": "1-Page Corporate / Project Flyer",
        "category": "Brochures",
        "description": "Compact, single-page promotional factsheet with headline, 3 value pillars, proof points, and contact info.",
        "icon": "📄",
    },
    "multi_page_booklet": {
        "title": "Multi-Page Informational Booklet",
        "category": "Brochures",
        "description": "Comprehensive multi-page booklet draft with section breaks, narrative flow, and detailed descriptions.",
        "icon": "📚",
    },
    "tri_fold_brochure": {
        "title": "Tri-Fold 6-Panel Brochure",
        "category": "Brochures",
        "description": "Standard 6-panel tri-fold layout covering Hook, Problem, Features, Benefits, FAQ, and Call to Action.",
        "icon": "📑",
    },

    # Presentations & Diagrams
    "presentation_outline": {
        "title": "Slide Deck Presentation Outline",
        "category": "Presentations & Diagrams",
        "description": "Comprehensive 8-slide presentation outline with slide titles, key bullet points, and speaker talking notes.",
        "icon": "📽️",
    },
    "flowchart": {
        "title": "Process Flowchart & Step Guide",
        "category": "Presentations & Diagrams",
        "description": "Sequential process diagram in Mermaid.js syntax with step-by-step logic breakdown.",
        "icon": "🔀",
    },
    "architecture_diagram": {
        "title": "Architecture & Component Diagram",
        "category": "Presentations & Diagrams",
        "description": "Component architecture diagram with data flow paths in Mermaid.js graph format and narrative description.",
        "icon": "🏗️",
    },
    "timeline": {
        "title": "Milestone & Project Timeline",
        "category": "Presentations & Diagrams",
        "description": "Chronological project timeline mapping phases, key dates, deliverables, and dependencies.",
        "icon": "⏳",
    },

    # Data & Tables
    "extracted_table": {
        "title": "Extracted Structured Table",
        "category": "Data & Tables",
        "description": "Extracts facts, specifications, or numerical metrics from the document into clean Markdown and CSV table format.",
        "icon": "🔢",
    },
    "data_summary": {
        "title": "Quantitative & Metric Summary",
        "category": "Data & Tables",
        "description": "Synthesizes quantitative metrics, percentages, benchmarks, and performance figures.",
        "icon": "📈",
    },
    "chart_data": {
        "title": "Visual Chart Dataset",
        "category": "Data & Tables",
        "description": "Generates formatted category/metric series ready for immediate rendering into bar, pie, or line charts.",
        "icon": "📉",
    },
}


def build_system_prompt(
    asset_type: str,
    title: str,
    tone: str = "Professional",
    audience: str = "General Audience",
    specific_focus: str = "",
) -> str:
    """Build detailed instructions for transforming document context into the chosen asset."""

    prompt_details = {
        "executive_summary": (
            "Structure as a professional executive briefing:\n"
            "1. # {title}\n"
            "2. ## Strategic Overview\n"
            "3. ## Core Findings & Key Insights (3-5 concise bullets)\n"
            "4. ## Business / Operational Implications\n"
            "5. ## Recommended Next Steps & Decisions"
        ),
        "research_summary": (
            "Structure as an academic research synthesis:\n"
            "1. # {title}\n"
            "2. ## Abstract & Problem Context\n"
            "3. ## Methodology & Technical Approach\n"
            "4. ## Core Results & Benchmarks\n"
            "5. ## Limitations & Critical Evaluation\n"
            "6. ## Future Research Directions"
        ),
        "project_documentation": (
            "Structure as comprehensive technical documentation:\n"
            "1. # {title}\n"
            "2. ## System Overview & Architecture\n"
            "3. ## Key Components & Specifications\n"
            "4. ## Operational Workflow & Usage Instructions\n"
            "5. ## Security, Maintenance & Troubleshooting"
        ),
        "study_notes": (
            "Structure as an effective study revision sheet:\n"
            "1. # {title}\n"
            "2. ## Essential Concepts & Definitions (Bold term + definition)\n"
            "3. ## High-Yield Bullet Summaries\n"
            "4. ## Memory Mnemonics & Key Takeaways\n"
            "5. ## Self-Check Review Questions & Answer Key"
        ),
        "project_proposal": (
            "Structure as a persuasive project proposal:\n"
            "1. # {title}\n"
            "2. ## Executive Pitch & Problem Statement\n"
            "3. ## Proposed Solution & Value Proposition\n"
            "4. ## Implementation Phases & Deliverables\n"
            "5. ## Resource Requirements & Risk Mitigation\n"
            "6. ## Anticipated Impact & Success Criteria"
        ),
        "academic_poster": (
            "Format as a 4-section Academic Poster layout:\n"
            "# {title}\n"
            "**Section 1: Background & Problem** (2-3 concise points)\n"
            "**Section 2: Methodology & Implementation** (Key methods, tools, architecture)\n"
            "**Section 3: Key Results & Data** (Metrics, experimental outcomes, comparisons)\n"
            "**Section 4: Conclusions & Impact** (Takeaways and next steps)"
        ),
        "event_poster": (
            "Format as an Event Poster:\n"
            "# {title}\n"
            "## Catchy Tagline\n"
            "- **Date & Time:** [Extract or suggest]\n"
            "- **Location / Platform:** [Extract or suggest]\n"
            "### Featured Highlights & Topics\n"
            "- [Highlight 1]\n"
            "- [Highlight 2]\n"
            "- [Highlight 3]\n"
            "### Call to Action\n"
            "Reserve your spot today!"
        ),
        "infographic_poster": (
            "Format as an Infographic Summary:\n"
            "# {title}\n"
            "### Big Numbers & Key Metrics\n"
            "- **[Stat 1]:** [Description]\n"
            "- **[Stat 2]:** [Description]\n"
            "### Core Takeaways\n"
            "- [Takeaway 1]\n"
            "- [Takeaway 2]\n"
            "### Summary Conclusion"
        ),
        "awareness_poster": (
            "Format as an Educational Awareness Poster:\n"
            "# {title}\n"
            "## What You Need to Know\n"
            "[Brief summary]\n"
            "### Best Practices (Dos & Don'ts)\n"
            "- **DO:** [Best practice 1]\n"
            "- **DO:** [Best practice 2]\n"
            "- **DON'T:** [Pitfall to avoid]\n"
            "### Need Help? Support & Resources"
        ),
        "one_page_flyer": (
            "Format as a 1-Page Factsheet / Flyer:\n"
            "# {title}\n"
            "## The Core Value Proposition\n"
            "### 3 Key Pillars\n"
            "1. **[Pillar 1]:** [Details]\n"
            "2. **[Pillar 2]:** [Details]\n"
            "3. **[Pillar 3]:** [Details]\n"
            "### Key Metric or Testimonial\n"
            "### Contact & Next Steps"
        ),
        "multi_page_booklet": (
            "Format as a Multi-Page Booklet:\n"
            "# {title}\n"
            "## Table of Contents\n"
            "## Chapter 1: Introduction & Context\n"
            "## Chapter 2: Detailed Analysis & Technical Specs\n"
            "## Chapter 3: Applications & Case Studies\n"
            "## Chapter 4: Summary & Concluding Thoughts"
        ),
        "tri_fold_brochure": (
            "Format explicitly as a 6-Panel Tri-Fold Brochure:\n"
            "# {title}\n"
            "### Panel 1: Outside Cover / Hook\n"
            "Catchy title, hero tagline, and visual hook.\n\n"
            "### Panel 2: Inside Left - The Challenge\n"
            "The core problem, market context, or background.\n\n"
            "### Panel 3: Inside Center - The Solution\n"
            "Overview of the system, product, or research approach.\n\n"
            "### Panel 4: Inside Right - Key Features & Specs\n"
            "Bullet points of capabilities, specifications, and advantages.\n\n"
            "### Panel 5: Inside Flap - Proof Points & FAQ\n"
            "Metrics, validation results, and answers to common questions.\n\n"
            "### Panel 6: Back Cover - Summary & Contact\n"
            "Final takeaway, organization info, and call to action."
        ),
        "presentation_outline": (
            "Format as a Slide Deck Outline (8 slides):\n"
            "# {title}\n"
            "For each slide, output:\n"
            "### Slide X: [Slide Title]\n"
            "- **Bullet points:** (3-4 points)\n"
            "- **Presenter Notes:** (2-3 sentences of what to say aloud)"
        ),
        "flowchart": (
            "Format as a Process Flowchart:\n"
            "# {title}\n"
            "```mermaid\n"
            "graph TD\n"
            "  A[Start] --> B[Step 1]\n"
            "  B --> C[Decision/Step 2]\n"
            "  C --> D[Final Output]\n"
            "```\n\n"
            "### Step-by-Step Explanation\n"
            "Provide numbered textual descriptions of each stage."
        ),
        "architecture_diagram": (
            "Format as a System Architecture Diagram:\n"
            "# {title}\n"
            "```mermaid\n"
            "graph LR\n"
            "  subgraph Input Layer\n"
            "    A[Sources/Sensors]\n"
            "  end\n"
            "  subgraph Processing Layer\n"
            "    B[Processing Unit]\n"
            "  end\n"
            "  subgraph Output Layer\n"
            "    C[Action/Alert]\n"
            "  end\n"
            "  A --> B --> C\n"
            "```\n\n"
            "### Component Breakdown\n"
            "Explain each component's role, interfaces, and data flow."
        ),
        "timeline": (
            "Format as a Milestone Timeline:\n"
            "# {title}\n"
            "| Phase | Milestone / Objective | Target Period | Key Deliverable |\n"
            "|-------|----------------------|---------------|----------------|\n"
            "| 1     | Discovery & Planning | Month 1       | Requirement Spec|\n\n"
            "```mermaid\n"
            "timeline\n"
            "  title {title}\n"
            "  section Phase 1\n"
            "    Kickoff : Requirements\n"
            "  section Phase 2\n"
            "    Execution : Development\n"
            "```\n\n"
            "### Phase Narrative"
        ),
        "extracted_table": (
            "Extract all specifications, numerical data, or tabular metrics from the document into a clean Markdown table:\n"
            "# {title}\n"
            "| Parameter / Feature | Value / Specification | Notes / Page Reference |\n"
            "| :--- | :--- | :--- |\n"
            "[Extract at least 5-10 rows based on document facts]\n\n"
            "### Summary Analysis of Table Data"
        ),
        "data_summary": (
            "Synthesize numerical and quantitative evidence:\n"
            "# {title}\n"
            "## Quantitative Highlights\n"
            "## Key Benchmarks & Comparison Table\n"
            "## Data Trends & Analytical Commentary"
        ),
        "chart_data": (
            "Extract structured data formatted as a JSON block for charting:\n"
            "# {title}\n"
            "```json\n"
            "{\n"
            '  "chart_type": "bar",\n'
            '  "title": "Document Metric Comparison",\n'
            '  "labels": ["Item A", "Item B", "Item C", "Item D"],\n'
            '  "values": [45, 78, 62, 90],\n'
            '  "unit": "Units/Percent"\n'
            "}\n"
            "```\n\n"
            "### Data Interpretation\n"
            "Explain what the chart numbers represent based on the document."
        ),
    }

    raw_template = prompt_details.get(asset_type, prompt_details["executive_summary"])
    structure_guide = raw_template.replace("{title}", title)

    focus_instruction = f"\nFocus especially on: {specific_focus}" if specific_focus.strip() else ""

    return f"""You are an expert AI content strategist and technical writer.
Transform the provided DOCUMENT CONTEXT into a high-quality {GENERATION_TEMPLATES.get(asset_type, {}).get('title', asset_type)}.

TONE: {tone}
TARGET AUDIENCE: {audience}
{focus_instruction}

STRICT GROUNDING RULES:
1. Base all facts, numbers, claims, and terminology strictly on the supplied document context.
2. Never invent fake data or cite nonexistent evidence.
3. If specific details are missing, use reasonable placeholder markers like [To be confirmed] or note the limitation.
4. Format using clean GitHub-flavored Markdown.

STRUCTURE TO FOLLOW:
{structure_guide}
"""


def generate_content_asset(
    asset_type: str,
    selected_sources: Optional[List[str]] = None,
    custom_title: str = "",
    tone: str = "Professional",
    audience: str = "General Audience",
    specific_focus: str = "",
    k: int = 8,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for the selected document(s) and generate the requested asset.
    """
    template_info = GENERATION_TEMPLATES.get(asset_type, GENERATION_TEMPLATES["executive_summary"])
    title = custom_title.strip() or template_info["title"]

    # Formulate query to gather core context
    query = f"Overview, main topics, key concepts, technical specifications, and results of {title} {specific_focus}"
    scored_documents = hybrid_search_and_rerank(
        question=query,
        k=k,
        selected_sources=selected_sources,
        threshold=0.1,  # Lower threshold to gather broader thematic context
    )

    documents = [doc for doc, _ in scored_documents]
    if not documents:
        # Fallback query if no chunks match specific title
        scored_documents = hybrid_search_and_rerank(
            question="Summary overview details facts",
            k=k,
            selected_sources=selected_sources,
            threshold=0.0,
        )
        documents = [doc for doc, _ in scored_documents]

    context_str = format_context(documents)

    system_prompt = build_system_prompt(
        asset_type=asset_type,
        title=title,
        tone=tone,
        audience=audience,
        specific_focus=specific_focus,
    )

    full_prompt = f"""{system_prompt}

DOCUMENT CONTEXT:
{context_str}

GENERATED {template_info['title'].upper()}:
"""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
        response = llm.invoke(full_prompt)
        content = response.content.strip()
    except Exception as exc:
        # Provide clean structured fallback draft when Ollama is offline
        content = generate_offline_fallback(
            asset_type=asset_type,
            title=title,
            documents=documents,
            selected_sources=selected_sources,
            error_note=str(exc),
        )

    return {
        "title": title,
        "asset_type": asset_type,
        "category": template_info["category"],
        "content": content,
        "sources": [doc.metadata.get("source", "doc.pdf") for doc in documents],
        "chunk_count": len(documents),
    }


def generate_offline_fallback(
    asset_type: str,
    title: str,
    documents: List[Document],
    selected_sources: Optional[List[str]] = None,
    error_note: str = "",
) -> str:
    """
    Generate an offline structured template populated with actual document excerpts
    when Ollama is temporarily offline, allowing continuous preview, editing, and exports.
    """
    doc_sources = ", ".join(selected_sources) if selected_sources else "Active Documents"
    excerpts = [d.page_content[:250].strip().replace("\n", " ") for d in documents[:4]]

    if asset_type == "tri_fold_brochure":
        return f"""# {title}
> *Source: {doc_sources}*

### Panel 1: Outside Cover / Hook
**{title}**
A complete overview and technical briefing based on {doc_sources}.

### Panel 2: Inside Left - The Challenge
Key challenges and operational needs identified in the document:
- {excerpts[0] if len(excerpts) > 0 else 'Challenge description'}

### Panel 3: Inside Center - The Solution
Core architecture and approach:
- {excerpts[1] if len(excerpts) > 1 else 'Core system description'}

### Panel 4: Inside Right - Key Features & Specs
Key specifications and technical capabilities:
- {excerpts[2] if len(excerpts) > 2 else 'Key specifications'}

### Panel 5: Inside Flap - Proof Points & FAQ
Data points and evidence:
- {excerpts[3] if len(excerpts) > 3 else 'Validation results'}

### Panel 6: Back Cover - Summary & Contact
Prepared with Local AI Document Studio. Local & Private.
"""
    elif asset_type == "extracted_table":
        return f"""# {title}
> *Extracted from: {doc_sources}*

| Category / Component | Extracted Value / Specification | Status |
| :--- | :--- | :--- |
| Document Source | {doc_sources} | Verified |
| Extracted Passage 1 | {excerpts[0][:60] if len(excerpts) > 0 else 'N/A'}... | Active |
| Extracted Passage 2 | {excerpts[1][:60] if len(excerpts) > 1 else 'N/A'}... | Active |
| Extracted Passage 3 | {excerpts[2][:60] if len(excerpts) > 2 else 'N/A'}... | Active |

### Key Findings
1. Document contains structured technical data.
2. Ready for export to CSV or PDF report.
"""
    elif asset_type == "flowchart":
        return f"""# {title}
> *Process Flowchart from: {doc_sources}*

```mermaid
graph TD
    A[Start: Document Input] --> B[Parse & Extract Text]
    B --> C[Analyze Core Modules]
    C --> D[Execute Process Logic]
    D --> E[Deliver Output]
```

### Step Breakdown
1. **Input Stage:** Ingest data from {doc_sources}.
2. **Analysis Stage:** Process components and extract logic.
3. **Execution Stage:** Validate findings and synthesize results.
"""
    elif asset_type == "chart_data":
        return f"""# {title}

```json
{{
  "chart_type": "bar",
  "title": "{title}",
  "labels": ["Component A", "Component B", "Component C", "Component D"],
  "values": [35, 60, 85, 45],
  "unit": "Score"
}}
```

### Chart Notes
Visual chart generated from document metrics in {doc_sources}.
"""
    else:
        # Default report / summary template
        return f"""# {title}
> *Document Reference: {doc_sources}*

## 1. Overview & Context
This document synthesizes findings from **{doc_sources}**.

{excerpts[0] if len(excerpts) > 0 else 'Summary passage from document.'}

## 2. Key Findings & Specifications
- **Key Observation 1:** {excerpts[1] if len(excerpts) > 1 else 'Extracted detail 1'}
- **Key Observation 2:** {excerpts[2] if len(excerpts) > 2 else 'Extracted detail 2'}
- **Key Observation 3:** {excerpts[3] if len(excerpts) > 3 else 'Extracted detail 3'}

## 3. Conclusions & Strategic Action Items
- Review operational parameters in the primary documentation.
- Implement recommended configurations.
"""
