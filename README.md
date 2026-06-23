# SLM Document Extractor

A robust, 100% local pipeline for extracting complex, heavily-formatted documents (such as competitive exam question banks containing advanced mathematical equations, tables, and mixed text) using Small Language Models (SLMs) and Vision AI.

## 🚀 Overview

Extracting structured data from PDFs is notoriously difficult when the documents contain complex mathematical LaTeX, watermarks, and dense layouts. Traditional OCR models either fail to capture mathematical formulas or hallucinate severe structural noise when encountering watermarks.

This pipeline solves this by orchestrating a dynamic, chunked local architecture:
1. **Vision-Based Extraction:** Uses **Pix2Text** (an orchestration of specialized vision models) to identify layout structures and directly translate mathematical equations from images into pure `LaTeX`.
2. **Context-Aware OCR Correction:** Because underlying OCR models are often trained on specific datasets (e.g., Chinese datasets), they can hallucinate foreign characters or garbled text when reading watermarks. This pipeline dynamically routes small chunks of the corrupted OCR text to a local **Llama-3.2 (3B)** agent.
3. **Dynamic Reassembly:** The SLM agent logically cleans and reconstructs the English text *without* hardcoded regex rules, while strictly preserving the mathematical LaTeX formatting. 

## ✨ Key Features

- **100% Local Processing:** No cloud APIs required. Runs entirely on local CPU hardware using Ollama and local PyTorch Vision models.
- **Math Equation Preservation:** Successfully extracts inline (`$`) and block (`$$`) math equations natively to LaTeX.
- **Anti-Freezing Architecture:** Avoids the $O(N^2)$ "Context Window Explosion" that crashes CPUs by strategically chunking the document into question blocks *before* feeding them to the SLM agent.
- **Database Generation:** Automatically parses the cleaned output into a structured SQLite database (`questions.db`) categorizing Questions, Options, and Explanations.

## 🛠️ Technology Stack

- **Pix2Text:** For vision-based document layout analysis and Math-to-LaTeX conversion.
- **Ollama (Llama-3.2 3B):** Local SLM agent for dynamic OCR noise correction and structural recovery.
- **Python (re, sqlite3):** For chunking routers, dynamic filtering, and database management.
- **Jupyter:** For rendering the final structured output and MathJax equations.

## 📝 Pipeline Architecture

1. **Input:** Dense PDF document.
2. **Vision Extraction:** Pix2Text converts page images into raw Markdown + LaTeX.
3. **Regex Filtering:** A lightweight filter strips out hallucinated CJK characters caused by background watermarks.
4. **Chunking Router:** The raw text is chunked into small, individual question blocks.
5. **SLM Validation:** The Llama-3.2 agent processes each chunk, semantically recovering missing English words and fixing structural noise.
6. **Output:** A cleanly structured SQLite Database.
