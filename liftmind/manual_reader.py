"""
LiftMind Manual Reader - Direct PDF Reading Fallback

When the database doesn't have a confident answer, read the actual PDFs directly.
This is the safety net that ensures no information is missed, similar to how
NotebookLM reads source documents directly.

Usage:
    from liftmind.manual_reader import search_manuals_direct
    results = await search_manuals_direct(query, model_filter="Tresa")
"""

import os
import re
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Dict, Optional
import threading

from liftmind.config import settings

logger = logging.getLogger(__name__)

# Page text cache: (filename, page_num) -> text
# Proper LRU cache using OrderedDict for efficient eviction
_page_cache: OrderedDict[tuple, str] = OrderedDict()
_page_cache_lock = threading.Lock()
_PAGE_CACHE_MAX = 500


def _get_cached_page(filepath: str, page_num: int) -> Optional[str]:
    """Get cached page text, moving accessed item to end (LRU)."""
    key = (os.path.basename(filepath), page_num)
    with _page_cache_lock:
        if key in _page_cache:
            # Move to end (most recently used)
            _page_cache.move_to_end(key)
            return _page_cache[key]
    return None


def _cache_page(filepath: str, page_num: int, text: str):
    """Cache page text with proper LRU eviction."""
    key = (os.path.basename(filepath), page_num)
    with _page_cache_lock:
        if key in _page_cache:
            # Update existing and move to end
            _page_cache.move_to_end(key)
            _page_cache[key] = text
        else:
            # Add new entry
            _page_cache[key] = text
            # Evict oldest (first) entries if over capacity
            while len(_page_cache) > _PAGE_CACHE_MAX:
                _page_cache.popitem(last=False)  # Remove oldest (first)


def _extract_query_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query for page matching."""
    stop_words = {'is', 'the', 'a', 'an', 'what', 'how', 'why', 'can', 'could',
                  'would', 'should', 'does', 'do', 'i', 'my', 'me', 'we', 'help',
                  'please', 'need', 'want', 'have', 'has', 'be', 'am', 'are', 'was',
                  'it', 'its', 'this', 'that', 'with', 'for', 'on', 'at', 'to', 'from',
                  'of', 'in', 'and', 'or', 'but', 'if', 'so'}

    words = re.findall(r'\b[a-zA-Z0-9]+\b', query)
    keywords = [w for w in words if w.lower() not in stop_words and len(w) >= 2]
    return keywords


def _search_single_pdf(filepath: str, keywords: List[str],
                       max_pages: int = 5) -> List[Dict]:
    """
    Search a single PDF for pages matching the query keywords.

    Returns list of matching page dicts with content and metadata.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not available for direct PDF reading")
        return []

    filename = os.path.basename(filepath)
    matching_pages = []
    keywords_lower = [k.lower() for k in keywords]

    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Check cache first
                cached = _get_cached_page(filepath, page_num)
                if cached is not None:
                    page_text = cached
                else:
                    page_text = page.extract_text() or ""
                    _cache_page(filepath, page_num, page_text)

                if not page_text.strip():
                    continue

                page_lower = page_text.lower()

                # Count keyword matches on this page
                match_count = sum(1 for kw in keywords_lower if kw in page_lower)

                if match_count >= 1:  # At least one keyword match
                    matching_pages.append({
                        "page_num": page_num,
                        "text": page_text,
                        "match_count": match_count,
                        "match_ratio": match_count / len(keywords_lower) if keywords_lower else 0,
                        "filename": filename
                    })

        # Sort by match count descending, take top pages
        matching_pages.sort(key=lambda x: x["match_count"], reverse=True)

        # For the top matches, also grab surrounding pages for context
        result_pages = []
        seen_page_nums = set()

        for match in matching_pages[:max_pages]:
            pn = match["page_num"]
            for offset in [0, -1, 1]:  # Current page + neighbors
                target_pn = pn + offset
                if target_pn < 1 or target_pn in seen_page_nums:
                    continue
                seen_page_nums.add(target_pn)

                # Get text (may need to re-read neighbor pages)
                cached = _get_cached_page(filepath, target_pn)
                if cached is not None:
                    text = cached
                else:
                    # Re-open to get neighbor page
                    try:
                        with pdfplumber.open(filepath) as pdf2:
                            if target_pn <= len(pdf2.pages):
                                text = pdf2.pages[target_pn - 1].extract_text() or ""
                                _cache_page(filepath, target_pn, text)
                            else:
                                continue
                    except Exception:
                        continue

                if text.strip():
                    result_pages.append({
                        "filename": filename,
                        "page_number": target_pn,
                        "content": text,
                        "match_count": match["match_count"] if offset == 0 else 0,
                        "source": "direct_pdf_read"
                    })

        return result_pages

    except Exception as e:
        logger.error(f"Error reading PDF {filepath}: {e}")
        return []


def _get_relevant_pdfs(model_filter: Optional[str] = None) -> List[str]:
    """Get list of PDF files to search, optionally filtered by model."""
    manuals_dir = settings.MANUALS_DIR
    if not os.path.exists(manuals_dir):
        return []

    pdf_files = []

    if model_filter:
        # Search the specific model directory first
        model_dir = os.path.join(manuals_dir, model_filter)
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                if f.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(model_dir, f))

        # If no model-specific PDFs, search all
        if not pdf_files:
            logger.info(f"No PDFs in {model_filter} dir, searching all models")
            model_filter = None

    if not model_filter:
        for model_dir_name in os.listdir(manuals_dir):
            model_path = os.path.join(manuals_dir, model_dir_name)
            if os.path.isdir(model_path):
                for f in os.listdir(model_path):
                    if f.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(model_path, f))

    return pdf_files


def search_manuals_direct(query: str,
                          model_filter: Optional[str] = None,
                          max_workers: int = 4,
                          max_results: int = 8) -> List[Dict]:
    """
    Search PDF manuals directly by reading pages and matching keywords.

    This is the fallback when the database doesn't have confident results.
    Runs in parallel across multiple PDFs for speed.

    Args:
        query: The user's search query
        model_filter: Optional lift model to narrow PDF selection
        max_workers: Number of parallel PDF readers
        max_results: Maximum results to return

    Returns:
        List of result dicts compatible with RAG pipeline format:
        [{filename, page_number, content, lift_model, source, cross_model}]
    """
    start_time = time.time()

    keywords = _extract_query_keywords(query)
    if not keywords:
        logger.warning("No meaningful keywords extracted from query for direct PDF search")
        return []

    pdf_files = _get_relevant_pdfs(model_filter)
    if not pdf_files:
        logger.warning(f"No PDF files found for model_filter={model_filter}")
        return []

    logger.info(f"Direct PDF search: {len(keywords)} keywords across {len(pdf_files)} PDFs "
                f"(model={model_filter})")

    # Search PDFs in parallel - use configurable timeout
    pdf_read_timeout = settings.PDF_READ_TIMEOUT
    all_results = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(pdf_files))) as executor:
        futures = {
            executor.submit(_search_single_pdf, pdf, keywords, max_pages=3): pdf
            for pdf in pdf_files
        }

        for future in as_completed(futures, timeout=pdf_read_timeout):
            try:
                results = future.result(timeout=5)
                all_results.extend(results)
            except Exception as e:
                pdf = futures[future]
                logger.error(f"PDF search failed for {os.path.basename(pdf)}: {e}")

    # Sort by match count, deduplicate
    all_results.sort(key=lambda x: x.get("match_count", 0), reverse=True)

    # Convert to RAG-compatible format
    formatted_results = []
    seen = set()
    for r in all_results:
        key = (r["filename"], r["page_number"])
        if key in seen:
            continue
        seen.add(key)

        # Infer lift model from directory structure
        lift_model = model_filter
        if not lift_model:
            # Try to infer from path
            for pdf in pdf_files:
                if os.path.basename(pdf) == r["filename"]:
                    parent_dir = os.path.basename(os.path.dirname(pdf))
                    lift_model = parent_dir
                    break

        formatted_results.append({
            "filename": r["filename"],
            "page_number": r["page_number"],
            "content": r["content"],
            "lift_model": lift_model,
            "source": "direct_pdf_read",
            "cross_model": False,
            "file_type": "pdf"
        })

        if len(formatted_results) >= max_results:
            break

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Direct PDF search found {len(formatted_results)} results in {elapsed_ms}ms")

    return formatted_results
