"""
LiftMind Multi-Stage Search Pipeline

Search strategy (fastest to slowest):
1. Check pre-computed Q&A pairs for direct match
2. Check entity registry for specific lookups (part numbers, specs)
3. Check verified fixes from feedback
4. Semantic search on facts
5. Fall back to chunk search

Every answer cites specific source.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from liftmind.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with source citation."""
    content: str
    source_type: str  # 'qa_pair', 'entity', 'verified_fix', 'fact', 'chunk'
    confidence: float  # 0.0 to 1.0
    source_document: Optional[str] = None
    source_page: Optional[int] = None
    lift_model: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class SearchResponse:
    """Complete search response with multiple results."""
    query: str
    lift_model: Optional[str]
    results: List[SearchResult]
    search_time_ms: int
    search_stages_used: List[str]
    has_images: bool = False
    relevant_images: List[Dict] = None


def get_db():
    """Get database connection with dict cursor."""
    conn = psycopg2.connect(settings.DATABASE_URL)
    return conn


def detect_query_intent(query: str) -> Tuple[str, Optional[str]]:
    """
    Detect the intent of a query and extract lift model if mentioned.

    Returns: (intent, lift_model)
    Intent types: 'spec_lookup', 'troubleshooting', 'howto', 'wiring', 'part_lookup', 'general'
    """
    query_lower = query.lower()

    # Detect lift model
    lift_model = None
    known_models = [
        'elfo', 'elfo 2', 'elfo xl', 'elfo traction', 'elfo hydraulic',
        'bari', 'e3', 'tresa', 'freedom', 'freedom maxi', 'freedom step',
        'p4', 'pollock', 'supermec', 'supermec 2', 'supermec 3',
        'elvoron', 'boutique'
    ]
    for model in known_models:
        if model in query_lower:
            lift_model = model.title()
            break

    # Detect intent
    if any(x in query_lower for x in ['error', 'fault', 'problem', 'issue', 'not working', "won't", 'stuck', 'troubleshoot']):
        return 'troubleshooting', lift_model

    if any(x in query_lower for x in ['how do i', 'how to', 'steps to', 'procedure for', 'install', 'replace', 'adjust']):
        return 'howto', lift_model

    if any(x in query_lower for x in ['wire', 'wiring', 'terminal', 'connection', 'circuit']):
        return 'wiring', lift_model

    if any(x in query_lower for x in ['part number', 'p/n', 'part #', 'order']):
        return 'part_lookup', lift_model

    if any(x in query_lower for x in ['what is', 'spec', 'specification', 'torque', 'voltage', 'setting', 'value']):
        return 'spec_lookup', lift_model

    return 'general', lift_model


def search_qa_pairs(query: str, lift_model: Optional[str] = None, limit: int = 3) -> List[SearchResult]:
    """
    Stage 1: Search pre-computed Q&A pairs.
    Fastest path - direct question matching.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Full-text search on questions
    if lift_model:
        cur.execute("""
            SELECT question, answer_summary, full_answer, category, lift_models,
                   ts_rank(question_tsv, plainto_tsquery('english', %s)) as rank
            FROM qa_pairs
            WHERE question_tsv @@ plainto_tsquery('english', %s)
              AND %s = ANY(lift_models)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, lift_model, limit))
    else:
        cur.execute("""
            SELECT question, answer_summary, full_answer, category, lift_models,
                   ts_rank(question_tsv, plainto_tsquery('english', %s)) as rank
            FROM qa_pairs
            WHERE question_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, limit))

    results = []
    for row in cur.fetchall():
        # Use full answer if available, otherwise summary
        content = row['full_answer'] or row['answer_summary']
        results.append(SearchResult(
            content=content,
            source_type='qa_pair',
            confidence=min(0.95, 0.5 + row['rank'] * 0.5),
            lift_model=row['lift_models'][0] if row['lift_models'] else None,
            metadata={
                'matched_question': row['question'],
                'category': row['category']
            }
        ))

    cur.close()
    conn.close()
    return results


def search_entities(query: str, lift_model: Optional[str] = None, limit: int = 5) -> List[SearchResult]:
    """
    Stage 2: Search entity registry.
    For spec lookups, part numbers, error codes.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Extract potential identifiers from query
    # Look for patterns like: E07, P/N 12345, T6, 25Nm
    potential_ids = re.findall(r'[A-Z]\d+|P/N\s*\d+|\d+\s*(?:Nm|mm|V|A|Hz)|T\d+', query, re.IGNORECASE)

    results = []

    # Search by identifier first (most precise)
    for identifier in potential_ids:
        normalized = identifier.lower().replace(" ", "")
        cur.execute("""
            SELECT entity_type, identifier, description, value, unit, lift_models
            FROM entities
            WHERE identifier_normalized = %s
        """, (normalized,))

        for row in cur.fetchall():
            if lift_model and lift_model not in (row['lift_models'] or []):
                continue
            results.append(SearchResult(
                content=f"{row['identifier']}: {row['description']}",
                source_type='entity',
                confidence=0.95,
                lift_model=row['lift_models'][0] if row['lift_models'] else None,
                metadata={
                    'entity_type': row['entity_type'],
                    'value': row['value'],
                    'unit': row['unit']
                }
            ))

    # Also do text search on descriptions
    if lift_model:
        cur.execute("""
            SELECT entity_type, identifier, description, value, unit, lift_models,
                   ts_rank(description_tsv, plainto_tsquery('english', %s)) as rank
            FROM entities
            WHERE description_tsv @@ plainto_tsquery('english', %s)
              AND %s = ANY(lift_models)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, lift_model, limit))
    else:
        cur.execute("""
            SELECT entity_type, identifier, description, value, unit, lift_models,
                   ts_rank(description_tsv, plainto_tsquery('english', %s)) as rank
            FROM entities
            WHERE description_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, limit))

    for row in cur.fetchall():
        # Avoid duplicates
        if any(r.content.startswith(row['identifier']) for r in results):
            continue
        results.append(SearchResult(
            content=f"{row['identifier']}: {row['description']}",
            source_type='entity',
            confidence=0.6 + row['rank'] * 0.3,
            lift_model=row['lift_models'][0] if row['lift_models'] else None,
            metadata={
                'entity_type': row['entity_type'],
                'value': row['value'],
                'unit': row['unit']
            }
        ))

    cur.close()
    conn.close()
    return results[:limit]


def search_verified_fixes(query: str, lift_model: Optional[str] = None, limit: int = 3) -> List[SearchResult]:
    """
    Stage 3: Search verified fixes from technician feedback.
    Human-confirmed solutions are gold.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Build search query
    search_terms = query.split()

    if lift_model:
        cur.execute("""
            SELECT id, lift_model, issue_description, symptoms, verified_solution,
                   was_bot_correct, contributor_name, times_cited, times_confirmed_helpful
            FROM verified_fixes
            WHERE approved = true
              AND lift_model = %s
              AND (
                  issue_description ILIKE ANY(%s)
                  OR verified_solution ILIKE ANY(%s)
              )
            ORDER BY times_confirmed_helpful DESC, times_cited DESC
            LIMIT %s
        """, (lift_model, [f'%{t}%' for t in search_terms], [f'%{t}%' for t in search_terms], limit))
    else:
        cur.execute("""
            SELECT id, lift_model, issue_description, symptoms, verified_solution,
                   was_bot_correct, contributor_name, times_cited, times_confirmed_helpful
            FROM verified_fixes
            WHERE approved = true
              AND (
                  issue_description ILIKE ANY(%s)
                  OR verified_solution ILIKE ANY(%s)
              )
            ORDER BY times_confirmed_helpful DESC, times_cited DESC
            LIMIT %s
        """, ([f'%{t}%' for t in search_terms], [f'%{t}%' for t in search_terms], limit))

    results = []
    for row in cur.fetchall():
        # Calculate confidence based on confirmation rate
        if row['times_cited'] > 0:
            confidence = 0.7 + (row['times_confirmed_helpful'] / row['times_cited']) * 0.3
        else:
            confidence = 0.75

        results.append(SearchResult(
            content=row['verified_solution'],
            source_type='verified_fix',
            confidence=confidence,
            lift_model=row['lift_model'],
            metadata={
                'issue': row['issue_description'],
                'symptoms': row['symptoms'],
                'contributor': row['contributor_name'],
                'times_used': row['times_cited']
            }
        ))

    cur.close()
    conn.close()
    return results


def search_facts(query: str, lift_model: Optional[str] = None, limit: int = 5) -> List[SearchResult]:
    """
    Stage 4: Search extracted facts.
    Structured knowledge from documents.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if lift_model:
        cur.execute("""
            SELECT f.id, f.fact_type, f.content, f.keywords, f.page, f.section_path,
                   f.confidence_score, f.lift_models,
                   d.filename, d.title,
                   ts_rank(f.content_tsv, plainto_tsquery('english', %s)) as rank
            FROM facts f
            JOIN documents d ON f.document_id = d.id
            WHERE f.content_tsv @@ plainto_tsquery('english', %s)
              AND %s = ANY(f.lift_models)
            ORDER BY f.confidence_score DESC, rank DESC
            LIMIT %s
        """, (query, query, lift_model, limit))
    else:
        cur.execute("""
            SELECT f.id, f.fact_type, f.content, f.keywords, f.page, f.section_path,
                   f.confidence_score, f.lift_models,
                   d.filename, d.title,
                   ts_rank(f.content_tsv, plainto_tsquery('english', %s)) as rank
            FROM facts f
            JOIN documents d ON f.document_id = d.id
            WHERE f.content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY f.confidence_score DESC, rank DESC
            LIMIT %s
        """, (query, query, limit))

    results = []
    for row in cur.fetchall():
        results.append(SearchResult(
            content=row['content'],
            source_type='fact',
            confidence=row['confidence_score'] * (0.5 + row['rank'] * 0.5),
            source_document=row['filename'],
            source_page=row['page'],
            lift_model=row['lift_models'][0] if row['lift_models'] else None,
            metadata={
                'fact_type': row['fact_type'],
                'section': row['section_path'],
                'keywords': row['keywords']
            }
        ))

    cur.close()
    conn.close()
    return results


def search_chunks(query: str, lift_model: Optional[str] = None, limit: int = 5) -> List[SearchResult]:
    """
    Stage 5: Fall back to raw chunk search.
    Broadest search, lowest confidence.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if lift_model:
        cur.execute("""
            SELECT c.id, c.content, c.section_path, c.page_start, c.chunk_type, c.lift_models,
                   d.filename, d.title,
                   ts_rank(c.content_tsv, plainto_tsquery('english', %s)) as rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.content_tsv @@ plainto_tsquery('english', %s)
              AND %s = ANY(c.lift_models)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, lift_model, limit))
    else:
        cur.execute("""
            SELECT c.id, c.content, c.section_path, c.page_start, c.chunk_type, c.lift_models,
                   d.filename, d.title,
                   ts_rank(c.content_tsv, plainto_tsquery('english', %s)) as rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, limit))

    results = []
    for row in cur.fetchall():
        results.append(SearchResult(
            content=row['content'],
            source_type='chunk',
            confidence=0.3 + row['rank'] * 0.4,  # Lower confidence for raw chunks
            source_document=row['filename'],
            source_page=row['page_start'],
            lift_model=row['lift_models'][0] if row['lift_models'] else None,
            metadata={
                'section': row['section_path'],
                'chunk_type': row['chunk_type']
            }
        ))

    cur.close()
    conn.close()
    return results


def search_images(query: str, lift_model: Optional[str] = None, limit: int = 3) -> List[Dict]:
    """Search for relevant images/diagrams."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if lift_model:
        cur.execute("""
            SELECT filename, file_path, relative_path, description, category, lift_models, page
            FROM images
            WHERE (
                description_tsv @@ plainto_tsquery('english', %s)
                OR extracted_text_tsv @@ plainto_tsquery('english', %s)
            )
            AND %s = ANY(lift_models)
            ORDER BY
                ts_rank(description_tsv, plainto_tsquery('english', %s)) +
                ts_rank(extracted_text_tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s
        """, (query, query, lift_model, query, query, limit))
    else:
        cur.execute("""
            SELECT filename, file_path, relative_path, description, category, lift_models, page
            FROM images
            WHERE description_tsv @@ plainto_tsquery('english', %s)
               OR extracted_text_tsv @@ plainto_tsquery('english', %s)
            ORDER BY
                ts_rank(description_tsv, plainto_tsquery('english', %s)) +
                ts_rank(extracted_text_tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s
        """, (query, query, query, query, limit))

    results = []
    for row in cur.fetchall():
        results.append({
            'filename': row['filename'],
            'path': row['file_path'],
            'description': row['description'],
            'category': row['category'],
            'lift_model': row['lift_models'][0] if row['lift_models'] else None
        })

    cur.close()
    conn.close()
    return results


def search(query: str, lift_model_hint: Optional[str] = None) -> SearchResponse:
    """
    Main search function - multi-stage pipeline.

    Order:
    1. Q&A pairs (fastest, highest confidence)
    2. Entities (specific lookups)
    3. Verified fixes (human-confirmed)
    4. Facts (structured knowledge)
    5. Chunks (fallback)
    """
    start_time = datetime.now()
    stages_used = []

    # Detect intent and model
    intent, detected_model = detect_query_intent(query)
    lift_model = lift_model_hint or detected_model

    all_results = []

    # Stage 1: Q&A pairs (skip for very short queries)
    if len(query.split()) >= 2:
        qa_results = search_qa_pairs(query, lift_model)
        if qa_results:
            all_results.extend(qa_results)
            stages_used.append('qa_pairs')

            # If we got a high-confidence Q&A match, that's often enough
            if qa_results[0].confidence >= 0.85:
                # Still search for supplementary info but with lower priority
                pass

    # Stage 2: Entities (especially for spec/part lookups)
    if intent in ['spec_lookup', 'part_lookup', 'wiring'] or len(all_results) < 2:
        entity_results = search_entities(query, lift_model)
        if entity_results:
            all_results.extend(entity_results)
            stages_used.append('entities')

    # Stage 3: Verified fixes (especially for troubleshooting)
    if intent == 'troubleshooting' or len(all_results) < 2:
        fix_results = search_verified_fixes(query, lift_model)
        if fix_results:
            all_results.extend(fix_results)
            stages_used.append('verified_fixes')

    # Stage 4: Facts
    if len(all_results) < 3:
        fact_results = search_facts(query, lift_model)
        if fact_results:
            all_results.extend(fact_results)
            stages_used.append('facts')

    # Stage 5: Chunks (fallback)
    if len(all_results) < 2:
        chunk_results = search_chunks(query, lift_model)
        if chunk_results:
            all_results.extend(chunk_results)
            stages_used.append('chunks')

    # Search for relevant images
    image_results = search_images(query, lift_model)

    # Sort by confidence
    all_results.sort(key=lambda x: x.confidence, reverse=True)

    # Deduplicate similar content
    seen_content = set()
    unique_results = []
    for result in all_results:
        content_key = result.content[:100].lower()
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_results.append(result)

    elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

    return SearchResponse(
        query=query,
        lift_model=lift_model,
        results=unique_results[:10],  # Top 10 results
        search_time_ms=elapsed_ms,
        search_stages_used=stages_used,
        has_images=len(image_results) > 0,
        relevant_images=image_results
    )


def log_query(user_id: str, user_name: str, query: str, response: SearchResponse):
    """Log query for analysis and learning."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO query_log (
            user_id, user_name, query_text,
            detected_lift_model, detected_intent,
            qa_matches, entity_matches, fact_matches, chunk_matches, verified_fix_matches,
            sources_used, response_time_ms
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        user_id,
        user_name,
        query,
        response.lift_model,
        detect_query_intent(query)[0],
        len([r for r in response.results if r.source_type == 'qa_pair']),
        len([r for r in response.results if r.source_type == 'entity']),
        len([r for r in response.results if r.source_type == 'fact']),
        len([r for r in response.results if r.source_type == 'chunk']),
        len([r for r in response.results if r.source_type == 'verified_fix']),
        None,  # sources_used JSON - can populate if needed
        response.search_time_ms
    ))

    conn.commit()
    cur.close()
    conn.close()


def format_context_for_claude(response: SearchResponse) -> str:
    """Format search results as context for Claude."""
    context_parts = []

    for result in response.results:
        source_info = []
        if result.source_document:
            source_info.append(f"Source: {result.source_document}")
        if result.source_page:
            source_info.append(f"Page {result.source_page}")
        if result.lift_model:
            source_info.append(f"Model: {result.lift_model}")

        source_str = ", ".join(source_info) if source_info else result.source_type

        if result.source_type == 'verified_fix':
            context_parts.append(f"[VERIFIED FIX - {source_str}]\n{result.content}")
        elif result.source_type == 'qa_pair':
            context_parts.append(f"[Q&A - {source_str}]\n{result.content}")
        else:
            context_parts.append(f"[{source_str}]\n{result.content}")

    if response.relevant_images:
        context_parts.append("\n[RELEVANT DIAGRAMS/IMAGES AVAILABLE]")
        for img in response.relevant_images:
            context_parts.append(f"- {img['filename']}: {img['description']}")

    return "\n\n---\n\n".join(context_parts)
