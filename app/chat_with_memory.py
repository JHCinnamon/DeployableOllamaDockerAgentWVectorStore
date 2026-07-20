import argparse
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field
from timescale_vector.client import uuid_from_time

from database.vector_store import VectorStore
from services.llm_factory import LLMFactory


class ChatAnswer(BaseModel):
    answer: str = Field(description="Assistant reply to the user")


SYSTEM_PROMPT = """
You are a highly inquisitive and personable AI assistant engaged in an ongoing conversation with the user.

CONVERSATION CONTEXT:
You have access to:
- The current user message
- A record of previous exchanges in this conversation (shown below in "MEMORY" section)

PERSONALITY & BEHAVIOR:
1. Be genuinely interested in the user - remember personal details they've shared
2. Ask thoughtful follow-up questions to learn more about them
3. Reference past statements to show continuity ("You mentioned earlier that...")
4. As the conversation progresses, become increasingly specific and personalized
5. If the user asks you to recall information, review the MEMORY section carefully and provide what you know
6. Don't apologize for not remembering things early in conversation - you're learning about them

INSTRUCTIONS FOR RESPONSE:
1. Answer the user's question directly and thoughtfully
2. Incorporate relevant memory when appropriate
3. Ask at least one clarifying or follow-up question to deepen understanding
4. Be warm, conversational, and show genuine curiosity
5. Keep responses concise but engaging
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with vector-based long-term memory."
    )
    parser.add_argument(
        "--conversation-id",
        default=str(uuid4()),
        help="Conversation ID used to group and retrieve memory turns.",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=10,
        help="How many relevant past turns to retrieve per reply.",
    )
    return parser.parse_args()


def _format_memory_for_context(df: pd.DataFrame) -> str:
    """Format memory as readable conversation history."""
    if df.empty:
        return "No previous context available."
    
    # Sort by created_at to maintain chronological order
    df_sorted = df.sort_values("created_at", na_position="last")
    
    lines = []
    for _, row in df_sorted.iterrows():
        role = row.get("role", "unknown")
        content = row.get("content", "")
        # Extract just the message part if it's prefixed with role
        if isinstance(content, str) and ": " in content:
            content = content.split(": ", 1)[1]
        lines.append(f"{role.upper()}: {content}")
    
    return "\n".join(lines)


def _get_comprehensive_memory(vec: VectorStore, conversation_id: str, user_text: str, memory_limit: int) -> pd.DataFrame:
    """Get both semantically relevant and recent memory turns."""
    # Get semantically similar turns
    semantic_results = vec.search(
        user_text,
        limit=memory_limit // 2,
        metadata_filter={"source": "conversation", "conversation_id": conversation_id},
    )
    
    # Get recent turns regardless of semantic similarity
    # This helps with chronological recall
    try:
        # Search for a common phrase to get all results, then filter by time
        all_results = vec.search(
            "conversation",
            limit=memory_limit * 3,
            metadata_filter={"source": "conversation", "conversation_id": conversation_id},
        )
        
        # Filter to last N hours
        if not all_results.empty and "created_at" in all_results.columns:
            recent_cutoff = datetime.now() - timedelta(hours=2)
            recent_results = all_results[
                pd.to_datetime(all_results["created_at"], errors="coerce") > recent_cutoff
            ].head(memory_limit // 2)
        else:
            recent_results = all_results.head(memory_limit // 2)
    except Exception:
        recent_results = pd.DataFrame()
    
    # Combine and deduplicate
    combined = pd.concat([semantic_results, recent_results], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["id"], keep="first")
    
    return combined.head(memory_limit)


def _store_turn(vec: VectorStore, conversation_id: str, role: str, text: str) -> None:
    now = datetime.now()
    contents = f"{role}: {text}"
    embedding = vec.get_embedding(contents)

    row = pd.DataFrame(
        [
            {
                "id": str(uuid_from_time(now)),
                "metadata": {
                    "source": "conversation",
                    "conversation_id": conversation_id,
                    "role": role,
                    "created_at": now.isoformat(),
                },
                "contents": contents,
                "embedding": embedding,
            }
        ]
    )
    vec.upsert(row)


def _generate_reply(vec: VectorStore, conversation_id: str, user_text: str, memory_limit: int) -> str:
    import logging
    logger = logging.getLogger(__name__)
    
    memory_df = _get_comprehensive_memory(vec, conversation_id, user_text, memory_limit)
    memory_context = _format_memory_for_context(memory_df)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"MEMORY (previous exchanges in this conversation):\n{memory_context}\n\nNEW MESSAGE FROM USER:\n{user_text}",
        },
    ]

    llm = LLMFactory("openai")
    try:
        response = llm.create_completion(response_model=ChatAnswer, messages=messages)
        result = response.answer if hasattr(response, 'answer') else str(response)
        if result and len(result.strip()) > 0:
            return result
    except Exception as e:
        logger.info(f"Structured response failed: {e}")
    
    # Fallback: create a raw OpenAI client (not instructor-wrapped)
    try:
        from openai import OpenAI
        from config.settings import get_settings
        
        settings = get_settings()
        raw_client = OpenAI(api_key=settings.openai.api_key, base_url=settings.openai.base_url)
        fallback_response = raw_client.chat.completions.create(
            model=settings.openai.default_model,
            messages=messages,
            temperature=settings.openai.temperature,
            max_tokens=500,
        )
        answer = fallback_response.choices[0].message.content
        if answer and len(answer.strip()) > 0:
            return answer
    except Exception as fallback_e:
        logger.info(f"Fallback failed: {fallback_e}")
    
    # Final fallback: return a generic response
    return "That's interesting! Tell me more about what you just shared."


def main() -> None:
    args = parse_args()
    vec = VectorStore()
    vec.create_tables()

    print(f"Conversation ID: {args.conversation_id}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        _store_turn(vec, args.conversation_id, "user", user_text)
        answer = _generate_reply(vec, args.conversation_id, user_text, args.memory_limit)
        _store_turn(vec, args.conversation_id, "assistant", answer)

        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()