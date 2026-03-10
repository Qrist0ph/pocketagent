"""
Actions triggered by the Shopify router.

- notify_discord: Send alert to Discord when customer is frustrated
- send_return_link: Reply with self-service portal link for returns
- resolve_conversation: Mark a Chatwoot conversation as resolved (e.g. for spam)
"""

import os
import logging
import httpx

logger = logging.getLogger("pocketagent.chatwoot.actions")


async def notify_discord(conversation_id: int, content: str, conversation_url: str = "", log_url: str = ""):
    """Send a notification to a Discord channel via webhook."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping Discord notification")
        return

    message = (
        f"**Kundenanfrage erfordert Aufmerksamkeit**\n"
        f"Konversation: {conversation_id}\n"
    )
    if conversation_url:
        message += f"Chatwoot: {conversation_url}\n"
    if log_url:
        message += f"Agent Log: {log_url}\n"
    message += f"---\n{content[:500]}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json={"content": message})
            if resp.status_code in (200, 204):
                logger.info("Discord notification sent for conversation %s", conversation_id)
            else:
                logger.error("Discord notification failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Error sending Discord notification: %s", e)


async def resolve_conversation(conversation_id: int, headers: dict):
    """Mark a Chatwoot conversation as resolved (used for spam/unwanted messages)."""
    chatwoot_host = os.environ.get("CHATWOOT_HOST", "localhost:3000")
    chatwoot_account = os.environ.get("CHATWOOT_ACCOUNT_ID", "2")
    url = f"http://{chatwoot_host}/api/v1/accounts/{chatwoot_account}/conversations/{conversation_id}/toggle_status"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json={"status": "resolved"})
            if resp.status_code == 200:
                logger.info("Conversation %s marked as resolved", conversation_id)
            else:
                logger.error("Failed to resolve conversation: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Error resolving conversation: %s", e)


async def send_return_link(conversation_id: int, headers: dict, log_url: str = ""):
    """Reply to a Chatwoot conversation with the self-service return portal link."""
    portal_url = os.environ.get("SELF_SERVICE_PORTAL_URL", "")
    if not portal_url:
        logger.warning("SELF_SERVICE_PORTAL_URL not set, skipping return link")
        return

    chatwoot_host = os.environ.get("CHATWOOT_HOST", "localhost:3000")
    chatwoot_account = os.environ.get("CHATWOOT_ACCOUNT_ID", "2")
    url = f"http://{chatwoot_host}/api/v1/accounts/{chatwoot_account}/conversations/{conversation_id}/messages"

    reply = (
        f"Vielen Dank fuer Ihre Nachricht.\n\n"
        f"Fuer Retouren und Ruecksendungen nutzen Sie bitte unser Self-Service Portal:\n"
        f"{portal_url}\n\n"
        f"Dort koennen Sie Ihre Retoure schnell und unkompliziert anmelden.\n"
        f"Bei weiteren Fragen stehen wir Ihnen gerne zur Verfuegung."
    )
    if log_url:
        reply += f"\n\n---\nAgent Log: {log_url}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json={"content": reply})
            if resp.status_code == 200:
                logger.info("Return link sent to conversation %s", conversation_id)
            else:
                logger.error("Failed to send return link: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Error sending return link: %s", e)
