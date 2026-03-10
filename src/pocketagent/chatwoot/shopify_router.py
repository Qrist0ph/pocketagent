"""
Shopify-focused router for Chatwoot email conversations.

Classifies incoming customer emails into:
- frustrated: Customer is upset or complains about missing response
- return_request: Customer wants to return an order
- refund_status: Customer asks about status of a pending refund/return
- spam: Unsolicited sales, marketing, or scam emails (not from real customers)
- ignore: Everything else (no action needed)
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("pocketagent.chatwoot.router")


SHOPIFY_ROUTER_PROMPT = """Klassifiziere die folgende E-Mail in genau eine Kategorie:

- 'frustrated': Ein echter Kunde ist veraergert, beschwert sich, droht, oder schreibt dass er noch keine Antwort erhalten hat.
- 'return_request': Ein echter Kunde moechte eine Retoure, Ruecksendung oder Umtausch NEU einleiten (hat noch keine Retoure gestartet).
- 'refund_status': Ein echter Kunde fragt nach dem Status einer bereits eingereichten Retoure oder Rueckerstattung (z.B. 'Gutschrift noch nicht erhalten', 'wann wird erstattet').
- 'spam': Unerwuenschte Verkaufsangebote, Werbung, Marketing, Lead-Generierung, SEO-Angebote, Kooperationsanfragen oder Betrug. Also alles was NICHT von einem echten Kunden kommt der etwas gekauft hat oder kaufen moechte.
- 'ignore': Alles andere (echte Kundenanfragen die keine besondere Aktion erfordern).

Antworte NUR mit dem Label: frustrated, return_request, refund_status, spam oder ignore."""


def classify_email(llm, content: str) -> str:
    """Classify a customer email using LLM with keyword fallback."""
    try:
        label = llm.invoke([
            SystemMessage(content=SHOPIFY_ROUTER_PROMPT),
            HumanMessage(content=content),
        ]).content.strip().lower()
        logger.debug("LLM raw classification: %s", label)
    except Exception as e:
        logger.warning("LLM classification failed, using keyword fallback: %s", e)
        label = _keyword_fallback(content)
        logger.debug("Keyword fallback result: %s", label)

    if "frustrated" in label:
        return "frustrated"
    elif "refund_status" in label:
        return "refund_status"
    elif "return" in label:
        return "return_request"
    elif "spam" in label:
        return "spam"
    return "ignore"


def _keyword_fallback(content: str) -> str:
    lower = content.lower()
    spam_keywords = [
        "mehr leads", "seo", "marketing", "wir helfen", "agentur",
        "kooperation", "zusammenarbeit", "angebot fuer sie",
        "kostenlose beratung", "umsatz steigern", "reichweite",
        "backlinks", "google ranking", "social media management",
        "influencer", "partnerschaft",
    ]
    frustrated_keywords = [
        "keine antwort", "immer noch nicht", "warte seit", "veraergert",
        "enttaeuscht", "unzufrieden", "beschwerde", "skandal", "frechheit",
        "anwalt", "bewertung", "nie wieder",
    ]
    refund_status_keywords = [
        "gutschrift", "noch nicht erhalten", "noch keine erstattung",
        "wann wird erstattet", "rueckerstattung status", "geld zurueck",
        "bereits retourniert", "bereits zurueckgeschickt",
    ]
    return_keywords = [
        "retoure", "retour", "ruecksendung", "rueckgabe", "umtausch",
        "zurueckschicken", "zuruecksenden", "return", "rueckerstattung",
    ]
    if any(kw in lower for kw in spam_keywords):
        return "spam"
    if any(kw in lower for kw in frustrated_keywords):
        return "frustrated"
    if any(kw in lower for kw in refund_status_keywords):
        return "refund_status"
    if any(kw in lower for kw in return_keywords):
        return "return_request"
    return "ignore"
