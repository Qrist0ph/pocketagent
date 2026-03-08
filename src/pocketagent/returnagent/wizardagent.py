
from typing import Literal, Dict, Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class ReturnAgent:
    def get_graph(self):
        """
        Returns a compiled LangGraph StateGraph for the return agent subgraph.
        Nodes: 'return_agent' (react agent), 'return_output' (output node)
        """
        from langgraph.graph import StateGraph, START, END
        from pocketagent import S
        graph = StateGraph(S)
        graph.add_node("return_agent", self.react_agent)
        graph.add_node("return_output", self.return_data_output_node)
        graph.add_edge(START, "return_agent")
        graph.add_edge("return_agent", "return_output")
        graph.add_edge("return_output", END)
        return graph.compile()
    def return_data_output_node(self, state):
        thread_id = state.get("thread_id", "demo")
        self.set_thread_id(thread_id)
        form_data = self._get_form(thread_id)
        answers = form_data.get("answers", {})
        required_fields = ["email", "order_number"]
        all_fields_filled = all(field in answers and answers[field] for field in required_fields)
        confirmed = form_data.get("confirmed", False)
        if all_fields_filled and confirmed:
            print("=== RETOUR-DATEN VOLLSTÄNDIG ===")
            print(f"Thread ID: {thread_id}")
            print(f"E-Mail: {answers.get('email')}")
            print(f"Bestellnummer: {answers.get('order_number')}")
            print("Status: Retoure wird bearbeitet")
            print("================================")
        else:
            print("=== RETOUR-DATEN UNVOLLSTÄNDIG ===")
            print(f"Thread ID: {thread_id}")
            print(f"E-Mail: {'✓' if 'email' in answers else '✗'} {answers.get('email', 'Nicht angegeben')}")
            print(f"Bestellnummer: {'✓' if 'order_number' in answers else '✗'} {answers.get('order_number', 'Nicht angegeben')}")
            print(f"Bestätigt: {'✓' if confirmed else '✗'}")
            print("Warten auf weitere Eingaben...")
            print("===================================")
        return state
    REQUIRED_FIELDS = ["email", "order_number"]
    
    SYSTEM_PROMPT = """Du bist ein Retouren-Agent für einen Online-Shop.
Ziel: Sammle dynamisch alle fehlenden Informationen für eine Retourenanfrage: {required_fields}.
Regeln:
- Stelle IMMER nur EINE Frage pro Nachricht.
- Entscheide selbst, welches Feld als nächstes gebraucht wird (dynamisch).
- Wenn der/die Nutzer:in direkt eine Antwort liefert, rufe ein Tool auf, z.B. save_answer(field, value).
- Prüfe bei 'email' mit validate_email(value). Wenn nicht OK: höflich nachbessern lassen.
- Wenn alle Felder vorhanden sind: fasse zusammen, frage nach Bestätigung (ja/nein).
- Bei Bestätigung: beende mit einer finalen Zusammenfassung und sage, dass die Retoure bearbeitet wird.
- Nutze missing_fields(thread_id) und get_answers(thread_id), um den Status zu sehen.
- WICHTIG: Keine mehrere Fragen auf einmal.
- Sei freundlich und hilfsbereit bei Retouren-Anfragen.
"""
    
    def __init__(self, llm=None, checkpointer=None, default_thread_id: str = "demo"):
        self.session_store: Dict[str, Dict[str, Any]] = {}
        self.checkpointer = checkpointer or MemorySaver()
        self.default_thread_id = default_thread_id  # Injected thread_id
        self.current_thread_id = default_thread_id  # Current session thread_id
        self.react_agent = create_react_agent(
            llm,
            tools=[self.save_answer, self.validate_email, self.missing_fields, self.get_answers, self.set_confirmed],
            prompt=self.SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )
    
    def set_thread_id(self, thread_id: str):
        """Set the current thread_id for this session."""
        self.current_thread_id = thread_id
    
    def set_confirmed(self, confirmed: bool = True, thread_id: str = None):
        """Set the confirmation status for the return request."""
        thread_id = thread_id or self.current_thread_id
        form = self._get_form(thread_id)
        form["confirmed"] = confirmed
        return f"Return request {'confirmed' if confirmed else 'unconfirmed'}."
    
    def _get_form(self, thread_id: str) -> Dict[str, Any]:
        """
        Hole oder initialisiere das Formular (Session-Status) für einen bestimmten Thread.
        
        Args:
            thread_id (str): Eindeutige ID für die Session/Conversation.
        
        Returns:
            dict: Dictionary mit bisherigen Antworten und Status.
        """
        return self.session_store.setdefault(thread_id, {"answers": {}, "confirmed": False})
    
    def save_answer(self, field: Literal["email", "order_number"], value: str, thread_id: str = None) -> str:
        """
        Speichert eine Antwort des Users in der Session.

        Args:
            field (Literal["email", "order_number"]): Das Feld, das beantwortet wird.
            value (str): Die Antwort des Users.
            thread_id (str): Eindeutige ID für die Session. Verwendet current_thread_id wenn None.

        Returns:
            str: Bestätigungsnachricht für den Agent.
        """
        thread_id = thread_id or self.current_thread_id
        form = self._get_form(thread_id)
        form["answers"][field] = value
        return f"Saved {field}."
    
    def validate_email(self, value: str) -> str:
        """
        Validiert das E-Mail-Feld.

        Args:
            value (str): User-Eingabe für die E-Mail-Adresse.
        
        Returns:
            str: "OK" wenn plausibel, sonst eine Fehlermeldung.
        """
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, value):
            return "OK"
        return "Bitte geben Sie eine gültige E-Mail-Adresse ein."
    
    def missing_fields(self, thread_id: str = None) -> str:
        """
        Liefert die aktuell noch unbeantworteten Felder zurück.

        Args:
            thread_id (str): Eindeutige ID für die Session. Verwendet current_thread_id wenn None.

        Returns:
            str: Komma-separierte Liste fehlender Felder oder "none".
        """
        thread_id = thread_id or self.current_thread_id
        form = self._get_form(thread_id)
        missing = [f for f in self.REQUIRED_FIELDS if f not in form["answers"]]
        return ",".join(missing) or "none"
    
    def get_answers(self, thread_id: str = None) -> str:
        """
        Liefert die aktuell gespeicherten Antworten zurück.

        Args:
            thread_id (str): Eindeutige ID für die Session. Verwendet current_thread_id wenn None.

        Returns:
            str: String-Darstellung aller gespeicherten Antworten.
        """
        thread_id = thread_id or self.current_thread_id
        return str(self._get_form(thread_id)["answers"])

# Create an instance for compatibility with existing imports
# agent = ReturnAgent()
