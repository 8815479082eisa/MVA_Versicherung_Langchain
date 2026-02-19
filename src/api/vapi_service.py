"""
Vapi.ai Integration Service

Dieser Service verwaltet die Integration mit der Vapi.ai API für AI-gestützte Telefonanrufe.
"""

import os
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
import logging

# Logger konfigurieren
logger = logging.getLogger(__name__)


class VapiService:
    """
    Service-Klasse für die Interaktion mit der Vapi.ai API
    """

    def __init__(self):
        """Initialisiert den Vapi Service mit API-Key aus Umgebungsvariablen"""
        self.api_key = os.getenv("VAPI_API_KEY")
        self.assistant_id = os.getenv("VAPI_ASSISTANT_ID")
        self.base_url = "https://api.vapi.ai"
        
        if not self.api_key:
            logger.warning("VAPI_API_KEY nicht gesetzt. Vapi Service wird nicht funktionieren.")
        
        if not self.assistant_id:
            logger.warning("VAPI_ASSISTANT_ID nicht gesetzt. Standardmäßiger Assistant wird verwendet.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _validate_phone_number(self, phone_number: str) -> bool:
        """
        Validiert das Telefonnummer-Format (E.164)
        
        Args:
            phone_number: Telefonnummer im Format +49123456789
            
        Returns:
            True wenn gültig, False sonst
        """
        if not phone_number:
            return False
        
        # E.164 Format: + gefolgt von 7-15 Ziffern
        if not phone_number.startswith("+"):
            return False
        
        # Nur Ziffern nach dem +
        digits = phone_number[1:]
        if not digits.isdigit():
            return False
        
        # Länge zwischen 7 und 15 Ziffern
        if len(digits) < 7 or len(digits) > 15:
            return False
        
        return True

    async def initiate_call(
        self,
        phone_number: str,
        customer_name: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initiiert einen ausgehenden Anruf über Vapi.ai
        
        Args:
            phone_number: Telefonnummer im E.164 Format (z.B. +49123456789)
            customer_name: Optional - Name des Kunden
            additional_context: Optional - Zusätzlicher Kontext für den Assistant
            
        Returns:
            Dict mit Call-Informationen inkl. call_id und status
            
        Raises:
            ValueError: Bei ungültiger Telefonnummer oder fehlenden Credentials
            httpx.HTTPError: Bei API-Fehlern
        """
        # Validierung
        if not self.api_key:
            raise ValueError("VAPI_API_KEY ist nicht konfiguriert")
        
        if not self._validate_phone_number(phone_number):
            raise ValueError(
                f"Ungültige Telefonnummer: {phone_number}. "
                "Bitte verwenden Sie das E.164 Format (z.B. +49123456789)"
            )

        # Payload für Vapi API
        payload = {
            "phoneNumberId": os.getenv("VAPI_PHONE_NUMBER_ID"),  # Ihre Vapi Telefonnummer
            "customer": {
                "number": phone_number,
            }
        }

        # Optional: Assistant ID hinzufügen
        if self.assistant_id:
            payload["assistantId"] = self.assistant_id

        # Optional: Kundenname hinzufügen
        if customer_name:
            payload["customer"]["name"] = customer_name

        # Optional: Zusätzlicher Kontext für den Assistant
        if additional_context:
            payload["assistantOverrides"] = {
                "variableValues": additional_context
            }

        logger.info(f"Initiiere Vapi-Anruf an {phone_number}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/call/phone",
                    headers=self.headers,
                    json=payload
                )
                
                # Fehlerbehandlung
                if response.status_code == 401:
                    raise ValueError("Ungültiger API-Key. Bitte VAPI_API_KEY überprüfen.")
                elif response.status_code == 400:
                    error_detail = response.json().get("message", "Ungültige Anfrage")
                    raise ValueError(f"Fehlerhafte Anfrage: {error_detail}")
                elif response.status_code >= 500:
                    raise httpx.HTTPError(
                        f"Vapi Server-Fehler (Status {response.status_code}). "
                        "Bitte versuchen Sie es später erneut."
                    )
                
                response.raise_for_status()
                
                call_data = response.json()
                
                logger.info(
                    f"Anruf erfolgreich initiiert. Call ID: {call_data.get('id')}"
                )
                
                return {
                    "call_id": call_data.get("id"),
                    "status": call_data.get("status", "initiated"),
                    "phone_number": phone_number,
                    "created_at": call_data.get("createdAt", datetime.utcnow().isoformat()),
                    "assistant_id": call_data.get("assistantId"),
                    "cost": call_data.get("cost"),
                    "metadata": call_data
                }

        except httpx.TimeoutException:
            logger.error("Timeout beim Verbinden mit Vapi API")
            raise httpx.HTTPError(
                "Verbindungs-Timeout. Bitte überprüfen Sie Ihre Internetverbindung."
            )
        except httpx.RequestError as e:
            logger.error(f"Netzwerkfehler bei Vapi-Anfrage: {e}")
            raise httpx.HTTPError(
                f"Netzwerkfehler: Konnte keine Verbindung zu Vapi.ai herstellen."
            )

    async def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """
        Ruft den Status eines bestehenden Anrufs ab
        
        Args:
            call_id: Die ID des Anrufs
            
        Returns:
            Dict mit aktuellem Call-Status
            
        Raises:
            ValueError: Bei fehlendem API-Key oder ungültiger Call-ID
            httpx.HTTPError: Bei API-Fehlern
        """
        if not self.api_key:
            raise ValueError("VAPI_API_KEY ist nicht konfiguriert")
        
        if not call_id:
            raise ValueError("Call ID ist erforderlich")

        logger.info(f"Rufe Status für Call ID {call_id} ab")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/call/{call_id}",
                    headers=self.headers
                )
                
                if response.status_code == 404:
                    raise ValueError(f"Anruf mit ID {call_id} nicht gefunden")
                
                response.raise_for_status()
                call_data = response.json()
                
                return {
                    "call_id": call_data.get("id"),
                    "status": call_data.get("status"),
                    "duration": call_data.get("duration"),
                    "cost": call_data.get("cost"),
                    "ended_reason": call_data.get("endedReason"),
                    "started_at": call_data.get("startedAt"),
                    "ended_at": call_data.get("endedAt"),
                    "metadata": call_data
                }

        except httpx.RequestError as e:
            logger.error(f"Fehler beim Abrufen des Call-Status: {e}")
            raise httpx.HTTPError("Konnte Call-Status nicht abrufen")

    async def end_call(self, call_id: str) -> Dict[str, Any]:
        """
        Beendet einen laufenden Anruf
        
        Args:
            call_id: Die ID des zu beendenden Anrufs
            
        Returns:
            Dict mit finalen Call-Informationen
            
        Raises:
            ValueError: Bei fehlendem API-Key oder ungültiger Call-ID
            httpx.HTTPError: Bei API-Fehlern
        """
        if not self.api_key:
            raise ValueError("VAPI_API_KEY ist nicht konfiguriert")
        
        if not call_id:
            raise ValueError("Call ID ist erforderlich")

        logger.info(f"Beende Anruf mit Call ID {call_id}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.patch(
                    f"{self.base_url}/call/{call_id}",
                    headers=self.headers,
                    json={"status": "ended"}
                )
                
                if response.status_code == 404:
                    raise ValueError(f"Anruf mit ID {call_id} nicht gefunden")
                
                response.raise_for_status()
                call_data = response.json()
                
                logger.info(f"Anruf {call_id} erfolgreich beendet")
                
                return {
                    "call_id": call_data.get("id"),
                    "status": call_data.get("status"),
                    "duration": call_data.get("duration"),
                    "cost": call_data.get("cost"),
                    "ended_at": call_data.get("endedAt"),
                    "metadata": call_data
                }

        except httpx.RequestError as e:
            logger.error(f"Fehler beim Beenden des Anrufs: {e}")
            raise httpx.HTTPError("Konnte Anruf nicht beenden")


# Singleton-Instanz
_vapi_service_instance: Optional[VapiService] = None


def get_vapi_service() -> VapiService:
    """
    Gibt die Singleton-Instanz des Vapi Service zurück
    
    Returns:
        VapiService Instanz
    """
    global _vapi_service_instance
    if _vapi_service_instance is None:
        _vapi_service_instance = VapiService()
    return _vapi_service_instance
