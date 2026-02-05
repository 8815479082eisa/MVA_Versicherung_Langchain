/**
 * API für die Web-UI - Verbindung zum FastAPI-Backend
 * 
 * Diese Datei stellt die Verbindung zwischen Frontend und Backend her.
 */

// Basis-URL für die API (kann über Umgebungsvariable überschrieben werden)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Typen für die API-Responses (müssen mit Backend-Modellen übereinstimmen)
export interface Source {
  documentId: string;
  documentTitle: string;
  page?: number;
  section?: string;
  snippet?: string;
}

export interface AnswerResponse {
  answer: string;
  sources: Source[];
  latencyMs?: number;
}

export interface AskQuestionPayload {
  question: string;
  options?: {
    shortAnswer?: boolean;
    structuredAnswer?: boolean;
  };
}

// Legacy-Interface für Kompatibilität mit bestehenden Komponenten
export interface LegacyAnswerResponse {
  answerText: string;
  sources: LegacySource[];
  answerId: string;
  createdAt: string;
}

export interface LegacySource {
  id: string;
  documentTitle: string;
  documentId: string;
  page: number;
  section?: string;
  snippet: string;
}

/**
 * Konvertiert neue Backend-Response zu Legacy-Format für Kompatibilität
 */
function convertToLegacyFormat(
  response: AnswerResponse,
  question: string
): LegacyAnswerResponse {
  return {
    answerText: response.answer,
    sources: response.sources.map((source, index) => ({
      id: `src${index + 1}`,
      documentTitle: source.documentTitle,
      documentId: source.documentId,
      page: source.page || 0,
      section: source.section,
      snippet: source.snippet || "",
    })),
    answerId: `ans_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    createdAt: new Date().toISOString(),
  };
}

/**
 * Stellt eine Frage an das Backend-RAG-System
 * 
 * @param payload - Frage und optionale Parameter
 * @returns Antwort mit Quellen und Metadaten
 */
export async function askQuestion(
  payload: AskQuestionPayload
): Promise<LegacyAnswerResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: payload.question,
        // options wird aktuell vom Backend noch nicht verwendet, aber für zukünftige Erweiterungen beibehalten
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage =
        errorData.detail ||
        `API-Fehler (${response.status}): ${response.statusText}`;
      throw new Error(errorMessage);
    }

    const data: AnswerResponse = await response.json();
    
    // Konvertiere zu Legacy-Format für Kompatibilität mit bestehenden Komponenten
    return convertToLegacyFormat(data, payload.question);
  } catch (error) {
    // Netzwerkfehler oder andere Fehler
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        "Verbindungsfehler: Backend nicht erreichbar. Bitte starten Sie das Backend auf Port 8000."
      );
    }
    
    if (error instanceof Error) {
      throw error;
    }
    
    throw new Error(
      "Fehler bei der Generierung der Antwort. Bitte versuchen Sie es erneut."
    );
  }
}

/**
 * Sendet Feedback zu einer Antwort an das Backend
 * 
 * @param answerId - ID der Antwort
 * @param useful - Ob die Antwort hilfreich war
 */
export async function sendFeedback(
  answerId: string,
  useful: boolean
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        answer_id: answerId,
        useful: useful,
      }),
    });

    if (!response.ok) {
      console.warn("Fehler beim Senden des Feedbacks:", response.statusText);
      // Feedback-Fehler sind nicht kritisch, daher kein Throw
    }
  } catch (error) {
    console.error("Fehler beim Senden des Feedbacks:", error);
    // Feedback-Fehler sind nicht kritisch, daher kein Throw
  }
}
