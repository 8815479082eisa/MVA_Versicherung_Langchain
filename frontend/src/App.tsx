/**
 * Baloise Dokumenten-Assistent (MVA Demo)
 * 
 * Einfache Web-UI für ein agentisches RAG-System zur Beantwortung von Fragen
 * zu Versicherungsdokumenten.
 * 
 * TODO: Backend-Integration
 * ========================
 * Die Mock-API in `src/api.ts` sollte später durch echte Backend-Endpoints ersetzt werden:
 * - POST /api/ask - Frage stellen und Antwort erhalten
 *   Body: { question: string, options?: { shortAnswer?: boolean, structuredAnswer?: boolean } }
 *   Response: { answerText: string, sources: Source[], answerId: string, createdAt: string }
 * 
 * - POST /api/feedback - Feedback zu einer Antwort senden
 *   Body: { answerId: string, useful: boolean }
 *   Response: { success: boolean }
 * 
 * Beispiel-Backend-Integration:
 * ```typescript
 * // In src/api.ts
 * export async function askQuestion(payload: AskQuestionPayload): Promise<AnswerResponse> {
 *   const response = await fetch('/api/ask', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify(payload)
 *   });
 *   if (!response.ok) throw new Error('Fehler bei der Anfrage');
 *   return response.json();
 * }
 * ```
 */

import React, { useState } from 'react';
import QuestionForm from './components/QuestionForm';
import AnswerView from './components/AnswerView';
import HistoryList, { HistoryEntry } from './components/HistoryList';
import CallAssistant from './components/CallAssistant';
import { askQuestion, AnswerResponse, Source } from './api';

/**
 * Hauptkomponente der Anwendung
 */
const App: React.FC = () => {
  // State für Tab-Auswahl
  const [activeTab, setActiveTab] = useState<'documents' | 'calls'>('documents');
  
  // State für aktuelle Frage
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  
  // State für aktuelle Antwort
  const [currentAnswer, setCurrentAnswer] = useState<AnswerResponse | null>(null);
  
  // State für Ladezustand
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // State für Fehlerzustand
  const [error, setError] = useState<string | null>(null);
  
  // State für Historie
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  /**
   * Behandelt das Absenden einer Frage
   */
  const handleQuestionSubmit = async (
    question: string,
    options: { shortAnswer?: boolean; structuredAnswer?: boolean }
  ) => {
    setCurrentQuestion(question);
    setError(null);
    setIsLoading(true);
    setCurrentAnswer(null);

    try {
      const response: AnswerResponse = await askQuestion({
        question,
        options,
      });

      setCurrentAnswer(response);

      // Füge zur Historie hinzu
      const historyEntry: HistoryEntry = {
        question,
        answerText: response.answerText,
        sources: response.sources,
        answerId: response.answerId,
        timestamp: new Date(),
      };

      setHistory((prevHistory) => [historyEntry, ...prevHistory]);
    } catch (err) {
      const errorMessage = err instanceof Error 
        ? err.message 
        : 'Fehler bei der Generierung der Antwort. Bitte versuchen Sie es erneut.';
      setError(errorMessage);
      console.error('Fehler beim Stellen der Frage:', err);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Behandelt die Auswahl eines Historie-Eintrags
   */
  const handleHistorySelect = (entry: HistoryEntry) => {
    setCurrentQuestion(entry.question);
    setCurrentAnswer({
      answerText: entry.answerText,
      sources: entry.sources,
      answerId: entry.answerId,
      createdAt: entry.timestamp.toISOString(),
    });
    setError(null);
    // Scroll zum Antwortbereich
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  /**
   * Behandelt Feedback zu einer Antwort
   */
  const handleFeedback = (answerId: string, useful: boolean) => {
    console.log(`Feedback für Antwort ${answerId}: ${useful ? 'hilfreich' : 'nicht hilfreich'}`);
    // Hier könnte später zusätzliche Logik hinzugefügt werden (z.B. Analytics)
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            MVA Versicherungs-Assistent
          </h1>
          <p className="text-gray-600">
            Dokumentenabfrage und AI-Telefonanrufe für Ihre Versicherungsanfragen.
          </p>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex -mb-px space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('documents')}
              className={`${
                activeTab === 'documents'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center transition-colors`}
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              Dokumenten-Assistent
            </button>
            <button
              onClick={() => setActiveTab('calls')}
              className={`${
                activeTab === 'calls'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center transition-colors`}
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                />
              </svg>
              AI Call Assistant
            </button>
          </nav>
        </div>
      </div>

      {/* Hauptinhalt */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Dokumenten-Assistent Tab */}
        {activeTab === 'documents' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Linke Spalte: Frageformular und Antwort */}
            <div className="lg:col-span-2 space-y-6">
            {/* Frageformular */}
            <QuestionForm onSubmit={handleQuestionSubmit} isLoading={isLoading} />

            {/* Ladeanzeige */}
            {isLoading && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-center">
                  <div className="animate-pulse space-y-4 w-full">
                    <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-4 bg-gray-200 rounded w-full"></div>
                    <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                    <div className="space-y-2 mt-6">
                      <div className="h-3 bg-gray-200 rounded w-1/4"></div>
                      <div className="h-20 bg-gray-200 rounded"></div>
                      <div className="h-20 bg-gray-200 rounded"></div>
                    </div>
                  </div>
                </div>
                <p className="text-center text-gray-500 mt-4">
                  Antwort wird generiert...
                </p>
              </div>
            )}

            {/* Fehleranzeige */}
            {error && !isLoading && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg
                      className="h-5 w-5 text-red-400"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">
                      Fehler
                    </h3>
                    <p className="mt-2 text-sm text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Antwortanzeige */}
            {currentAnswer && !isLoading && !error && (
              <AnswerView
                answerText={currentAnswer.answerText}
                sources={currentAnswer.sources}
                answerId={currentAnswer.answerId}
                onFeedback={handleFeedback}
              />
            )}

            {/* Platzhalter wenn keine Antwort */}
            {!currentAnswer && !isLoading && !error && (
              <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <p className="text-gray-500 text-lg">
                  Stellen Sie eine Frage, um eine Antwort zu erhalten.
                </p>
              </div>
            )}
          </div>

            {/* Rechte Spalte: Historie */}
            <div className="lg:col-span-1">
              {history.length > 0 && (
                <HistoryList 
                  history={history} 
                  onSelectEntry={handleHistorySelect}
                />
              )}
            </div>
          </div>
        )}

        {/* AI Call Assistant Tab */}
        {activeTab === 'calls' && (
          <div className="max-w-3xl mx-auto">
            <CallAssistant />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            Hinweis: Die Antworten ersetzen keine rechtliche Beratung.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;

