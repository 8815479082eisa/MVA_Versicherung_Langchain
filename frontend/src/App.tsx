import React, { useState } from "react";
import QuestionForm from "./components/QuestionForm";
import AnswerView from "./components/AnswerView";
import HistoryList, { HistoryEntry } from "./components/HistoryList";
import RepoExplorer from "./components/RepoExplorer";
import { askQuestion, LegacyAnswerResponse } from "./api";

type AppView = "assistant" | "repo";

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<AppView>("assistant");
  const [currentAnswer, setCurrentAnswer] = useState<LegacyAnswerResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const handleQuestionSubmit = async (
    question: string,
    options: { shortAnswer?: boolean; structuredAnswer?: boolean }
  ) => {
    setError(null);
    setIsLoading(true);
    setCurrentAnswer(null);

    try {
      const response = await askQuestion({
        question,
        options,
      });

      setCurrentAnswer(response);

      const historyEntry: HistoryEntry = {
        question,
        answerText: response.answerText,
        sources: response.sources,
        answerId: response.answerId,
        timestamp: new Date(),
      };

      setHistory((prevHistory) => [historyEntry, ...prevHistory]);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Fehler bei der Generierung der Antwort. Bitte versuchen Sie es erneut.";
      setError(errorMessage);
      console.error("Fehler beim Stellen der Frage:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleHistorySelect = (entry: HistoryEntry) => {
    setCurrentAnswer({
      answerText: entry.answerText,
      sources: entry.sources,
      answerId: entry.answerId,
      createdAt: entry.timestamp.toISOString(),
    });
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleFeedback = (answerId: string, useful: boolean) => {
    console.log(`Feedback fuer Antwort ${answerId}: ${useful ? "hilfreich" : "nicht hilfreich"}`);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Baloise Dokumenten-Assistent (MVA Demo)</h1>
          <p className="text-gray-600">Stellen Sie eine Frage zu Produkten, Bedingungen oder Tarifen.</p>
          <div className="mt-4 flex items-center gap-2">
            <button
              type="button"
              onClick={() => setActiveView("assistant")}
              className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                activeView === "assistant"
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
              }`}
            >
              Assistent
            </button>
            <button
              type="button"
              onClick={() => setActiveView("repo")}
              className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                activeView === "repo"
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
              }`}
            >
              Repo Map
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeView === "assistant" ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <QuestionForm onSubmit={handleQuestionSubmit} isLoading={isLoading} />

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
                  <p className="text-center text-gray-500 mt-4">Antwort wird generiert...</p>
                </div>
              )}

              {error && !isLoading && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800">Fehler</h3>
                      <p className="mt-2 text-sm text-red-700">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {currentAnswer && !isLoading && !error && (
                <AnswerView
                  answerText={currentAnswer.answerText}
                  sources={currentAnswer.sources}
                  answerId={currentAnswer.answerId}
                  onFeedback={handleFeedback}
                />
              )}

              {!currentAnswer && !isLoading && !error && (
                <div className="bg-white rounded-lg shadow-md p-12 text-center">
                  <p className="text-gray-500 text-lg">Stellen Sie eine Frage, um eine Antwort zu erhalten.</p>
                </div>
              )}
            </div>

            <div className="lg:col-span-1">
              {history.length > 0 && <HistoryList history={history} onSelectEntry={handleHistorySelect} />}
            </div>
          </div>
        ) : (
          <RepoExplorer />
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">Hinweis: Die Antworten ersetzen keine rechtliche Beratung.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
