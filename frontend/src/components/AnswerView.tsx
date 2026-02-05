import React, { useState } from 'react';
import { LegacySource, sendFeedback } from '../api';

/**
 * Props für die AnswerView Komponente
 */
interface AnswerViewProps {
  answerText: string;
  sources: LegacySource[];
  answerId: string;
  onFeedback?: (answerId: string, useful: boolean) => void;
}

/**
 * Komponente zur Anzeige der Antwort und Quellen
 */
const AnswerView: React.FC<AnswerViewProps> = ({ 
  answerText, 
  sources, 
  answerId,
  onFeedback 
}) => {
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [feedbackGiven, setFeedbackGiven] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);

  /**
   * Quellen-Ausschnitt ein-/ausklappen
   */
  const toggleSource = (sourceId: string) => {
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(sourceId)) {
      newExpanded.delete(sourceId);
    } else {
      newExpanded.add(sourceId);
    }
    setExpandedSources(newExpanded);
  };

  /**
   * Antwort in die Zwischenablage kopieren
   */
  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(answerText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Fehler beim Kopieren:', err);
    }
  };

  /**
   * Antwort mit Quellen in die Zwischenablage kopieren
   */
  const copyAnswerWithSources = async () => {
    const sourcesText = sources.map(source => {
      const section = source.section ? `, Abschnitt ${source.section}` : '';
      return `- ${source.documentTitle} (${source.documentId}) – Seite ${source.page}${section}`;
    }).join('\n');

    const fullText = `${answerText}\n\nVerwendete Quellen:\n${sourcesText}`;
    
    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Fehler beim Kopieren:', err);
    }
  };

  /**
   * Feedback senden
   */
  const handleFeedback = async (useful: boolean) => {
    if (feedbackGiven !== null) return; // Feedback bereits gegeben
    
    setFeedbackGiven(useful);
    
    try {
      await sendFeedback(answerId, useful);
      if (onFeedback) {
        onFeedback(answerId, useful);
      }
    } catch (err) {
      console.error('Fehler beim Senden des Feedbacks:', err);
      setFeedbackGiven(null); // Zurücksetzen bei Fehler
    }
  };

  // Formatierung des Antworttextes (Zeilenumbrüche werden als Absätze dargestellt)
  const formatAnswer = () => {
    const lines = answerText.split('\n');
    const elements: JSX.Element[] = [];
    let currentList: string[] = [];
    let keyCounter = 0;

    const flushList = () => {
      if (currentList.length > 0) {
        elements.push(
          <ul key={keyCounter++} className="list-disc list-inside space-y-1 my-3 ml-4">
            {currentList.map((item, idx) => (
              <li key={idx} className="text-gray-700">
                {item.replace(/^[•\-\d.\s]+/, '').trim()}
              </li>
            ))}
          </ul>
        );
        currentList = [];
      }
    };

    lines.forEach((line, index) => {
      const trimmedLine = line.trim();
      
      // Prüfe ob die Zeile eine Aufzählung ist (beginnt mit •, -, oder Nummer)
      if (trimmedLine && /^[•\-\d]/.test(trimmedLine)) {
        currentList.push(trimmedLine);
      } else {
        // Flushe die aktuelle Liste, wenn vorhanden
        flushList();
        
        // Normale Absätze
        if (trimmedLine) {
          elements.push(
            <p key={keyCounter++} className="text-gray-700 leading-relaxed mb-3">
              {trimmedLine}
            </p>
          );
        } else if (index < lines.length - 1) {
          // Leere Zeilen als Abstand
          elements.push(<div key={keyCounter++} className="mb-2"></div>);
        }
      }
    });

    // Flushe verbleibende Liste
    flushList();

    return elements;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-xl font-semibold text-gray-800">
          Antwort
        </h2>
        <div className="flex gap-2">
          <button
            onClick={copyAnswer}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 
                     border border-gray-300 rounded hover:bg-gray-50
                     transition-colors duration-200"
            title="Antwort kopieren"
          >
            {copied ? '✓ Kopiert' : 'Antwort kopieren'}
          </button>
          <button
            onClick={copyAnswerWithSources}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 
                     border border-gray-300 rounded hover:bg-gray-50
                     transition-colors duration-200"
            title="Antwort mit Quellen kopieren"
          >
            {copied ? '✓ Kopiert' : 'Mit Quellen kopieren'}
          </button>
        </div>
      </div>

      <div className="prose max-w-none mb-6">
        {formatAnswer()}
      </div>

      {sources && sources.length > 0 && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Verwendete Quellen
          </h3>
          <div className="space-y-3">
            {sources.map((source) => {
              const section = source.section ? `, Abschnitt ${source.section}` : '';
              const sourceText = `${source.documentTitle} (${source.documentId}) – Seite ${source.page}${section}`;
              const isExpanded = expandedSources.has(source.id);

              return (
                <div 
                  key={source.id}
                  className="border border-gray-200 rounded-md overflow-hidden"
                >
                  <button
                    onClick={() => toggleSource(source.id)}
                    className="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 
                             text-left flex justify-between items-center
                             transition-colors duration-200"
                  >
                    <span className="text-sm font-medium text-gray-700">
                      {sourceText}
                    </span>
                    <span className="text-gray-500">
                      {isExpanded ? '▼' : '▶'}
                    </span>
                  </button>
                  {isExpanded && (
                    <div className="px-4 py-3 bg-white border-t border-gray-200">
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {source.snippet}
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-sm text-gray-500 mb-4">
          Bitte prüfen Sie die Antwort bei Bedarf im Originaldokument nach.
        </p>

        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-700">War diese Antwort hilfreich?</span>
          <div className="flex gap-2">
            <button
              onClick={() => handleFeedback(true)}
              disabled={feedbackGiven !== null}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200
                ${feedbackGiven === true 
                  ? 'bg-green-100 text-green-700 cursor-default' 
                  : feedbackGiven === false
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-gray-100 text-gray-700 hover:bg-green-50 hover:text-green-700'
                }`}
            >
              👍 Antwort war hilfreich
            </button>
            <button
              onClick={() => handleFeedback(false)}
              disabled={feedbackGiven !== null}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200
                ${feedbackGiven === false 
                  ? 'bg-red-100 text-red-700 cursor-default' 
                  : feedbackGiven === true
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-gray-100 text-gray-700 hover:bg-red-50 hover:text-red-700'
                }`}
            >
              👎 Antwort war nicht hilfreich
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnswerView;

