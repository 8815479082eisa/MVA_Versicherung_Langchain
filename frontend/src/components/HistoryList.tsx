import React from 'react';
import { LegacySource } from '../api';

/**
 * Eintrag in der Historie
 */
export interface HistoryEntry {
  question: string;
  answerText: string;
  sources: LegacySource[];
  answerId: string;
  timestamp: Date;
}

/**
 * Props für die HistoryList Komponente
 */
interface HistoryListProps {
  history: HistoryEntry[];
  onSelectEntry: (entry: HistoryEntry) => void;
}

/**
 * Komponente zur Anzeige der Fragehistorie
 */
const HistoryList: React.FC<HistoryListProps> = ({ history, onSelectEntry }) => {
  /**
   * Formatiert Zeitstempel in deutschem Format (z.B. "14:23 Uhr")
   */
  const formatTime = (date: Date): string => {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes} Uhr`;
  };

  /**
   * Kürzt Fragetext auf eine Zeile (max. 60 Zeichen)
   */
  const truncateQuestion = (question: string): string => {
    if (question.length <= 60) {
      return question;
    }
    return question.substring(0, 57) + '...';
  };

  if (history.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Letzte Fragen in dieser Sitzung
      </h2>
      
      <div className="space-y-2">
        {history.slice(0, 10).map((entry, index) => (
          <button
            key={entry.answerId}
            onClick={() => onSelectEntry(entry)}
            className="w-full text-left px-4 py-3 border border-gray-200 rounded-md 
                     hover:bg-gray-50 hover:border-primary-300
                     transition-colors duration-200"
          >
            <div className="flex justify-between items-start">
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-700 truncate">
                  {truncateQuestion(entry.question)}
                </p>
              </div>
              <div className="ml-3 flex-shrink-0">
                <span className="text-xs text-gray-500">
                  {formatTime(entry.timestamp)}
                </span>
              </div>
            </div>
          </button>
        ))}
      </div>

      {history.length > 10 && (
        <p className="mt-4 text-xs text-gray-500 text-center">
          Zeige die letzten 10 von {history.length} Fragen
        </p>
      )}
    </div>
  );
};

export default HistoryList;

