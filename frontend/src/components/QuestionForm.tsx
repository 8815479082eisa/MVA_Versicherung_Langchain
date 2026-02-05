import React, { useState } from 'react';

/**
 * Props für die QuestionForm Komponente
 */
interface QuestionFormProps {
  onSubmit: (question: string, options: { shortAnswer?: boolean; structuredAnswer?: boolean }) => void;
  isLoading: boolean;
}

/**
 * Frageformular-Komponente
 * Ermöglicht es Benutzern, Fragen zu stellen und Optionen zu wählen
 */
const QuestionForm: React.FC<QuestionFormProps> = ({ onSubmit, isLoading }) => {
  const [question, setQuestion] = useState('');
  const [shortAnswer, setShortAnswer] = useState(false);
  const [structuredAnswer, setStructuredAnswer] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !isLoading) {
      onSubmit(question.trim(), {
        shortAnswer: shortAnswer,
        structuredAnswer: structuredAnswer,
      });
      // Formular zurücksetzen nach dem Absenden
      setQuestion('');
      setShortAnswer(false);
      setStructuredAnswer(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Frage stellen
      </h2>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label 
            htmlFor="question" 
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Ihre Frage
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={isLoading}
            rows={3}
            className="w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm 
                     focus:ring-2 focus:ring-primary-500 focus:border-primary-500
                     disabled:bg-gray-100 disabled:cursor-not-allowed
                     resize-none"
            placeholder="Stellen Sie hier Ihre Frage zu Produkten, Bedingungen oder Tarifen..."
            required
          />
        </div>

        <div className="mb-4 flex flex-wrap gap-4">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={shortAnswer}
              onChange={(e) => setShortAnswer(e.target.checked)}
              disabled={isLoading}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded 
                       focus:ring-primary-500 disabled:opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Antwort kurz halten
            </span>
          </label>

          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={structuredAnswer}
              onChange={(e) => setStructuredAnswer(e.target.checked)}
              disabled={isLoading}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded 
                       focus:ring-primary-500 disabled:opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Antwort als Aufzählung strukturieren
            </span>
          </label>
        </div>

        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          className="w-full bg-primary-600 text-white py-3 px-6 rounded-md 
                   font-medium hover:bg-primary-700 focus:outline-none 
                   focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
                   disabled:bg-gray-400 disabled:cursor-not-allowed
                   transition-colors duration-200"
        >
          {isLoading ? 'Wird geladen…' : 'Antwort generieren'}
        </button>
      </form>
    </div>
  );
};

export default QuestionForm;

