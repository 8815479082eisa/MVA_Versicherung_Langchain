/**
 * CallAssistant - AI-gestützte Telefonanruf-Komponente
 * 
 * Ermöglicht das Initiieren, Verfolgen und Beenden von AI-Anrufen über Vapi.ai
 */

import React, { useState, useEffect } from 'react';
import { initiateCall, getCallStatus, endCall, CallStatusResponse } from '../api';

/**
 * Status-Badge mit Farben je nach Call-Status
 */
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'initiated':
      case 'queued':
        return 'bg-blue-100 text-blue-800';
      case 'ringing':
        return 'bg-yellow-100 text-yellow-800';
      case 'in-progress':
      case 'in_progress':
        return 'bg-green-100 text-green-800';
      case 'ended':
        return 'bg-gray-100 text-gray-800';
      case 'failed':
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusText = (status: string) => {
    switch (status.toLowerCase()) {
      case 'initiated':
        return 'Initiiert';
      case 'queued':
        return 'In Warteschlange';
      case 'ringing':
        return 'Klingelt';
      case 'in-progress':
      case 'in_progress':
        return 'Aktiv';
      case 'ended':
        return 'Beendet';
      case 'failed':
        return 'Fehlgeschlagen';
      default:
        return status;
    }
  };

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(
        status
      )}`}
    >
      <span className="w-2 h-2 mr-2 rounded-full bg-current animate-pulse"></span>
      {getStatusText(status)}
    </span>
  );
};

/**
 * Hauptkomponente für AI Call Assistant
 */
const CallAssistant: React.FC = () => {
  // State Management
  const [phoneNumber, setPhoneNumber] = useState<string>('');
  const [customerName, setCustomerName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [callStatus, setCallStatus] = useState<CallStatusResponse | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);

  /**
   * Validiert Telefonnummer im E.164 Format
   */
  const validatePhoneNumber = (number: string): boolean => {
    // E.164 Format: + gefolgt von 7-15 Ziffern
    const e164Regex = /^\+[1-9]\d{6,14}$/;
    return e164Regex.test(number);
  };

  /**
   * Formatiert Telefonnummer während der Eingabe
   */
  const handlePhoneNumberChange = (value: string) => {
    // Nur Zahlen und + erlauben
    const cleaned = value.replace(/[^\d+]/g, '');
    setPhoneNumber(cleaned);
    setError(null);
  };

  /**
   * Startet einen AI-Anruf
   */
  const handleStartCall = async () => {
    // Validierung
    if (!phoneNumber.trim()) {
      setError('Bitte geben Sie eine Telefonnummer ein.');
      return;
    }

    if (!validatePhoneNumber(phoneNumber)) {
      setError(
        'Ungültige Telefonnummer. Bitte verwenden Sie das internationale Format (z.B. +491234567890).'
      );
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await initiateCall(
        phoneNumber,
        customerName || undefined
      );

      setCallStatus(result);
      setIsPolling(true);
      console.log('Anruf erfolgreich initiiert:', result);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Fehler beim Starten des Anrufs. Bitte versuchen Sie es erneut.';
      setError(errorMessage);
      console.error('Fehler beim Initiieren des Anrufs:', err);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Beendet einen laufenden Anruf
   */
  const handleEndCall = async () => {
    if (!callStatus?.callId) return;

    setIsLoading(true);
    setError(null);

    try {
      await endCall(callStatus.callId);
      
      // Status aktualisieren
      const updatedStatus = await getCallStatus(callStatus.callId);
      setCallStatus(updatedStatus);
      setIsPolling(false);
      
      console.log('Anruf erfolgreich beendet');
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Fehler beim Beenden des Anrufs.';
      setError(errorMessage);
      console.error('Fehler beim Beenden des Anrufs:', err);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Setzt den Status zurück für einen neuen Anruf
   */
  const handleNewCall = () => {
    setCallStatus(null);
    setPhoneNumber('');
    setCustomerName('');
    setError(null);
    setIsPolling(false);
  };

  /**
   * Polling für Status-Updates
   */
  useEffect(() => {
    if (!isPolling || !callStatus?.callId) return;

    const pollInterval = setInterval(async () => {
      try {
        const updatedStatus = await getCallStatus(callStatus.callId);
        setCallStatus(updatedStatus);

        // Stoppe Polling wenn Anruf beendet
        if (updatedStatus.status === 'ended' || updatedStatus.status === 'failed') {
          setIsPolling(false);
        }
      } catch (err) {
        console.error('Fehler beim Abrufen des Status:', err);
        // Bei wiederholten Fehlern Polling stoppen
        setIsPolling(false);
      }
    }, 3000); // Poll alle 3 Sekunden

    return () => clearInterval(pollInterval);
  }, [isPolling, callStatus?.callId]);

  /**
   * Formatiert Dauer in Minuten:Sekunden
   */
  const formatDuration = (seconds?: number): string => {
    if (!seconds) return '-';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  /**
   * Formatiert Kosten
   */
  const formatCost = (cost?: number): string => {
    if (cost === undefined || cost === null) return '-';
    return `$${cost.toFixed(4)}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center">
          <svg
            className="w-6 h-6 mr-2 text-blue-600"
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
        </h2>
        <p className="text-gray-600">
          Starten Sie einen AI-gestützten Telefonanruf mit dem MVA Versicherungs-Assistenten.
        </p>
      </div>

      {/* Fehleranzeige */}
      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
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
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Call Status Display */}
      {callStatus && (
        <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Status:</span>
              <StatusBadge status={callStatus.status} />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">
                Telefonnummer:
              </span>
              <span className="text-sm text-gray-900 font-mono">
                {callStatus.phoneNumber}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Call ID:</span>
              <span className="text-xs text-gray-600 font-mono">
                {callStatus.callId}
              </span>
            </div>

            {callStatus.duration !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Dauer:</span>
                <span className="text-sm text-gray-900">
                  {formatDuration(callStatus.duration)}
                </span>
              </div>
            )}

            {callStatus.cost !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Kosten:</span>
                <span className="text-sm text-gray-900">
                  {formatCost(callStatus.cost)}
                </span>
              </div>
            )}

            {callStatus.endedReason && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">
                  Grund:
                </span>
                <span className="text-sm text-gray-900">
                  {callStatus.endedReason}
                </span>
              </div>
            )}
          </div>

          {/* Call Actions */}
          <div className="mt-4 flex gap-2">
            {callStatus.status !== 'ended' && callStatus.status !== 'failed' && (
              <button
                onClick={handleEndCall}
                disabled={isLoading}
                className="flex-1 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {isLoading ? 'Wird beendet...' : 'Anruf beenden'}
              </button>
            )}

            {(callStatus.status === 'ended' || callStatus.status === 'failed') && (
              <button
                onClick={handleNewCall}
                className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Neuer Anruf
              </button>
            )}
          </div>
        </div>
      )}

      {/* Call Form */}
      {!callStatus && (
        <div className="space-y-4">
          {/* Telefonnummer */}
          <div>
            <label
              htmlFor="phoneNumber"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Telefonnummer <span className="text-red-500">*</span>
            </label>
            <input
              type="tel"
              id="phoneNumber"
              value={phoneNumber}
              onChange={(e) => handlePhoneNumberChange(e.target.value)}
              placeholder="+491234567890"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono"
              disabled={isLoading}
            />
            <p className="mt-1 text-xs text-gray-500">
              Verwenden Sie das internationale Format (z.B. +49 für Deutschland)
            </p>
          </div>

          {/* Kundenname (Optional) */}
          <div>
            <label
              htmlFor="customerName"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Kundenname (Optional)
            </label>
            <input
              type="text"
              id="customerName"
              value={customerName}
              onChange={(e) => setCustomerName(e.target.value)}
              placeholder="Max Mustermann"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
          </div>

          {/* Start Button */}
          <button
            onClick={handleStartCall}
            disabled={isLoading || !phoneNumber}
            className="w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium flex items-center justify-center"
          >
            {isLoading ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Anruf wird initiiert...
              </>
            ) : (
              <>
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
                AI-Anruf starten
              </>
            )}
          </button>
        </div>
      )}

      {/* Info Box */}
      <div className="mt-6 bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">
          ℹ️ Hinweise:
        </h3>
        <ul className="text-xs text-gray-600 space-y-1">
          <li>• Der AI-Assistent spricht Deutsch und kann Versicherungsfragen beantworten</li>
          <li>• Die Telefonnummer muss im E.164 Format sein (+49...)</li>
          <li>• Der Angerufene erhält einen Anruf von Ihrer konfigurierten Vapi-Nummer</li>
          <li>• Der Status wird automatisch aktualisiert während des Anrufs</li>
        </ul>
      </div>
    </div>
  );
};

export default CallAssistant;
