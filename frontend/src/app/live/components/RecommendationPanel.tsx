"use client";

import React, { useState } from 'react';

interface RecommendationPanelProps {
  candidates?: any[];
  episodeId: string;
  operatorAction?: string;
}

export default function RecommendationPanel({ candidates, episodeId, operatorAction }: RecommendationPanelProps) {
  const [submitting, setSubmitting] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  
  if (!candidates || candidates.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm h-full flex flex-col">
        <h3 className="text-lg font-medium text-white mb-4">Recommendations</h3>
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          Generating recommendations...
        </div>
      </div>
    );
  }

  const handleFeedback = async (recId: string, action: string) => {
    setSubmitting(true);
    setFeedbackError(null);
    try {
      const res = await fetch(`/api/v1/recommendations/${recId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operator_id: 'op-123', // Hardcoded for demo
          action: action,
        }),
      });
      if (!res.ok) throw new Error('Feedback submission failed');
      // Success is handled optimistically or via websocket refresh
    } catch (err: any) {
      setFeedbackError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-sm overflow-hidden flex flex-col h-full">
      <div className="p-6 border-b border-gray-700 bg-gray-800/50">
        <h3 className="text-lg font-medium text-white">Recommended Actions</h3>
        <p className="text-sm text-gray-400 mt-1">Validated against safety rules</p>
      </div>

      <div className="p-6 space-y-6 overflow-y-auto flex-1">
        {feedbackError && (
          <div className="bg-red-900/30 border border-red-800 text-red-400 text-sm p-3 rounded">
            {feedbackError}
          </div>
        )}

        {candidates.map((rec, idx) => (
          <div key={idx} className="bg-gray-750 border border-gray-700 rounded-lg overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800/80">
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="text-white font-medium">{rec.variable_name}</h4>
                  <p className="text-sm text-gray-400 mt-0.5">
                    Proposed: <span className="text-white font-mono">{rec.proposed_value}</span>
                    {rec.clamped_value && (
                      <span className="ml-2 text-yellow-400 text-xs bg-yellow-900/30 px-2 py-0.5 rounded">
                        Clamped to {rec.clamped_value}
                      </span>
                    )}
                  </p>
                </div>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium ${
                  rec.safety_status === 'approved' ? 'bg-green-900/30 text-green-400 border border-green-800/50' : 
                  rec.safety_status === 'clamped' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-800/50' : 
                  'bg-red-900/30 text-red-400 border border-red-800/50'
                }`}>
                  {rec.safety_status}
                </span>
              </div>
            </div>
            
            <div className="p-4 bg-gray-800/40">
              <p className="text-sm text-gray-300">
                <span className="text-gray-500 font-medium">Expected Result:</span> {rec.expected_improvement}
              </p>
              
              {rec.historical_support && rec.historical_support.length > 0 && (
                <p className="text-xs text-gray-500 mt-2">
                  Based on history: {rec.historical_support.join(', ')}
                </p>
              )}
            </div>

            {/* Operator Actions - only show if status is not rejected and no action taken yet */}
            {rec.safety_status !== 'rejected' && !operatorAction && (
              <div className="p-3 bg-gray-700/30 border-t border-gray-700 flex gap-3">
                <button 
                  onClick={() => handleFeedback(rec.recommendation_id || 'mock-id', 'accepted')}
                  disabled={submitting}
                  className="flex-1 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                >
                  Apply
                </button>
                <button 
                  onClick={() => handleFeedback(rec.recommendation_id || 'mock-id', 'rejected')}
                  disabled={submitting}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                >
                  Dismiss
                </button>
              </div>
            )}
            
            {operatorAction && (
              <div className="p-3 bg-gray-800 border-t border-gray-700 text-center">
                <span className={`text-sm font-medium ${
                  operatorAction === 'accepted' ? 'text-green-400' : 'text-gray-400'
                }`}>
                  Operator action recorded: {operatorAction}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
