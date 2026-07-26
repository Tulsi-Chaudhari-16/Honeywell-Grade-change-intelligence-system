"use client";

import React from 'react';

interface ExplanationCardProps {
  explanation?: any;
}

export default function ExplanationCard({ explanation }: ExplanationCardProps) {
  if (!explanation) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex flex-col h-full">
        <h3 className="text-lg font-medium text-white mb-4">AI Explanation</h3>
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          Awaiting explanation...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex flex-col h-full">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          AI Explanation
        </h3>
      </div>

      <div className="space-y-5 text-sm">
        <div>
          <h4 className="text-gray-400 font-medium mb-1">Prediction Summary</h4>
          <p className="text-gray-200">{explanation.prediction_summary}</p>
        </div>

        <div>
          <h4 className="text-gray-400 font-medium mb-1">Root Cause Factor</h4>
          <p className="text-gray-200">{explanation.reason}</p>
        </div>

        <div>
          <h4 className="text-gray-400 font-medium mb-1">Evidence Cited</h4>
          <p className="text-gray-200">{explanation.evidence}</p>
        </div>

        <div>
          <h4 className="text-gray-400 font-medium mb-1">Historical Context</h4>
          <p className="text-gray-200">{explanation.historical_match}</p>
        </div>

        <div className="pt-4 border-t border-gray-700 grid grid-cols-2 gap-4">
          <div>
            <h4 className="text-xs text-gray-500 uppercase">Confidence</h4>
            <p className="text-gray-300 mt-1 font-medium">{explanation.confidence_statement}</p>
          </div>
          <div>
            <h4 className="text-xs text-gray-500 uppercase">Safety Check</h4>
            <p className="text-gray-300 mt-1 font-medium">{explanation.safety_check_status}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
