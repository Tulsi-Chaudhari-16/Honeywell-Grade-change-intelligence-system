"use client";

import React from 'react';

interface RootCausePanelProps {
  rootCause?: {
    ranked_factors: any[];
    attribution_method: string;
    narrative: string;
  };
}

export default function RootCausePanel({ rootCause }: RootCausePanelProps) {
  if (!rootCause || !rootCause.ranked_factors || rootCause.ranked_factors.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex flex-col h-full">
        <h3 className="text-lg font-medium text-white mb-4">Root Cause Analysis</h3>
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          Awaiting RCA data...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex flex-col h-full">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium text-white">Root Cause Analysis</h3>
        <span className="text-xs text-gray-400 bg-gray-700 px-2 py-1 rounded">
          {rootCause.attribution_method}
        </span>
      </div>

      <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4 mb-6">
        <p className="text-sm text-blue-200 leading-relaxed italic">
          "{rootCause.narrative}"
        </p>
      </div>

      <div className="space-y-4">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Top Contributing Factors</h4>
        {rootCause.ranked_factors.map((factor, idx) => (
          <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-gray-700/30 border border-gray-700/50">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center text-xs font-bold text-gray-300">
                {factor.rank}
              </div>
              <div>
                <p className="text-sm font-medium text-white">{factor.feature_name}</p>
                <p className="text-xs text-gray-400 mt-0.5">Value: {factor.feature_value}</p>
              </div>
            </div>
            <div className="text-right">
              <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                factor.direction === 'too_high' ? 'bg-red-900/30 text-red-400' : 'bg-blue-900/30 text-blue-400'
              }`}>
                {factor.direction === 'too_high' ? '↑ High' : '↓ Low'}
              </span>
              <p className="text-xs text-gray-400 mt-1">SHAP: {factor.shap_value.toFixed(4)}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
