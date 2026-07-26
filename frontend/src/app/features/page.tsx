"use client";

import React from 'react';

export default function FeatureImportancePage() {
  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Global Feature Importance</h1>
          <p className="text-gray-400 mt-1">Aggregated SHAP values across all historical off-spec events.</p>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex items-center justify-center h-96">
        <p className="text-gray-500 text-lg">Feature Importance Bar Chart (Coming Soon)</p>
      </div>
    </div>
  );
}
