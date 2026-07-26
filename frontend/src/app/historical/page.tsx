"use client";

import React, { useState } from 'react';

export default function HistoricalExplorerPage() {
  const [gradePair, setGradePair] = useState('GRADE-A_GRADE-B');
  const [loading, setLoading] = useState(false);
  const [matches, setMatches] = useState<any[]>([]);

  const handleSearch = () => {
    setLoading(true);
    // Mock API call to Qdrant vector search
    setTimeout(() => {
      setMatches([
        {
          id: 'hist-1',
          score: 0.94,
          outcome: 'success',
          recovery_time_min: 32,
          actions: 'Increased machine_speed by 2%, decreased steam_pressure by 0.5 bar',
        },
        {
          id: 'hist-2',
          score: 0.88,
          outcome: 'success',
          recovery_time_min: 45,
          actions: 'Decreased headbox_consistency by 0.1%',
        },
        {
          id: 'hist-3',
          score: 0.81,
          outcome: 'offspec',
          recovery_time_min: 120,
          actions: 'Increased machine_speed by 5% (too fast, caused break)',
        }
      ]);
      setLoading(false);
    }, 800);
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Historical Similarity Explorer</h1>
          <p className="text-gray-400 mt-1">Search past transitions using vector embeddings to find similar operating conditions.</p>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex gap-4 items-end">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-400 mb-1">Grade Transition Pair</label>
          <input 
            type="text" 
            value={gradePair}
            onChange={(e) => setGradePair(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g. GRADE-A_GRADE-B"
          />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-400 mb-1">Machine Context (Optional)</label>
          <input 
            type="text" 
            className="w-full bg-gray-900 border border-gray-700 rounded-md py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g. PM1"
            defaultValue="PM1"
          />
        </div>
        <button 
          onClick={handleSearch}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 px-6 rounded-md transition-colors disabled:opacity-50 h-[42px]"
        >
          {loading ? 'Searching...' : 'Search Vector Space'}
        </button>
      </div>

      {matches.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-white">Top Matches</h3>
          
          <div className="grid grid-cols-1 gap-4">
            {matches.map((match, i) => (
              <div key={i} className="bg-gray-800 rounded-xl border border-gray-700 shadow-sm overflow-hidden flex">
                <div className={`w-2 ${match.outcome === 'success' ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <div className="p-6 flex-1">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h4 className="text-white font-medium text-lg">Episode {match.id}</h4>
                      <div className="flex gap-4 mt-1 text-sm text-gray-400">
                        <span className="flex items-center gap-1">
                          <span className={`w-2 h-2 rounded-full ${match.outcome === 'success' ? 'bg-green-500' : 'bg-red-500'}`}></span>
                          {match.outcome === 'success' ? 'Successful Transition' : 'Off-Spec Event'}
                        </span>
                        <span>•</span>
                        <span>Recovery: {match.recovery_time_min} min</span>
                      </div>
                    </div>
                    <div className="bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-700 text-center">
                      <p className="text-xs text-gray-500 font-medium uppercase">Similarity</p>
                      <p className="text-blue-400 font-bold font-mono">{Math.round(match.score * 100)}%</p>
                    </div>
                  </div>
                  
                  <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
                    <p className="text-sm font-medium text-gray-400 mb-1">Operator Actions Taken:</p>
                    <p className="text-gray-200">{match.actions}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
