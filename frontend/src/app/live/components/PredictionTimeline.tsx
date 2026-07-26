"use client";

import React from 'react';

interface PredictionTimelineProps {
  trajectory: number[];
  pOffspec: number;
}

export default function PredictionTimeline({ trajectory, pOffspec }: PredictionTimelineProps) {
  const isHighRisk = pOffspec > 0.8;
  const isMediumRisk = pOffspec > 0.5 && !isHighRisk;
  const statusColor = isHighRisk ? 'text-red-400' : isMediumRisk ? 'text-yellow-400' : 'text-green-400';
  const barColor = isHighRisk ? 'bg-red-500' : isMediumRisk ? 'bg-yellow-500' : 'bg-green-500';

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm flex flex-col h-64">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-medium text-white">Prediction Timeline</h3>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Current p_offspec:</span>
          <span className={`text-lg font-bold ${statusColor}`}>
            {Math.round(pOffspec * 100)}%
          </span>
        </div>
      </div>

      <div className="flex-1 flex items-end gap-2 mt-auto">
        {trajectory && trajectory.length > 0 ? (
          trajectory.map((val, idx) => (
            <div key={idx} className="flex-1 flex flex-col items-center gap-2 group">
              <span className="text-xs text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity">
                {Math.round(val * 100)}%
              </span>
              <div className="w-full bg-gray-700 rounded-t-sm overflow-hidden h-32 relative">
                <div 
                  className={`absolute bottom-0 left-0 w-full rounded-t-sm transition-all duration-500 ${barColor}`}
                  style={{ height: `${val * 100}%`, opacity: 0.5 + (idx / trajectory.length) * 0.5 }}
                ></div>
              </div>
              <span className="text-xs text-gray-400">t+{idx}</span>
            </div>
          ))
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500 text-sm">
            Waiting for prediction trajectory...
          </div>
        )}
      </div>
    </div>
  );
}
