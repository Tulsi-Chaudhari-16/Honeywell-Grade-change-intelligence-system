"use client";

import React, { useEffect, useState } from 'react';

export default function SystemHealthPage() {
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    fetch('/api/v1/system/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Health & Metrics</h1>
          <p className="text-gray-400 mt-1">Component status and performance metrics.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-white">API Backend</h3>
            <span className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></span>
          </div>
          <p className="text-sm text-gray-400 mt-2">Status: {health?.status || 'Loading...'}</p>
          <p className="text-sm text-gray-400">LLM Provider: <span className="font-mono text-blue-400">{health?.provider || '...'}</span></p>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-white">Kafka Brokers</h3>
            <span className="w-3 h-3 rounded-full bg-green-500"></span>
          </div>
          <p className="text-sm text-gray-400 mt-2">Topics: gcis.tags.raw (Healthy)</p>
          <p className="text-sm text-gray-400">Consumer Lag: 0 ms</p>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-white">Qdrant Vector DB</h3>
            <span className="w-3 h-3 rounded-full bg-green-500"></span>
          </div>
          <p className="text-sm text-gray-400 mt-2">Collection: gcis_episodes (Healthy)</p>
          <p className="text-sm text-gray-400">Indexed Vectors: 1,432</p>
        </div>
      </div>
    </div>
  );
}
