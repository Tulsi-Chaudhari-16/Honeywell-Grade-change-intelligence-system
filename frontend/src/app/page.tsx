"use client";

import React, { useEffect, useState } from 'react';
import Link from 'next/link';

export default function OverviewPage() {
  const [activeEpisodes, setActiveEpisodes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In a real app, this would fetch from /api/v1/episodes/active
    // For now, we mock it.
    setTimeout(() => {
      setActiveEpisodes([
        { episode_id: 'e1-uuid', machine: 'PM1', state: 'FeatureBuilding', p_offspec: 0.12 },
        { episode_id: 'e2-uuid', machine: 'PM2', state: 'AwaitingOperator', p_offspec: 0.88 },
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <h1 className="text-3xl font-bold tracking-tight">Fleet Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* KPI Cards */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm">
          <p className="text-gray-400 text-sm font-medium">Active Grade Changes</p>
          <p className="text-4xl font-bold mt-2 text-white">{activeEpisodes.length}</p>
        </div>
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm">
          <p className="text-gray-400 text-sm font-medium">Avg Transition Time</p>
          <p className="text-4xl font-bold mt-2 text-white">42m</p>
          <p className="text-green-400 text-xs mt-2 font-medium">↓ 12% vs last month</p>
        </div>
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-sm">
          <p className="text-gray-400 text-sm font-medium">Off-Spec Events</p>
          <p className="text-4xl font-bold mt-2 text-white">1</p>
          <p className="text-yellow-400 text-xs mt-2 font-medium">Action Required</p>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-sm overflow-hidden mt-8">
        <div className="px-6 py-5 border-b border-gray-700 flex justify-between items-center">
          <h3 className="text-lg font-medium text-white">Live Grade Transitions</h3>
          <button className="text-sm bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded transition-colors font-medium">
            Start Simulation
          </button>
        </div>
        
        {loading ? (
          <div className="p-8 text-center text-gray-400">Loading active episodes...</div>
        ) : (
          <table className="min-w-full divide-y divide-gray-700">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Machine</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">State</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Risk Level</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">Action</th>
              </tr>
            </thead>
            <tbody className="bg-gray-800 divide-y divide-gray-700">
              {activeEpisodes.map((ep, idx) => (
                <tr key={idx} className="hover:bg-gray-750 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                    {ep.machine}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-900/50 text-blue-300 border border-blue-800">
                      {ep.state}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {ep.p_offspec > 0.8 ? (
                      <span className="text-red-400 font-medium">High Risk ({Math.round(ep.p_offspec * 100)}%)</span>
                    ) : ep.p_offspec > 0.5 ? (
                      <span className="text-yellow-400 font-medium">Medium ({Math.round(ep.p_offspec * 100)}%)</span>
                    ) : (
                      <span className="text-green-400 font-medium">Low ({Math.round(ep.p_offspec * 100)}%)</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <Link href={`/live?episode=${ep.episode_id}`} className="text-blue-400 hover:text-blue-300">
                      View Console →
                    </Link>
                  </td>
                </tr>
              ))}
              {activeEpisodes.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-6 py-8 text-center text-gray-400 text-sm">
                    No active grade transitions.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
