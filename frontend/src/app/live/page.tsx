"use client";

import React from 'react';
import { useSearchParams } from 'next/navigation';
import { useEpisodeWebSocket } from '@/lib/ws-client';
import PredictionTimeline from './components/PredictionTimeline';
import RootCausePanel from './components/RootCausePanel';
import RecommendationPanel from './components/RecommendationPanel';
import ExplanationCard from './components/ExplanationCard';

export default function LiveProcessPage() {
  const searchParams = useSearchParams();
  const episodeId = searchParams.get('episode');
  const { state, status } = useEpisodeWebSocket(episodeId);

  if (!episodeId) {
    return (
      <div className="flex h-96 items-center justify-center text-gray-400">
        <p>Please select an active episode from the Overview page.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Live Process Console</h1>
          <p className="text-gray-400 mt-1">Episode: <span className="font-mono text-sm">{episodeId}</span></p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Status:</span>
            {status === 'connected' ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-900/50 text-green-400 border border-green-800">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>
                Connected
              </span>
            ) : status === 'connecting' ? (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-900/50 text-yellow-400 border border-yellow-800">
                Connecting...
              </span>
            ) : (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-900/50 text-red-400 border border-red-800">
                Disconnected
              </span>
            )}
          </div>
          <div className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-md text-sm">
            Node: <span className="text-blue-400 font-medium">{state?.current_node || 'Loading...'}</span>
          </div>
        </div>
      </div>

      {!state ? (
        <div className="flex h-64 items-center justify-center">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
            <p className="text-gray-400 mt-4">Waiting for episode state...</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Column */}
          <div className="lg:col-span-2 space-y-6">
            <PredictionTimeline trajectory={state.trajectory} pOffspec={state.p_offspec} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <RootCausePanel rootCause={state.root_cause_report} />
              <ExplanationCard explanation={state.explanation_card} />
            </div>
          </div>
          
          {/* Side Column */}
          <div className="space-y-6">
            <RecommendationPanel 
              candidates={state.validated_candidates} 
              episodeId={episodeId} 
              operatorAction={state.operator_action} 
            />
          </div>
        </div>
      )}
    </div>
  );
}
