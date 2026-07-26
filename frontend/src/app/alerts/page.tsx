"use client";

import React, { useEffect, useState } from 'react';
import { useAlertsWebSocket, AlertMessage } from '@/lib/ws-client';

export default function AlertsPage() {
  const { alerts } = useAlertsWebSocket();
  const [history, setHistory] = useState<AlertMessage[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch historical alerts on mount
    fetch('/api/v1/alerts?acknowledged=false')
      .then(res => res.json())
      .then(data => {
        setHistory(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await fetch(`/api/v1/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operator_id: 'op-123' })
      });
      // Remove from lists
      setHistory(prev => prev.filter(a => a.alert_id !== alertId));
    } catch (err) {
      console.error(err);
    }
  };

  // Combine websocket live alerts and historical alerts
  const combinedAlerts = [...alerts, ...history].reduce((acc, curr) => {
    if (!acc.find(a => a.alert_id === curr.alert_id)) {
      acc.push(curr);
    }
    return acc;
  }, [] as AlertMessage[]).sort((a, b) => b.ts - a.ts);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Alerts</h1>
          <p className="text-gray-400 mt-1">Live and active alerts requiring attention.</p>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-400">Loading alerts...</div>
        ) : combinedAlerts.length === 0 ? (
          <div className="p-12 flex flex-col items-center justify-center text-gray-400">
            <svg className="w-12 h-12 mb-4 text-green-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-lg font-medium text-white">All Clear</p>
            <p className="text-sm">No active alerts requiring attention.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-700">
            {combinedAlerts.map((alert) => (
              <div key={alert.alert_id} className={`p-4 flex items-start justify-between hover:bg-gray-750 transition-colors ${
                alert.severity === 'critical' ? 'bg-red-900/10' : 
                alert.severity === 'warning' ? 'bg-yellow-900/10' : 'bg-blue-900/10'
              }`}>
                <div className="flex gap-4">
                  <div className={`mt-1 flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    alert.severity === 'critical' ? 'bg-red-900/30 text-red-400 border border-red-800' : 
                    alert.severity === 'warning' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-800' : 
                    'bg-blue-900/30 text-blue-400 border border-blue-800'
                  }`}>
                    {alert.severity === 'critical' ? '!' : alert.severity === 'warning' ? '⚠' : 'i'}
                  </div>
                  <div>
                    <h4 className="text-white font-medium">{alert.message}</h4>
                    <div className="flex gap-3 mt-1 text-xs font-medium text-gray-400">
                      <span className="uppercase tracking-wider">{alert.alert_type}</span>
                      <span>•</span>
                      <span>{new Date(alert.ts * 1000).toLocaleTimeString()}</span>
                      {alert.episode_id && (
                        <>
                          <span>•</span>
                          <span className="text-gray-500 font-mono">Ep: {alert.episode_id.substring(0,8)}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <button 
                  onClick={() => acknowledgeAlert(alert.alert_id)}
                  className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
                >
                  Acknowledge
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
