// frontend/lib/ws-client.ts
import { useEffect, useState, useCallback, useRef } from 'react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export interface EpisodeState {
  episode_id: string;
  machine_id: string;
  current_node: string;
  p_offspec: number;
  trajectory: number[];
  eta_stabilize_min: number;
  confidence: number;
  degraded_mode: boolean;
  explanation_card?: any;
  operator_action?: string;
  errors: string[];
}

export interface AlertMessage {
  alert_id: string;
  alert_type: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  episode_id?: string;
  ts: number;
}

export function useEpisodeWebSocket(episodeId: string | null) {
  const [state, setState] = useState<EpisodeState | null>(null);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!episodeId) return;

    const connect = () => {
      setStatus('connecting');
      const ws = new WebSocket(`${WS_URL}/ws/episodes/${episodeId}`);
      wsRef.current = ws;

      ws.onopen = () => setStatus('connected');
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setState(data);
        } catch (err) {
          console.error("Failed to parse websocket message", err);
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        // Simple reconnect logic
        setTimeout(connect, 3000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [episodeId]);

  return { state, status };
}

export function useAlertsWebSocket() {
  const [alerts, setAlerts] = useState<AlertMessage[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`${WS_URL}/ws/alerts`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const newAlert: AlertMessage = JSON.parse(event.data);
          setAlerts((prev) => [newAlert, ...prev].slice(0, 50)); // Keep last 50
        } catch (err) {
          console.error("Failed to parse alert message", err);
        }
      };

      ws.onclose = () => {
        setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return { alerts };
}
