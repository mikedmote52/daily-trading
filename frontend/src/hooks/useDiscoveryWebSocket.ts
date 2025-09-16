import { useEffect, useRef, useState, useCallback } from 'react'
import { toast } from 'react-hot-toast'

interface DiscoveryMessage {
  type: 'scan_started' | 'scan_complete' | 'scan_error' | 'connected' | 'pong'
  scan_id?: string
  data?: any
  error?: string
  message?: string
  timestamp: string
}

interface UseDiscoveryWebSocketReturn {
  isConnected: boolean
  lastMessage: DiscoveryMessage | null
  sendMessage: (message: any) => void
  startDiscovery: () => void
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
}

export const useDiscoveryWebSocket = (
  url?: string
): UseDiscoveryWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<DiscoveryMessage | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<
    'connecting' | 'connected' | 'disconnected' | 'error'
  >('disconnected')

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  const wsUrl = url || import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setConnectionStatus('connecting')

    try {
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('🚀 WebSocket connected to Discovery API')
        setIsConnected(true)
        setConnectionStatus('connected')
        reconnectAttempts.current = 0

        toast.success('Connected to Discovery System', {
          id: 'ws-connection',
          duration: 2000
        })
      }

      wsRef.current.onmessage = (event) => {
        try {
          const message: DiscoveryMessage = JSON.parse(event.data)
          setLastMessage(message)

          // Handle different message types
          switch (message.type) {
            case 'scan_started':
              toast.loading('Discovery scan in progress...', {
                id: message.scan_id,
                duration: Infinity
              })
              break

            case 'scan_complete':
              toast.success(
                `Discovery complete! Found ${message.data?.results?.length || 0} candidates`,
                {
                  id: message.scan_id,
                  duration: 5000
                }
              )
              break

            case 'scan_error':
              toast.error(`Discovery failed: ${message.error}`, {
                id: message.scan_id,
                duration: 5000
              })
              break

            case 'connected':
              console.log('✅ Discovery API connection confirmed')
              break
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      wsRef.current.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason)
        setIsConnected(false)
        setConnectionStatus('disconnected')

        toast.dismiss('ws-connection')

        // Attempt reconnection if not intentional close
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000)
          reconnectAttempts.current++

          toast.loading(`Reconnecting... (${reconnectAttempts.current}/${maxReconnectAttempts})`, {
            id: 'ws-reconnect'
          })

          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setConnectionStatus('error')
          toast.error('Connection lost - please refresh page', {
            id: 'ws-reconnect',
            duration: Infinity
          })
        }
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('error')
        toast.error('Connection error - retrying...', {
          id: 'ws-error',
          duration: 3000
        })
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionStatus('error')
    }
  }, [wsUrl])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }

    setIsConnected(false)
    setConnectionStatus('disconnected')
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, cannot send message')
      toast.error('Not connected - please wait for connection')
    }
  }, [])

  const startDiscovery = useCallback(async () => {
    try {
      // Make HTTP request to start discovery
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/discover`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      console.log('Discovery scan initiated:', result)

      toast.success('Discovery scan started!', {
        duration: 3000
      })

    } catch (error) {
      console.error('Failed to start discovery:', error)
      toast.error(`Failed to start discovery: ${error}`)
    }
  }, [])

  // Ping/Pong keepalive
  useEffect(() => {
    if (!isConnected) return

    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' })
    }, 30000) // Ping every 30 seconds

    return () => clearInterval(pingInterval)
  }, [isConnected, sendMessage])

  // Auto-connect on mount
  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    startDiscovery,
    connectionStatus
  }
}