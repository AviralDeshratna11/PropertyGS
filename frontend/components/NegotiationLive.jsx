import { useEffect, useState } from 'react'
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

export default function NegotiationLive({sessionId}){
  const [rounds, setRounds] = useState([])
  const [connected, setConnected] = useState(false)

  useEffect(()=>{
    if(!sessionId) return
    const url = `${API_BASE}/negotiation/${sessionId}/stream`
    const es = new EventSource(url)
    es.onopen = () => setConnected(true)
    es.onmessage = (e) => {
      try{
        const data = JSON.parse(e.data)
        setRounds(prev => [...prev, data])
      }catch(err){ console.error('parse', err) }
    }
    es.onerror = (err) => { console.warn('SSE error', err); es.close(); setConnected(false) }
    return ()=> es.close()
  },[sessionId])

  return (
    <div className="mt-4">
      <div className={`text-sm ${connected? 'text-green-600':'text-red-600'}`}>{connected? 'Live':'Disconnected'}</div>
      <div className="mt-2 space-y-2 max-h-64 overflow-auto">
        {rounds.length===0 ? <div className="text-gray-500">No rounds yet</div> : rounds.map((r, i)=> (
          <div key={i} className="p-2 bg-gray-50 rounded border">Round {r.round_number}: {r.buyer_action} → {r.seller_action} • Bid {r.buyer_amount_usd} / Ask {r.seller_amount_usd}</div>
        ))}
      </div>
    </div>
  )
}
