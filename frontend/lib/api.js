const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

export async function listProperties(){
  const res = await fetch(`${API_BASE}/properties/`)
  if(!res.ok) throw new Error('Failed to fetch properties')
  return res.json()
}

export async function getProperty(id){
  const res = await fetch(`${API_BASE}/properties/${id}`)
  if(!res.ok) throw new Error('Property not found')
  return res.json()
}

export async function lifestyleSearch(payload){
  const res = await fetch(`${API_BASE}/search/lifestyle`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
  })
  return res.json()
}

export async function startNegotiation(payload){
  const res = await fetch(`${API_BASE}/negotiation/start`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
  })
  return res.json()
}

export async function negotiationRound(session_id){
  const res = await fetch(`${API_BASE}/negotiation/${session_id}/round`, {method: 'POST'})
  return res.json()
}

export async function getNegotiationStatus(session_id){
  const res = await fetch(`${API_BASE}/negotiation/${session_id}/status`)
  return res.json()
}

export async function runInspection(property_id){
  const res = await fetch(`${API_BASE}/inspection/scan?property_id=${property_id}`, {method: 'POST'})
  return res.json()
}

export async function requestPassport(buyer_id, threshold_usd){
  const res = await fetch(`${API_BASE}/zkp/passport/request`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({buyer_id, threshold_usd})
  })
  if(!res.ok) throw new Error('Failed to request passport')
  return res.json()
}

export async function verifyPassport(token){
  const res = await fetch(`${API_BASE}/zkp/passport/verify`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({token})
  })
  if(!res.ok) throw new Error('Failed to verify passport')
  return res.json()
}

export async function createVoiceSession(property_id, language='en', gender='neutral'){
  const res = await fetch(`${API_BASE}/voice/sessions`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({property_id, language, gender})
  })
  return res.json()
}

export async function voiceText(session_id, text){
  const res = await fetch(`${API_BASE}/voice/${session_id}/text`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text})
  })
  return res.json()
}

export async function voiceAudio(session_id, audioBlob){
  const fd = new FormData();
  fd.append('file', audioBlob, 'input.webm')
  const res = await fetch(`${API_BASE}/voice/${session_id}/audio`, {method: 'POST', body: fd})
  // Try to parse JSON, otherwise return blob
  const ct = res.headers.get('content-type') || ''
  if(ct.includes('application/json')) return res.json()
  const buf = await res.arrayBuffer()
  return { audio: buf }
}
