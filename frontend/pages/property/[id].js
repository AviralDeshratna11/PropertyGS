import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { getProperty, startNegotiation, negotiationRound, runInspection } from '../../lib/api'
import GSplatViewer from '../../components/GSplatViewer'
import NegotiationLive from '../../components/NegotiationLive'
import VoiceAssistant from '../../components/VoiceAssistant'
import ZKPassport from '../../components/ZKPassport'

export default function PropertyPage(){
  const router = useRouter()
  const { id } = router.query
  const [prop, setProp] = useState(null)
  const [sceneId, setSceneId] = useState(null)
  const [sceneMeta, setSceneMeta] = useState(null)
  const [negSession, setNegSession] = useState(null)
  const [negPlan, setNegPlan] = useState(null)

  useEffect(()=>{
    if(!id) return
    getProperty(id).then(setProp).catch(()=>setProp(null))
    // fetch any pre-generated GSplat scenes for this property
    const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
    fetch(`${api}/perception/scenes?property_id=${id}`)
      .then(r=>r.json())
      .then(list=>{
        if(Array.isArray(list) && list.length>0){
          const s = list[0]
          setSceneId(s.scene_id || s.sceneId || s.id)
          setSceneMeta(s)
        }
      }).catch(()=>{})
  },[id])

  async function onStartNeg(){
    const payload = {
      property_id: parseInt(id), buyer_id: 'buyer-1', seller_id: 'seller-1', buyer_max_budget_usd: prop.asking_price_usd, seller_reserve_price_usd: prop.asking_price_usd, buyer_urgency: 0.6, seller_urgency: 0.4
    }
    const res = await startNegotiation(payload)
    setNegSession(res.session_id)
    if(res.plan) setNegPlan(res.plan)
  }

  async function onRound(){
    if(!negSession) return
    const res = await negotiationRound(negSession)
    // server will emit via stream; keep this to trigger server-side round
    console.log('round executed', res)
  }

  async function onInspect(){
    const res = await runInspection(parseInt(id))
    alert(`Inspection complete: ${res.report_id}`)
  }

  if(!prop) return <div>Loading property…</div>

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2">
          <h1 className="text-2xl font-semibold">{prop.title}</h1>
          <p className="text-gray-600">{prop.address} • {prop.city}</p>
          <p className="mt-4 text-sky-600 font-bold">${prop.asking_price_usd?.toLocaleString()}</p>
          <div className="mt-6">
            <GSplatViewer sceneId={sceneId} />
            {sceneMeta && (
              <div className="mt-2 text-sm text-gray-200">
                Scene: {sceneMeta.scene_id || sceneMeta.sceneId} • Gaussians: {sceneMeta.num_gaussians || sceneMeta.numGaussians}
              </div>
            )}
          </div>
        </div>
        <aside className="bg-white p-4 rounded shadow space-y-4">
          <div className="mb-3">Bedrooms: {prop.bedrooms}</div>
          <div className="mb-3">Bathrooms: {prop.bathrooms}</div>
          <div className="mb-3">Area: {prop.area_sqft} sqft</div>
          <div className="mt-4 flex flex-col gap-2">
            <button onClick={onStartNeg} className="bg-sky-600 text-white px-3 py-2 rounded">Start Negotiation</button>
            <button onClick={onRound} className="bg-white border px-3 py-2 rounded">Run One Negotiation Round</button>
            <button onClick={onInspect} className="bg-amber-500 text-white px-3 py-2 rounded">Run AI Inspection</button>
          </div>
          {negSession && <div className="mt-4 text-sm text-gray-600">Negotiation session: {negSession}</div>}
          {negPlan && (
            <div className="mt-3 p-3 bg-gray-50 rounded">
              <div className="text-sm font-semibold">Negotiation Plan</div>
              <div className="text-xs text-gray-600 mt-1">{negPlan.llm_text || negPlan.summary}</div>
              <div className="mt-2 text-xs">
                <div><strong>Buyer:</strong> Open with ${negPlan.buyer_strategy?.open_with}</div>
                <div><strong>Seller:</strong> Open with ${negPlan.seller_strategy?.open_with}</div>
              </div>
            </div>
          )}
          {negSession && <NegotiationLive sessionId={negSession} />}
          <VoiceAssistant propertyId={parseInt(id || 1)} />
          <ZKPassport buyerId={`buyer-${parseInt(id||1)}`} />
        </aside>
      </div>
    </div>
  )
}
