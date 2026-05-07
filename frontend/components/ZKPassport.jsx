import { useState } from 'react'
import { requestPassport, verifyPassport } from '../lib/api'

export default function ZKPassport({buyerId}){
  const [passport, setPassport] = useState(null)
  const [status, setStatus] = useState('idle')

  async function onRequest(){
    setStatus('requesting')
    try{
      const res = await requestPassport(buyerId, 10000)
      setPassport(res)
      setStatus('ready')
    }catch(e){ setStatus('error') }
  }

  async function onVerify(){
    if(!passport) return
    setStatus('verifying')
    try{
      const v = await verifyPassport(passport.token)
      setPassport({...passport, verified: v.verified})
      setStatus('verified')
    }catch(e){ setStatus('error') }
  }

  return (
    <div className="mt-3 p-3 bg-white rounded shadow-sm">
      <div className="text-sm font-semibold">ZK Investor Passport</div>
      <div className="mt-2">
        <button onClick={onRequest} className="px-2 py-1 bg-sky-600 text-white rounded">Request Passport</button>
        {passport && <button onClick={onVerify} className="ml-2 px-2 py-1 bg-emerald-600 text-white rounded">Verify</button>}
      </div>
      <div className="mt-3 text-xs text-gray-600">
        {status!=='idle' && <div>Status: {status}</div>}
        {passport && (
          <div className="mt-2 border p-2 rounded bg-gray-50 text-xs">
            <div><strong>Token:</strong> {passport.token}</div>
            <div><strong>Proof:</strong> {passport.proof_hash}</div>
            <div><strong>Threshold:</strong> ${passport.threshold_usd}</div>
            <div><strong>Verified:</strong> {String(passport.verified)}</div>
          </div>
        )}
      </div>
    </div>
  )
}
