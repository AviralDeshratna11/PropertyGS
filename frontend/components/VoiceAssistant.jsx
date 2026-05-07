import { useEffect, useRef, useState } from 'react'
import { createVoiceSession, voiceAudio, voiceText } from '../lib/api'

export default function VoiceAssistant({propertyId}){
  const [session, setSession] = useState(null)
  const [recording, setRecording] = useState(false)
  const [mediaRec, setMediaRec] = useState(null)
  const [log, setLog] = useState([])
  const audioRef = useRef()

  useEffect(()=>{
    return ()=>{ if(mediaRec && mediaRec.state !== 'inactive') mediaRec.stop() }
  },[mediaRec])

  async function startSession(){
    const res = await createVoiceSession(propertyId)
    setSession(res.session_id)
    setLog(prev=>[...prev, {type:'info', msg:'Session started '+res.session_id}])
  }

  async function startRecording(){
    if(!navigator.mediaDevices) return alert('No media devices')
    const stream = await navigator.mediaDevices.getUserMedia({audio:true})
    const mr = new MediaRecorder(stream)
    const chunks = []
    mr.ondataavailable = e=> chunks.push(e.data)
    mr.onstop = async ()=>{
      const blob = new Blob(chunks, {type:'audio/webm'})
      setRecording(false)
      setLog(prev=>[...prev, {type:'info', msg:'Sending audio to server'}])
      const res = await voiceAudio(session, blob)
      if(res.audio){
        const audioBlob = new Blob([res.audio], {type:'audio/mpeg'})
        audioRef.current.src = URL.createObjectURL(audioBlob)
        audioRef.current.play()
        setLog(prev=>[...prev, {type:'assistant', msg:'Played audio response'}])
      } else {
        setLog(prev=>[...prev, {type:'assistant', msg: JSON.stringify(res)}])
      }
    }
    mr.start()
    setMediaRec(mr)
    setRecording(true)
  }

  function stopRecording(){ if(mediaRec) mediaRec.stop() }

  async function sendText(t){
    const res = await voiceText(session, t)
    setLog(prev=>[...prev, {type:'assistant', msg: JSON.stringify(res)}])
  }

  return (
    <div className="bg-white p-4 rounded shadow">
      <h3 className="font-semibold">Voice Assistant</h3>
      {!session ? <button onClick={startSession} className="mt-2 bg-sky-600 text-white px-3 py-1 rounded">Start Session</button> : (
        <div className="mt-2 space-y-2">
          <div className="flex gap-2">
            {!recording ? <button onClick={startRecording} className="bg-red-600 text-white px-3 py-1 rounded">Record</button>
            : <button onClick={stopRecording} className="bg-gray-300 px-3 py-1 rounded">Stop</button>}
            <button onClick={()=>sendText('Hello, tell me about this property')} className="bg-sky-600 text-white px-3 py-1 rounded">Send Text</button>
          </div>
          <audio ref={audioRef} controls className="mt-2 w-full" />
          <div className="mt-2 text-xs text-gray-700 space-y-1">
            {log.map((l,i)=>(<div key={i} className={l.type==='assistant'? 'text-indigo-600':''}>{l.msg}</div>))}
          </div>
        </div>
      )}
    </div>
  )
}
