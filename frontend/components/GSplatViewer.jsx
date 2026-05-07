import { Canvas } from '@react-three/fiber'
import { useRef, useEffect, useState } from 'react'

function Box(){
  const ref = useRef()
  return (
    <mesh ref={ref} rotation={[0.4, 0.6, 0]}>
      <boxGeometry args={[2,2,2]} />
      <meshStandardMaterial color="#0ea5e9" />
    </mesh>
  )
}

export default function GSplatViewer({sceneId}){
  const [config, setConfig] = useState(null)

  useEffect(()=>{
    if(!sceneId) return
    fetch(`${process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000/api/v1'}/perception/scenes/${sceneId}/viewer`)
      .then(r=>r.json()).then(setConfig).catch(()=>setConfig(null))
  },[sceneId])

  return (
    <div className="w-full h-96 bg-black rounded overflow-hidden">
      <Canvas camera={{ position: [0,0,6] }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10,10,5]} />
        <Box />
      </Canvas>
      {config ? <div className="p-2 text-xs text-white">Scene: {config.scene || sceneId}</div> : <div className="p-2 text-xs text-white">Placeholder viewer (no scene)</div>}
    </div>
  )
}
