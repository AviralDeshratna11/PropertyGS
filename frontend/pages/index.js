import { useEffect, useState } from 'react'
import { listProperties } from '../lib/api'
import PropertyCard from '../components/PropertyCard'

export default function Home(){
  const [propsList, setPropsList] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(()=>{
    listProperties().then(setPropsList).catch(()=>setPropsList([])).finally(()=>setLoading(false))
  },[])

  return (
    <div>
      <section className="mb-8">
        <div className="rounded-lg bg-gradient-to-r from-sky-600 to-indigo-600 text-white p-10">
          <h1 className="text-3xl font-bold">PropOS — Intelligent Property Marketplace</h1>
          <p className="mt-2 text-sky-100">Agentic search, MARL negotiation, ZKP verification, and inspection overlays.</p>
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Featured Listings</h2>
        </div>

        {loading ? <div>Loading...</div> : (
          <div className="grid gap-4">
            {propsList.map(p => <PropertyCard key={p.id} p={p} />)}
          </div>
        )}
      </section>
    </div>
  )
}
