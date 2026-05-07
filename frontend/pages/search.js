import { useState } from 'react'
import PropertyCard from '../components/PropertyCard'
import { lifestyleSearch } from '../lib/api'

export default function Search(){
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)

  async function onSearch(e){
    e.preventDefault()
    setLoading(true)
    const payload = { query, city: 'Dubai', max_results: 10 }
    try{
      const res = await lifestyleSearch(payload)
      setResults(res.results || [])
    }catch(err){
      console.error(err)
      setResults([])
    }finally{ setLoading(false) }
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-4">Agentic Lifestyle Search</h1>
      <form onSubmit={onSearch} className="flex gap-2 mb-6">
        <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="e.g. Quiet villa with morning sun" className="flex-1 p-3 rounded border" />
        <button className="bg-sky-600 text-white px-4 rounded">Search</button>
      </form>

      {loading ? <div>Loading AI results…</div> : (
        <div className="grid gap-4">
          {results.map(r => <PropertyCard key={r.property_id} p={{
            id: r.property_id, title: r.title, city: r.city, district: r.district, asking_price_usd: r.asking_price_usd, bedrooms: r.bedrooms, bathrooms: r.bathrooms, area_sqft: r.area_sqft
          }} />)}
        </div>
      )}
    </div>
  )
}
