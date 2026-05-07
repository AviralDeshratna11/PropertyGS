import Link from 'next/link'

export default function PropertyCard({p}){
  return (
    <div className="bg-white rounded-lg shadow p-4 sm:flex sm:gap-4">
      <div className="w-full sm:w-40 h-48 sm:h-28 bg-gray-200 rounded-md overflow-hidden flex-shrink-0">
        <img src={p.image_url || '/placeholder-house.png'} alt="property" className="w-full h-full object-cover" />
      </div>
      <div className="flex-1 mt-3 sm:mt-0">
        <h3 className="text-lg font-semibold">{p.title}</h3>
        <p className="text-sm text-gray-600">{p.city} {p.district ? `— ${p.district}` : ''}</p>
        <p className="mt-2 text-sky-600 font-bold">${p.asking_price_usd?.toLocaleString()}</p>
        <div className="mt-4 flex items-center justify-between">
          <div className="text-xs text-gray-500">{p.bedrooms || '-'} bd • {p.bathrooms || '-'} ba • {p.area_sqft || '-'} sqft</div>
          <Link href={`/property/${p.id}`}><a className="text-sm text-white bg-sky-600 px-3 py-1 rounded">View</a></Link>
        </div>
      </div>
    </div>
  )
}
