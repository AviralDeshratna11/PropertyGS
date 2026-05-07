import Link from 'next/link'

export default function Header(){
  return (
    <header className="bg-white shadow">
      <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
        <Link href="/">
          <a className="text-2xl font-semibold text-sky-600">PropOS</a>
        </Link>
        <nav className="space-x-4">
          <Link href="/search"><a className="text-sm text-gray-700 hover:text-sky-600">Search</a></Link>
          <Link href="/"><a className="text-sm text-gray-700 hover:text-sky-600">Listings</a></Link>
        </nav>
      </div>
    </header>
  )
}
