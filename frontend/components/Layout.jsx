import Header from './Header'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Header />
      <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
    </div>
  )
}
