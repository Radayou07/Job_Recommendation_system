import { useState, useEffect } from 'react'

const PROVINCES = [
  "Phnom Penh", "Kandal", "Siem Reap", "Sihanoukville", "Battambang",
  "Kampong Cham", "Kampot", "Kratié", "Mondulkiri", "Preah Vihear",
  "Ratanakiri", "Takeo", "Remote"
]

const MODELS = [
  { id: 'hybrid', name: 'AI Hybrid', desc: 'SBERT + TF-IDF' },
  { id: 'sbert', name: 'Semantic', desc: 'Context & Intent' },
  { id: 'tfidf', name: 'Keywords', desc: 'Exact Matching' },
  { id: 'bow', name: 'Frequency', desc: 'Word Count' },
]

const NAV_ITEMS = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
          d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
  },
  {
    id: 'job recommendation',
    label: 'Find Jobs',
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    id: 'returnee',
    label: 'Returnee',
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    ),
  },
  {
    id: 'interacy job',
    label: 'Connect',
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
          d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
  },
]

function JobCard({ job, variant = 'recommend' }) {
  return (
    <div className={`
      bg-white flex flex-col h-full rounded-2xl border
      transition-all duration-300 group overflow-hidden
      ${variant === 'returnee'
        ? 'border-slate-200 hover:border-emerald-300 hover:shadow-lg hover:shadow-emerald-500/10'
        : 'border-slate-200 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-500/10'}
    `}>
      {/* Top accent strip */}
      <div className={`h-1 w-full ${variant === 'returnee' ? 'bg-emerald-400' : 'bg-blue-500'}`} />

      <div className="p-6 flex flex-col flex-grow">
        {/* Header row */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex flex-col gap-2">
            <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1 rounded-full tracking-widest uppercase border ${
              variant === 'returnee' 
                ? 'bg-emerald-50 text-emerald-700 border-emerald-100' 
                : 'bg-blue-50 text-blue-600 border-blue-100'
            }`}>
              {variant === 'returnee' && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />}
              {Math.round(job.score * 100)}% Match
            </span>
          </div>
          
          <div className="flex flex-col items-end gap-1.5">
            {job.job_type && job.job_type !== 'nan' && (
              <span className="bg-slate-50 text-slate-500 text-[10px] font-bold px-2 py-0.5 rounded-md border border-slate-100 uppercase tracking-wider">
                {job.job_type}
              </span>
            )}
          </div>
        </div>

        {/* Title & Company */}
        <h3 className={`text-base font-bold leading-snug mb-1 transition-colors
          ${variant === 'returnee'
            ? 'text-slate-800 group-hover:text-emerald-700'
            : 'text-slate-800 group-hover:text-blue-700'}`}>
          {job.job_title}
        </h3>
        <p className="text-sm text-slate-500 font-medium mb-4">{job.company_name}</p>

        {/* Skills Tags */}
        {job.skills && job.skills.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-5">
            {job.skills.slice(0, 4).map((skill, i) => (
              <span 
                key={i} 
                className={`text-[10px] font-bold px-2 py-0.5 rounded-md border transition-colors
                  ${variant === 'returnee'
                    ? 'bg-emerald-50 text-emerald-600 border-emerald-100/50'
                    : 'bg-blue-50 text-blue-600 border-blue-100/50'}`}
              >
                {skill}
              </span>
            ))}
            {job.skills.length > 4 && (
              <span className="text-[10px] font-bold text-slate-400 px-1 py-0.5">
                +{job.skills.length - 4} more
              </span>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="mt-auto pt-4 border-t border-slate-100 flex items-center justify-between">
          <div className="flex gap-1.5">
            <span className="text-[10px] font-semibold bg-slate-100 text-slate-500 px-2 py-0.5 rounded uppercase tracking-wider">
              {job.experience}y exp
            </span>
            <span className="text-[10px] font-semibold bg-slate-100 text-slate-500 px-2 py-0.5 rounded uppercase tracking-wider">
              {job.education}
            </span>
          </div>
          <div className="flex items-center gap-1 text-[11px] text-slate-400 font-semibold">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
            </svg>
            {job.job_location}
          </div>
        </div>
      </div>
    </div>
  )
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-32 text-center">
      <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center mb-5">
        <svg className="w-8 h-8 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      </div>
      <p className="text-slate-400 font-bold text-xs uppercase tracking-[0.2em]">{message}</p>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('job recommendation')
  const [query, setQuery] = useState('')
  const [location, setLocation] = useState('Phnom Penh')
  const [modelType, setModelType] = useState('hybrid')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (activeTab === 'returnee' || (activeTab === 'job recommendation' && query)) {
      handleSearch()
    } else if (activeTab === 'job recommendation' && !query) {
      setResults([])
    }
  }, [activeTab, location])

  const handleSearch = async (e) => {
    if (e) e.preventDefault()
    
    setLoading(true)
    setError('')
    try {
      const isReturnee = activeTab === 'returnee'
      const endpoint = isReturnee ? 'http://localhost:8000/returnee' : 'http://localhost:8000/recommend'
      
      const payload = { 
        query: query.trim(), 
        user_location: location, 
        top_n: isReturnee ? 12 : 9, 
        model_type: modelType 
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error()
      setResults(await res.json())
    } catch {
      setError('Connection failed. Please ensure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen bg-white font-sans">

      {/* ── SIDEBAR ── deep dark blue */}
      <aside
        className="w-64 flex-shrink-0 flex flex-col sticky top-0 h-screen"
        style={{ backgroundColor: '#0b1f45' }}
      >
        {/* Logo */}
        <div className="px-6 pt-8 pb-6">
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-xl flex items-center justify-center text-sm font-black text-white"
              style={{ backgroundColor: '#2563eb' }}
            >
              R
            </div>
            <span className="text-white text-lg font-black tracking-tight">
              Rec<span style={{ color: '#60a5fa' }}>AI</span>
            </span>
          </div>
        </div>

        {/* Section label */}
        <div className="px-6 mb-2">
          <span className="text-[9px] font-bold tracking-[0.18em] uppercase" style={{ color: '#4a6fa5' }}>
            Navigation
          </span>
        </div>

        {/* Nav links */}
        <nav className="px-3 space-y-0.5 flex-grow">
          {NAV_ITEMS.map((item) => {
            const active = activeTab === item.id
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group"
                style={{
                  backgroundColor: active ? '#1e3a6e' : 'transparent',
                  color: active ? '#ffffff' : '#7ea3cc',
                }}
                onMouseEnter={e => { if (!active) e.currentTarget.style.backgroundColor = '#142d55' }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.backgroundColor = 'transparent' }}
              >
                {/* Active indicator bar */}
                <span
                  className="absolute left-0 w-0.5 h-7 rounded-r transition-opacity duration-200"
                  style={{
                    backgroundColor: '#3b82f6',
                    opacity: active ? 1 : 0,
                  }}
                />
                <span style={{ color: active ? '#60a5fa' : '#4a6fa5' }}>{item.icon}</span>
                <span className="text-sm font-semibold">{item.label}</span>
                {active && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#3b82f6' }} />
                )}
              </button>
            )
          })}
        </nav>

        {/* Model status pill */}
        <div className="px-4 pb-8">
          <div
            className="rounded-xl px-4 py-3"
            style={{ backgroundColor: '#0d2554', border: '1px solid #1a3a6e' }}
          >
            <p className="text-[9px] font-bold tracking-[0.15em] uppercase mb-2" style={{ color: '#4a6fa5' }}>
              Active Model
            </p>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold" style={{ color: '#93c5fd' }}>
                {MODELS.find(m => m.id === modelType)?.name}
              </span>
              <span className="flex items-center gap-1.5 text-[10px] font-semibold" style={{ color: '#34d399' }}>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
                Live
              </span>
            </div>
            <p className="text-[10px] mt-0.5" style={{ color: '#4a6fa5' }}>
              {MODELS.find(m => m.id === modelType)?.desc}
            </p>
          </div>
        </div>
      </aside>

      {/* ── MAIN ── pure white */}
      <div className="flex-1 flex flex-col min-h-screen bg-white">

        {/* Sticky header */}
        <header className="sticky top-0 z-30 bg-white border-b border-slate-100 px-8 py-4">
          <div className="max-w-5xl mx-auto flex items-center justify-between">
            <div>
              <h1 className="text-lg font-black text-slate-800 capitalize tracking-tight">{activeTab}</h1>
              <p className="text-xs text-slate-400 font-medium mt-0.5">
                {activeTab === 'returnee'
                  ? 'Entry-level pathways for returning workers'
                  : 'AI-powered job matching engine'}
              </p>
            </div>

            <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
              <svg className="w-3.5 h-3.5 text-slate-400" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
              </svg>
              <select
                className="bg-transparent border-none text-slate-700 font-semibold text-xs outline-none cursor-pointer"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
              >
                {PROVINCES.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
          </div>
        </header>

        <main className="max-w-5xl mx-auto w-full px-8 py-8 flex-grow">

          {/* ── SHARED SEARCH BAR ── */}
          {(activeTab === 'job recommendation' || activeTab === 'returnee') && (
            <div className="bg-white border border-slate-200 rounded-2xl p-2 shadow-sm shadow-slate-100 mb-8">
              <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-2">
                <div className="flex-1 flex items-center gap-3 bg-slate-50 rounded-xl px-4 py-3 border border-transparent focus-within:border-blue-200 focus-within:bg-white transition-all">
                  <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    type="text"
                    placeholder={activeTab === 'returnee' ? "Search entry-level pathways..." : "Skills, role, or job type..."}
                    className="bg-transparent w-full text-sm text-slate-800 font-medium placeholder-slate-400 outline-none"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                  />
                </div>

                <div className="flex gap-2">
                  <select
                    className="bg-slate-50 border border-slate-200 text-slate-600 font-semibold text-xs rounded-xl px-3 outline-none cursor-pointer"
                    value={modelType}
                    onChange={e => setModelType(e.target.value)}
                  >
                    {MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>

                  <button
                    type="submit"
                    disabled={loading}
                    className="text-white text-sm font-bold px-7 py-3 rounded-xl transition-all active:scale-95 disabled:opacity-50"
                    style={{ backgroundColor: activeTab === 'returnee' ? '#10b981' : '#1d4ed8' }}
                  >
                    {loading ? 'Matching…' : 'Match'}
                  </button>
                </div>
              </form>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-100 text-red-600 text-sm font-semibold p-4 rounded-xl text-center mb-8">
              {error}
            </div>
          )}

          {/* ── FIND JOBS TAB ── */}
          {activeTab === 'job recommendation' && (
            <div className="space-y-8">
              {/* Results */}
              {loading ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                  {[...Array(6)].map((_, i) => (
                    <div key={i} className="h-48 bg-slate-100 rounded-2xl animate-pulse" />
                  ))}
                </div>
              ) : results.length > 0 ? (
                <>
                  <p className="text-xs text-slate-400 font-semibold uppercase tracking-widest">
                    {results.length} matches found
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                    {results.map((job, i) => <JobCard key={i} job={job} variant="recommend" />)}
                  </div>
                </>
              ) : (
                <EmptyState message="Enter a skill or role to find matches" />
              )}
            </div>
          )}

          {/* ── RETURNEE TAB ── */}
          {activeTab === 'returnee' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-black text-slate-800">Available Pathways</h2>
                  <p className="text-xs text-slate-400 font-medium mt-1">Entry-level · High School education</p>
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-600 bg-emerald-50 border border-emerald-100 px-3 py-1.5 rounded-full">
                  {results.length} pathways
                </span>
              </div>

              {loading ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                  {[...Array(6)].map((_, i) => <div key={i} className="h-48 bg-slate-100 rounded-2xl animate-pulse" />)}
                </div>
              ) : results.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                  {results.map((job, i) => <JobCard key={i} job={job} variant="returnee" />)}
                </div>
              ) : (
                <EmptyState message={`No pathway jobs in ${location}`} />
              )}
            </div>
          )}

          {/* ── OTHER TABS ── */}
          {activeTab !== 'job recommendation' && activeTab !== 'returnee' && (
            <div className="flex flex-col items-center justify-center h-full py-40 text-center">
              <div className="w-20 h-20 rounded-2xl bg-slate-100 flex items-center justify-center text-3xl mb-6">
                🚧
              </div>
              <h2 className="text-2xl font-black text-slate-800 uppercase tracking-tight mb-2 capitalize">
                {activeTab}
              </h2>
              <p className="text-xs text-slate-400 font-bold tracking-[0.2em] uppercase">
                In development · Q3 2026
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}