import { useState, useEffect, useRef } from 'react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  PieChart, Pie, Cell, LabelList 
} from 'recharts'

const PROVINCES = [
  "All", "Phnom Penh", "Kandal", "Siem Reap", "Sihanoukville", "Battambang",
  "Kampong Cham", "Kampot", "Kratié", "Mondulkiri", "Preah Vihear",
  "Ratanakiri", "Takeo", "Remote"
]

const MODELS = [
  { id: 'sbert', name: 'SBERT', desc: 'Context & Intent' },
  { id: 'tfidf', name: 'TF-IDF', desc: 'Exact Matching' },
  { id: 'bow', name: 'BoW', desc: 'Word Count' },
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
]

function CustomSelect({ value, onChange, options, icon, className = "" }) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const selectedOption = options.find(opt => 
    typeof opt === 'string' ? opt === value : opt.id === value
  )
  const label = typeof selectedOption === 'string' ? selectedOption : selectedOption?.name
  
  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between gap-2 bg-white border border-blue-100/50 shadow-sm rounded-xl px-4 py-3 transition-all hover:border-blue-200 active:scale-95 focus:outline-none focus:ring-0"
      >
        <div className="flex items-center gap-2">
          {icon && <span className="text-blue-400">{icon}</span>}
          <span className="text-slate-700 font-bold text-xs truncate">{label}</span>
        </div>
        <svg className={`w-3 h-3 text-slate-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      <div className={`
        absolute right-0 mt-2 w-48 bg-white border border-blue-100/50 shadow-xl rounded-2xl overflow-hidden z-50
        transition-all duration-200 origin-top-right transform
        ${isOpen ? 'opacity-100 scale-100 translate-y-0' : 'opacity-0 scale-95 -translate-y-2 pointer-events-none'}
      `}>
        <div className="py-1 max-h-60 overflow-y-auto custom-scrollbar">
          {options.map((opt) => {
            const id = typeof opt === 'string' ? opt : opt.id
            const name = typeof opt === 'string' ? opt : opt.name
            const active = id === value

            return (
              <button
                key={id}
                type="button"
                onClick={() => {
                  onChange(id)
                  setIsOpen(false)
                }}
                className={`w-full text-left px-4 py-2.5 text-xs font-bold transition-colors focus:outline-none focus:ring-0
                  ${active ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'}
                `}
              >
                {name}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function JobCard({ job, variant = 'recommend' }) {
  const isReturnee = variant === 'returnee'
  
  return (
    <div className={`
      bg-white flex flex-col h-full rounded-[24px] border border-blue-100/30
      transition-all duration-500 group overflow-hidden
      hover:shadow-2xl hover:shadow-blue-500/10 hover:-translate-y-1.5
    `}>
      {/* Top accent strip */}
      <div className={`h-1.5 w-full ${isReturnee ? 'bg-blue-300' : 'bg-blue-600'}`} />

      <div className="p-7 flex flex-col flex-grow">
        {/* Header row */}
        <div className="flex items-start justify-between mb-5">
          <div className="flex flex-col gap-2">
            <span className={`inline-flex items-center gap-1.5 text-[10px] font-extrabold px-3 py-1.5 rounded-xl tracking-widest uppercase border ${
              isReturnee 
                ? 'bg-blue-50/50 text-blue-600 border-blue-100/50' 
                : 'bg-blue-50/50 text-blue-700 border-blue-100/50'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${isReturnee ? 'bg-blue-400' : 'bg-blue-500'} inline-block`} />
              {Math.round(job.score * 100)}% Match
            </span>
          </div>
          
          <div className="flex flex-col items-end gap-1.5">
            {job.job_type && job.job_type !== 'nan' && (
              <span className="bg-slate-50 text-slate-500 text-[9px] font-black px-2.5 py-1 rounded-lg border border-slate-100 uppercase tracking-widest">
                {job.job_type}
              </span>
            )}
          </div>
        </div>

        {/* Title & Company */}
        <h3 className={`text-[17px] font-black leading-tight mb-1.5 transition-colors duration-300
          text-slate-800 group-hover:text-blue-600`}>
          {job.job_title}
        </h3>
        <p className="text-sm text-slate-400 font-bold mb-5 tracking-tight">{job.company_name}</p>

        {/* Skills Tags */}
        {job.skills && job.skills.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6">
            {job.skills.slice(0, 4).map((skill, i) => (
              <span 
                key={i} 
                className="text-[10px] font-bold px-2.5 py-1 rounded-lg bg-blue-50/30 text-blue-500 border border-blue-100/20"
              >
                {skill}
              </span>
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="mt-auto pt-5 border-t border-slate-50 flex items-center justify-between">
          <div className="flex gap-2">
            <span className="text-[10px] font-black bg-slate-100/50 text-slate-400 px-2.5 py-1 rounded-lg uppercase tracking-wider">
              {job.experience}y exp
            </span>
            <span className="text-[10px] font-black bg-slate-100/50 text-slate-400 px-2.5 py-1 rounded-lg uppercase tracking-wider">
              {job.education.toLowerCase() === 'none' ? 'General Ed' : job.education}
            </span>
          </div>

          <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-black uppercase tracking-tighter">
            <svg className="w-3.5 h-3.5 text-blue-300" fill="currentColor" viewBox="0 0 24 24">
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

function DashboardView({ location }) {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // High-contrast diverse palette for maximum distinction
  const COLORS = [
    '#3b82f6', // Blue
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#8b5cf6', // Violet
    '#ec4899', // Pink
    '#06b6d4', // Cyan
    '#f97316', // Orange
    '#6366f1', // Indigo
    '#84cc16'  // Lime
  ]

  useEffect(() => {
    fetchStats()
  }, [location])

  const fetchStats = async () => {
    setLoading(true)
    try {
      const res = await fetch(`http://localhost:8000/dashboard-stats?location=${location}`)
      if (!res.ok) throw new Error()
      setStats(await res.json())
    } catch {
      setError('Failed to load dashboard data.')
    } finally {
      setLoading(false)
    }
  }

  if (loading) return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse">
      {[...Array(6)].map((_, i) => <div key={i} className="h-64 bg-white/50 rounded-3xl" />)}
    </div>
  )

  if (error) return <div className="text-center py-20 text-blue-500 font-bold bg-white/50 rounded-3xl border border-blue-100">{error}</div>
  if (!stats) return null

  return (
    <div className="space-y-8 pb-12">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        {[
          { 
            label: 'Total Postings', 
            value: stats.total_jobs, 
            icon: (
              <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            )
          },
          { 
            label: 'Top Industry', 
            value: stats.jobs_by_category[0]?.name, 
            icon: (
              <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            )
          },
          { 
            label: 'Top Location', 
            value: stats.jobs_by_location[0]?.name, 
            icon: (
              <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            )
          }
        ].map((kpi, i) => (
          <div key={i} className="bg-white/70 backdrop-blur-sm p-7 rounded-[28px] border border-blue-100/50 shadow-sm hover:shadow-md transition-all duration-300">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center">
                {kpi.icon}
              </div>
              <p className="text-[11px] font-black text-blue-400 uppercase tracking-[0.15em]">{kpi.label}</p>
            </div>
            <h4 className="text-2xl font-black text-slate-800 truncate">{kpi.value}</h4>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Jobs by Category */}
        <div className="bg-white/70 backdrop-blur-sm p-8 rounded-[32px] border border-blue-100/50 shadow-sm">
          <h3 className="text-xs font-black text-blue-500 mb-8 uppercase tracking-[0.2em]">Industry Demand</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.jobs_by_category} layout="vertical" margin={{ right: 40 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" opacity={0.5} />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  width={140} 
                  tick={{fontSize: 10, fontWeight: 700, fill: '#64748b'}} 
                  axisLine={false} 
                  tickLine={false} 
                />
                <Tooltip 
                  cursor={{fill: '#f1f5f9'}}
                  contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', background: 'rgba(255, 255, 255, 0.9)'}}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={16} animationDuration={300} animationBegin={0}>
                  {stats.jobs_by_category.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                  <LabelList dataKey="value" position="right" style={{ fill: '#64748b', fontSize: '10px', fontWeight: 'bold' }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Jobs by Location */}
        <div className="bg-white/70 backdrop-blur-sm p-8 rounded-[32px] border border-blue-100/50 shadow-sm">
          <h3 className="text-xs font-black text-blue-500 mb-8 uppercase tracking-[0.2em]">Regional Distribution</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats.jobs_by_location}
                  innerRadius={70}
                  outerRadius={100}
                  paddingAngle={8}
                  dataKey="value"
                  stroke="none"
                  animationDuration={300}
                  animationBegin={0}
                  label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                >
                  {stats.jobs_by_location.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                   contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', background: 'rgba(255, 255, 255, 0.9)'}}
                />
                <Legend iconType="circle" wrapperStyle={{fontSize: '10px', fontWeight: '800', paddingTop: '20px'}} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Job Type Distribution */}
        <div className="bg-white/70 backdrop-blur-sm p-8 rounded-[32px] border border-blue-100/50 shadow-sm">
          <h3 className="text-xs font-black text-blue-500 mb-8 uppercase tracking-[0.2em]">Job Type Split</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.job_type_distribution} margin={{ top: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" opacity={0.5} />
                <XAxis 
                  dataKey="name" 
                  tick={{fontSize: 10, fontWeight: 700, fill: '#64748b'}} 
                  axisLine={false} 
                  tickLine={false} 
                />
                <YAxis hide />
                <Tooltip 
                  cursor={{fill: '#f1f5f9'}}
                  contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', background: 'rgba(255, 255, 255, 0.9)'}}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={40} animationDuration={300} animationBegin={0}>
                  {stats.job_type_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[(index + 2) % COLORS.length]} />
                  ))}
                  <LabelList dataKey="value" position="top" style={{ fill: '#64748b', fontSize: '10px', fontWeight: 'bold' }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Salary Distribution */}
        <div className="bg-white/70 backdrop-blur-sm p-8 rounded-[32px] border border-blue-100/50 shadow-sm">
          <h3 className="text-xs font-black text-blue-500 mb-8 uppercase tracking-[0.2em]">Salary Brackets</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.salary_distribution} margin={{ top: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" opacity={0.5} />
                <XAxis 
                  dataKey="name" 
                  tick={{fontSize: 10, fontWeight: 700, fill: '#64748b'}} 
                  axisLine={false} 
                  tickLine={false} 
                />
                <YAxis hide />
                <Tooltip 
                  cursor={{fill: '#f1f5f9'}}
                  contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', background: 'rgba(255, 255, 255, 0.9)'}}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={40} animationDuration={300} animationBegin={0}>
                  {stats.salary_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[(index + 4) % COLORS.length]} />
                  ))}
                  <LabelList dataKey="value" position="top" style={{ fill: '#64748b', fontSize: '10px', fontWeight: 'bold' }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Education Required */}
        <div className="bg-white/70 backdrop-blur-sm p-8 rounded-[32px] border border-blue-100/50 shadow-sm">
          <h3 className="text-xs font-black text-blue-500 mb-8 uppercase tracking-[0.2em]">Education Requirements</h3>
          <div className="h-[300px] w-full relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats.education_distribution}
                  innerRadius={60}
                  outerRadius={95}
                  paddingAngle={4}
                  dataKey="value"
                  stroke="none"
                  animationDuration={300}
                  animationBegin={0}
                  label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                >
                  {stats.education_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[(index + 1) % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                   contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', background: 'rgba(255, 255, 255, 0.9)'}}
                />
                <Legend 
                  layout="vertical" 
                  verticalAlign="middle" 
                  align="right"
                  iconType="circle"
                  formatter={(value) => (
                    <span className="text-[10px] font-black text-slate-500 uppercase ml-1">{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Avg Salary by Industry */}
        <div className="bg-white/70 backdrop-blur-sm p-8 rounded-[32px] border border-blue-100/50 shadow-sm">
          <h3 className="text-xs font-black text-blue-500 mb-8 uppercase tracking-[0.2em]">Avg Salary by Industry</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.avg_salary_by_industry} layout="vertical" margin={{ right: 50 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" opacity={0.5} />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  width={140} 
                  tick={{fontSize: 10, fontWeight: 700, fill: '#64748b'}} 
                  axisLine={false} 
                  tickLine={false} 
                />
                <Tooltip 
                  cursor={{fill: '#f1f5f9'}}
                  formatter={(value) => [`$${value}`, 'Avg Salary']}
                  contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', background: 'rgba(255, 255, 255, 0.9)'}}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={16} animationDuration={300} animationBegin={0}>
                  {stats.avg_salary_by_industry.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[(index + 3) % COLORS.length]} />
                  ))}
                  <LabelList dataKey="value" position="right" formatter={(v) => `$${v}`} style={{ fill: '#64748b', fontSize: '10px', fontWeight: 'bold' }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [query, setQuery] = useState('')
  const [location, setLocation] = useState('All')
  const [modelType, setModelType] = useState('sbert')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [buttonLoading, setButtonLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (activeTab === 'returnee' || (activeTab === 'job recommendation' && query)) {
      handleSearch(null, false)
    } else if (activeTab === 'job recommendation' && !query) {
      setResults([])
    }
  }, [activeTab, location])

  const handleSearch = async (e, isManual = true) => {
    if (e) e.preventDefault()
    
    setLoading(true)
    if (isManual) setButtonLoading(true)
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
      setButtonLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen bg-[#f0f7ff]"> 

      {/* ── SIDEBAR ── light blue tint */}
      <aside
        className="w-64 flex-shrink-0 flex flex-col sticky top-0 h-screen bg-blue-600 rounded-r-[32px] shadow-2xl z-40"
      >
        {/* Logo */}
        <div className="px-7 pt-9 pb-8">
          <div className="flex items-center gap-3">
            <span className="text-white text-xl font-black tracking-tight">
              little<span className="text-blue-200">help</span>
            </span>
          </div>
        </div>

        {/* Section label */}
        <div className="px-7 mb-3">
          <span className="text-[10px] font-bold tracking-[0.2em] uppercase text-blue-100/60">
            Navigation
          </span>
        </div>

        {/* Nav links */}
        <nav className="px-4 space-y-1 flex-grow">
          {NAV_ITEMS.map((item) => {
            const active = activeTab === item.id
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className="w-full flex items-center gap-3.5 px-4 py-3 rounded-2xl transition-all duration-300 group relative"
                style={{
                  backgroundColor: active ? '#ffffff' : 'transparent',
                  color: active ? '#2563eb' : '#dbeafe',
                  boxShadow: active ? '0 10px 15px -3px rgba(0, 0, 0, 0.1)' : 'none',
                }}
                onMouseEnter={e => { if (!active) e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)' }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.backgroundColor = 'transparent' }}
              >
                <span className={`transition-colors duration-300 ${active ? 'text-blue-600' : 'text-blue-200 group-hover:text-white'}`}>
                  {item.icon}
                </span>
                <span className="text-sm font-bold tracking-tight">{item.label}</span>
                {active && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-600 shadow-sm" />
                )}
              </button>
            )
          })}
        </nav>

        {/* Model status pill */}
        <div className="px-5 pb-10">
          <div
            className="rounded-2xl px-5 py-4 bg-blue-700/40 border border-blue-500/30 shadow-sm"
          >
            <p className="text-[9px] font-bold tracking-[0.15em] uppercase mb-2 text-blue-100/80">
              Active Engine
            </p>
            <div className="flex items-center justify-between">
              <span className="text-xs font-extrabold text-white">
                {MODELS.find(m => m.id === modelType)?.name}
              </span>
              <span className="flex items-center gap-1.5 text-[10px] font-bold text-blue-200">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-300 animate-pulse" />
                Active
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* ── MAIN ── light blue bg */}
      <div className="flex-1 flex flex-col min-h-screen">

        {/* Header */}
        <header className="sticky top-0 z-30 bg-[#f0f7ff]/90 backdrop-blur-sm px-10 py-7">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div>
              <h1 className="text-xl font-black text-slate-800 capitalize tracking-tight">{activeTab}</h1>
              <p className="text-[11px] text-slate-400 font-bold uppercase tracking-widest mt-0.5 opacity-70">
                {activeTab === 'dashboard' ? 'Market Intelligence' : 
                 activeTab === 'returnee' ? 'Entry-Level Pathways' : 
                 'AI Job Matching'}
              </p>
            </div>

            <CustomSelect 
              value={location} 
              onChange={setLocation} 
              options={PROVINCES}
              icon={(
                <svg className="w-4 h-4 text-blue-400" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                </svg>
              )}
            />
          </div>
        </header>

        <main className="max-w-6xl mx-auto w-full px-10 pt-2 pb-10 flex-grow">
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

                <div className="flex items-center gap-2">
                  <CustomSelect 
                    value={modelType} 
                    onChange={setModelType} 
                    options={MODELS} 
                    className="w-36"
                  />

                  <button
                    type="submit"
                    disabled={loading}
                    className="text-white text-sm font-bold px-7 py-3 rounded-xl transition-all active:scale-95 disabled:opacity-50 shadow-md shadow-blue-500/20 hover:shadow-lg"
                    style={{ backgroundColor: '#3b82f6' }}
                  >
                    {buttonLoading ? 'Matching…' : 'Match'}
                  </button>                </div>
              </form>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-100 text-red-600 text-sm font-semibold p-4 rounded-xl text-center mb-8 fade-slide-in">
              {error}
            </div>
          )}

          {/* ── TAB CONTENT ── animated individually to prevent total page flickering */}
          <div key={activeTab} className="fade-slide-in">
            {activeTab === 'dashboard' && <DashboardView location={location} />}
            
            {activeTab === 'job recommendation' && (
              <div className="space-y-8">
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

            {activeTab === 'returnee' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-black text-slate-800">Available Pathways</h2>
                    <p className="text-xs text-slate-400 font-medium mt-1">Entry-level · High School education</p>
                  </div>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-blue-600 bg-blue-50 border border-blue-100 px-3 py-1.5 rounded-full">
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

            {/* fallback for WIP tabs */}
            {activeTab !== 'job recommendation' && activeTab !== 'returnee' && activeTab !== 'dashboard' && (
              <div className="flex flex-col items-center justify-center h-full py-40 text-center">
                <div className="w-24 h-24 rounded-3xl bg-white border border-blue-100/50 shadow-sm flex items-center justify-center mb-8">
                  <svg className="w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-black text-slate-800 uppercase tracking-tight mb-2 capitalize">
                  {activeTab}
                </h2>
                <p className="text-xs text-blue-400 font-bold tracking-[0.2em] uppercase">
                  Engineering in progress · Q3 2026
                </p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
