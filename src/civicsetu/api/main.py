from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from civicsetu.api.middleware.logging import LoggingMiddleware
from civicsetu.api.routes import health, query
from civicsetu.config.settings import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

def get_landing_page_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>CivicSetu — AI-Powered RERA Research</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="Open-source RAG system for querying Indian civic and legal documents with accurate citations and cross-reference traversal.">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .answer-body { font-family: 'Merriweather', serif; }
        .gradient-text {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #d946ef 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .glass {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        .pulse-ring {
            animation: pulse-ring 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse-ring {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .slide-up {
            animation: slideUp 0.4s ease-out forwards;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .citation-card {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .citation-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        .example-chip {
            transition: all 0.2s ease;
        }
        .example-chip:hover {
            transform: scale(1.02);
        }
    </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
    <header class="glass sticky top-0 z-50 border-b border-white/20">
        <div class="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center shadow-lg">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"/>
                    </svg>
                </div>
                <div>
                    <h1 class="text-xl font-bold gradient-text">CivicSetu</h1>
                    <p class="text-xs text-gray-500">RERA Research Engine</p>
                </div>
            </div>
            <div class="flex items-center gap-4">
                <span class="hidden sm:inline-flex items-center gap-1.5 text-xs text-green-600 bg-green-50 px-3 py-1 rounded-full">
                    <span class="w-2 h-2 bg-green-500 rounded-full pulse-ring"></span>
                    Live
                </span>
                <a href="https://github.com/adeshboudh/civicsetu" target="_blank" class="text-gray-600 hover:text-gray-900 transition-colors">
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                        <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd"/>
                    </svg>
                </a>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-6 py-12">
        <div class="text-center mb-12">
            <h2 class="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
                AI-Powered <span class="gradient-text">RERA Research</span>
            </h2>
            <p class="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
                Query Indian real estate regulations across 5 jurisdictions. Get cited, structured answers
                with cross-reference traversal and conflict detection — powered by LangGraph agents.
            </p>
        </div>

        <div class="glass rounded-3xl shadow-2xl p-8 mb-8 border border-white/30">
            <div class="flex flex-col lg:flex-row gap-4 mb-6">
                <div class="flex-1 relative">
                    <svg class="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    <input id="query" type="text"
                           placeholder="Ask about RERA: promoter obligations, agent registration, penalties..."
                           class="w-full pl-12 pr-4 py-4 text-lg border-2 border-gray-200 rounded-2xl focus:border-blue-400 focus:outline-none focus:ring-4 focus:ring-blue-100 transition-all">
                </div>
                <div class="flex gap-3">
                    <select id="jurisdiction" class="px-4 py-4 text-gray-700 bg-white border-2 border-gray-200 rounded-2xl focus:border-blue-400 focus:outline-none cursor-pointer">
                        <option value="">All India</option>
                        <option value="CENTRAL">RERA Act (Central)</option>
                        <option value="MAHARASHTRA">Maharashtra</option>
                        <option value="UTTAR_PRADESH">Uttar Pradesh</option>
                        <option value="KARNATAKA">Karnataka</option>
                        <option value="TAMIL_NADU">Tamil Nadu</option>
                    </select>
                    <button onclick="doQuery()"
                            class="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-2xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-200 flex items-center gap-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                        </svg>
                        <span>Ask</span>
                    </button>
                </div>
            </div>

            <div class="flex flex-wrap gap-2">
                <span class="text-sm text-gray-500 mr-2">Try:</span>
                <button onclick="setQuery(this.textContent)" class="example-chip text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 px-4 py-2 rounded-full border border-blue-200">
                    What are promoter obligations under RERA?
                </button>
                <button onclick="setQuery(this.textContent)" class="example-chip text-sm bg-purple-50 hover:bg-purple-100 text-purple-700 px-4 py-2 rounded-full border border-purple-200">
                    Penalties for delayed possession
                </button>
                <button onclick="setQuery(this.textContent)" class="example-chip text-sm bg-green-50 hover:bg-green-100 text-green-700 px-4 py-2 rounded-full border border-green-200">
                    Agent registration requirements
                </button>
                <button onclick="setQuery(this.textContent)" class="example-chip text-sm bg-amber-50 hover:bg-amber-100 text-amber-700 px-4 py-2 rounded-full border border-amber-200">
                    Complaint filing process
                </button>
            </div>
        </div>

        <div id="response" class="hidden space-y-6">
            <div id="loading" class="glass rounded-2xl p-12 text-center">
                <div class="inline-flex items-center gap-3">
                    <div class="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                    <span class="text-gray-600 text-lg">Analyzing RERA regulations...</span>
                </div>
                <p class="text-gray-400 text-sm mt-3">This may take 10-15 seconds</p>
            </div>

            <div id="answer-card" class="glass rounded-2xl shadow-xl p-8 slide-up hidden">
                <div class="flex items-start gap-4 mb-6">
                    <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center flex-shrink-0">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                    </div>
                    <div class="flex-1">
                        <h3 class="text-lg font-semibold text-gray-800 mb-1">Answer</h3>
                        <div id="query-type-badge" class="inline-block text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full"></div>
                    </div>
                </div>
                <div id="answer" class="answer-body text-gray-800 text-lg leading-relaxed"></div>
                <div id="warnings" class="mt-6 hidden"></div>
            </div>

            <div id="stats-row" class="grid grid-cols-2 md:grid-cols-4 gap-4 hidden">
                <div class="glass rounded-xl p-4 text-center">
                    <div id="confidence-value" class="text-2xl font-bold text-blue-600">--</div>
                    <div class="text-xs text-gray-500">Confidence</div>
                </div>
                <div class="glass rounded-xl p-4 text-center">
                    <div id="citations-count" class="text-2xl font-bold text-purple-600">--</div>
                    <div class="text-xs text-gray-500">Citations</div>
                </div>
                <div class="glass rounded-xl p-4 text-center">
                    <div id="chunks-count" class="text-2xl font-bold text-green-600">--</div>
                    <div class="text-xs text-gray-500">Chunks</div>
                </div>
                <div class="glass rounded-xl p-4 text-center">
                    <div id="retry-count" class="text-2xl font-bold text-amber-600">--</div>
                    <div class="text-xs text-gray-500">Retries</div>
                </div>
            </div>

            <div id="citations-card" class="glass rounded-2xl shadow-xl p-8 hidden">
                <h3 class="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                    </svg>
                    Legal Citations
                </h3>
                <div id="citations" class="grid gap-3"></div>
            </div>

            <div class="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
                <div class="flex items-start gap-2">
                    <svg class="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                    </svg>
                    <div>
                        <strong>Legal Disclaimer:</strong> This is AI-generated legal information, not legal advice.
                        Always verify with a qualified lawyer or the official gazette before making any legal decisions.
                    </div>
                </div>
            </div>
        </div>

        <div class="mt-16 grid md:grid-cols-3 gap-6">
            <div class="glass rounded-2xl p-6 border border-white/30 hover:shadow-lg transition-shadow">
                <div class="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center mb-4">
                    <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
                    </svg>
                </div>
                <h3 class="font-semibold text-gray-800 mb-2">5-Jurisdiction Coverage</h3>
                <p class="text-sm text-gray-600">Central RERA Act + Maharashtra, UP, Karnataka, and Tamil Nadu state rules</p>
            </div>
            <div class="glass rounded-2xl p-6 border border-white/30 hover:shadow-lg transition-shadow">
                <div class="w-12 h-12 rounded-xl bg-purple-100 flex items-center justify-center mb-4">
                    <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
                    </svg>
                </div>
                <h3 class="font-semibold text-gray-800 mb-2">Cross-Reference Graph</h3>
                <p class="text-sm text-gray-600">Neo4j-powered traversal between sections and DERIVED_FROM relationships</p>
            </div>
            <div class="glass rounded-2xl p-6 border border-white/30 hover:shadow-lg transition-shadow">
                <div class="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center mb-4">
                    <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                </div>
                <h3 class="font-semibold text-gray-800 mb-2">Hallucination Detection</h3>
                <p class="text-sm text-gray-600">Validator agent with confidence scoring and citation verification</p>
            </div>
        </div>

        <div class="mt-12 glass rounded-2xl p-8 border border-white/30">
            <h3 class="text-lg font-semibold text-gray-800 mb-6">Document Coverage</h3>
            <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div class="flex items-center gap-3 p-4 bg-white/50 rounded-xl">
                    <div class="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-sm">RERA</div>
                    <div>
                        <div class="font-medium text-gray-800">RERA Act 2016</div>
                        <div class="text-xs text-gray-500">Central • 224 sections</div>
                    </div>
                </div>
                <div class="flex items-center gap-3 p-4 bg-white/50 rounded-xl">
                    <div class="w-10 h-10 rounded-lg bg-purple-100 flex items-center justify-center text-purple-600 font-bold text-sm">MH</div>
                    <div>
                        <div class="font-medium text-gray-800">Maharashtra Rules 2017</div>
                        <div class="text-xs text-gray-500">Maharashtra • 214 sections</div>
                    </div>
                </div>
                <div class="flex items-center gap-3 p-4 bg-white/50 rounded-xl">
                    <div class="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center text-green-600 font-bold text-sm">UP</div>
                    <div>
                        <div class="font-medium text-gray-800">UP RERA Rules 2016</div>
                        <div class="text-xs text-gray-500">Uttar Pradesh</div>
                    </div>
                </div>
                <div class="flex items-center gap-3 p-4 bg-white/50 rounded-xl">
                    <div class="w-10 h-10 rounded-lg bg-amber-100 flex items-center justify-center text-amber-600 font-bold text-sm">KA</div>
                    <div>
                        <div class="font-medium text-gray-800">Karnataka Rules 2017</div>
                        <div class="text-xs text-gray-500">Karnataka</div>
                    </div>
                </div>
                <div class="flex items-center gap-3 p-4 bg-white/50 rounded-xl">
                    <div class="w-10 h-10 rounded-lg bg-red-100 flex items-center justify-center text-red-600 font-bold text-sm">TN</div>
                    <div>
                        <div class="font-medium text-gray-800">Tamil Nadu Rules 2017</div>
                        <div class="text-xs text-gray-500">Tamil Nadu</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer class="mt-16 border-t border-gray-200 bg-white/50">
        <div class="max-w-6xl mx-auto px-6 py-8">
            <div class="flex flex-col md:flex-row items-center justify-between gap-4">
                <div class="text-sm text-gray-500">
                    <span class="font-semibold text-gray-700">CivicSetu</span> — Open-source RAG for Indian civic documents
                </div>
                <div class="flex items-center gap-6 text-sm text-gray-500">
                    <a href="https://github.com/adeshboudh/civicsetu" class="hover:text-gray-700 transition-colors">GitHub</a>
                    <a href="/docs" class="hover:text-gray-700 transition-colors">API Docs</a>
                </div>
            </div>
        </div>
    </footer>

    <script>
        function setQuery(text) {
            document.getElementById('query').value = text;
            document.getElementById('query').focus();
        }

        async function doQuery() {
            const queryText = document.getElementById('query').value.trim();
            if (!queryText) return;

            const jurisdiction = document.getElementById('jurisdiction').value;
            const responseDiv = document.getElementById('response');

            responseDiv.classList.remove('hidden');
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('answer-card').classList.add('hidden');
            document.getElementById('stats-row').classList.add('hidden');
            document.getElementById('citations-card').classList.add('hidden');
            document.getElementById('warnings').classList.add('hidden');

            const payload = { query: queryText, top_k: 5 };
            if (jurisdiction) payload.jurisdiction_filter = jurisdiction;

            try {
                const res = await fetch('/api/v1/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                document.getElementById('loading').classList.add('hidden');
                document.getElementById('answer-card').classList.remove('hidden');
                document.getElementById('answer').innerHTML = (data.answer || '').replace(/\\n/g, '<br>');
                document.getElementById('query-type-badge').textContent = data.query_type_resolved || 'general';

                const warningsDiv = document.getElementById('warnings');
                let warningsHtml = '';
                if (data.conflict_warnings && data.conflict_warnings.length) {
                    warningsHtml += '<div class="mb-3"><strong class="text-amber-700">Conflict Detected:</strong> ' + data.conflict_warnings.join(', ') + '</div>';
                }
                if (data.amendment_notice) {
                    warningsHtml += '<div class="mb-3"><strong class="text-blue-700">Amendment Notice:</strong> ' + data.amendment_notice + '</div>';
                }
                if (warningsHtml) {
                    warningsDiv.innerHTML = '<div class="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm">' + warningsHtml + '</div>';
                    warningsDiv.classList.remove('hidden');
                }

                document.getElementById('stats-row').classList.remove('hidden');
                document.getElementById('confidence-value').textContent = ((data.confidence_score || 0) * 100).toFixed(0) + '%';
                document.getElementById('citations-count').textContent = data.citations ? data.citations.length : 0;

                const citationsCard = document.getElementById('citations-card');
                const citationsDiv = document.getElementById('citations');
                if (data.citations && data.citations.length) {
                    citationsCard.classList.remove('hidden');
                    citationsDiv.innerHTML = data.citations.map(c => `
                        <div class="citation-card bg-white rounded-xl p-4 border-l-4 border-blue-500 shadow-sm">
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="font-mono text-sm text-blue-600 font-semibold">${c.section_id}</div>
                                    <div class="font-medium text-gray-800">${c.doc_name}</div>
                                </div>
                                <span class="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">${c.jurisdiction}</span>
                            </div>
                            <div class="text-xs text-gray-500 mt-2">Effective: ${c.effective_date || 'N/A'}</div>
                        </div>
                    `).join('');
                }

            } catch (error) {
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('answer-card').classList.remove('hidden');
                document.getElementById('answer').innerHTML = '<div class="text-red-600 p-4 bg-red-50 rounded-xl"><strong>Error:</strong> ' + error.message + '</div>';
            }
        }

        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') doQuery();
        });
    </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    log.info("civicsetu_starting", env=settings.api_env)

    # Pre-compile the graph once at startup — not on first request
    from civicsetu.agent.graph import get_compiled_graph
    app.state.graph = get_compiled_graph()
    log.info("langgraph_compiled")

    yield

    log.info("civicsetu_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CivicSetu API",
        description="RAG system for Indian civic and legal documents",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/", include_in_schema=False)
    async def landing_page():
        return HTMLResponse(get_landing_page_html())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)

    app.include_router(health.router, tags=["health"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])

    return app


app = create_app()
