let rougeChart = null;
let latencyChart = null;

async function initCharts() {
    try {
        const res = await fetch('/api/benchmark/stats');
        const data = await res.json();
        
        renderRougeChart(data.rouge_scores);
        renderLatencyChart(data.latency_percentiles);
        setupCrossoverCalculator(data.cost_crossover);
    } catch (e) {
        console.error("Could not load benchmark stats:", e);
    }
}

function renderRougeChart(scores) {
    const ctx = document.getElementById('rougeChart');
    if (!ctx) return;

    if (rougeChart) rougeChart.destroy();

    rougeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['ROUGE-1 (Unigram)', 'ROUGE-2 (Bigram)', 'ROUGE-L (Fluency)'],
            datasets: [
                {
                    label: 'Sovereign 7B (QLoRA Fine-Tuned)',
                    data: [scores.sovereign_qlora.rouge1, scores.sovereign_qlora.rouge2, scores.sovereign_qlora.rougeL],
                    backgroundColor: 'rgba(6, 182, 212, 0.8)',
                    borderColor: '#06b6d4',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'Frontier API (Gemini 3.8 Flash)',
                    data: [scores.gemini_3_8_flash.rouge1, scores.gemini_3_8_flash.rouge2, scores.gemini_3_8_flash.rougeL],
                    backgroundColor: 'rgba(59, 130, 246, 0.8)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'Zero-Shot Base Model (No Fine-Tuning)',
                    data: [scores.zero_shot_base.rouge1, scores.zero_shot_base.rouge2, scores.zero_shot_base.rougeL],
                    backgroundColor: 'rgba(100, 116, 139, 0.4)',
                    borderColor: '#64748b',
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#cbd5e1' } },
                tooltip: { backgroundColor: '#0f172a' }
            },
            scales: {
                y: {
                    min: 0,
                    max: 0.6,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function renderLatencyChart(latencies) {
    const ctx = document.getElementById('latencyChart');
    if (!ctx) return;

    if (latencyChart) latencyChart.destroy();

    latencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['P50 (Median)', 'P90', 'P95 (Tail)', 'P99'],
            datasets: [
                {
                    label: 'Sovereign Engine (vLLM Continuous Batching)',
                    data: [
                        latencies.sovereign_vllm.p50,
                        latencies.sovereign_vllm.p90,
                        latencies.sovereign_vllm.p95,
                        latencies.sovereign_vllm.p99
                    ],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 5,
                    pointBackgroundColor: '#10b981'
                },
                {
                    label: 'Frontier API (Gemini 3.8 Flash Network)',
                    data: [
                        latencies.gemini_api.p50,
                        latencies.gemini_api.p90,
                        latencies.gemini_api.p95,
                        latencies.gemini_api.p99
                    ],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.05)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 5,
                    pointBackgroundColor: '#f59e0b'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#cbd5e1' } },
                tooltip: { backgroundColor: '#0f172a' }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: {
                        color: '#94a3b8',
                        callback: (v) => v + ' ms'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function setupCrossoverCalculator(costData) {
    const slider = document.getElementById('volumeSlider');
    const volumeDisplay = document.getElementById('volumeDisplay');
    const apiCostDisplay = document.getElementById('apiCostDisplay');
    const sovereignCostDisplay = document.getElementById('sovereignCostDisplay');
    const savingsDisplay = document.getElementById('savingsDisplay');
    const crossoverBadge = document.getElementById('crossoverBadge');

    if (!slider) return;

    function update() {
        const calls = parseInt(slider.value);
        volumeDisplay.innerText = calls.toLocaleString() + ' calls / month';

        // Cost formulas
        const frontierCost = (calls / 1000) * costData.frontier_cost_per_1k_usd;
        const gpuCost = costData.gpu_monthly_fixed_cost_usd;

        apiCostDisplay.innerText = '$' + frontierCost.toLocaleString(undefined, { maximumFractionDigits: 0 });
        sovereignCostDisplay.innerText = '$' + gpuCost.toLocaleString();

        const diff = frontierCost - gpuCost;
        if (diff > 0) {
            savingsDisplay.innerText = '+$' + diff.toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' Saved / mo';
            savingsDisplay.className = 'text-xl font-bold text-emerald-400';
            crossoverBadge.innerText = '✅ Sovereign AI is Cheaper at this volume';
            crossoverBadge.className = 'text-xs px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-semibold border border-emerald-500/30';
        } else {
            savingsDisplay.innerText = '-$' + Math.abs(diff).toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' / mo';
            savingsDisplay.className = 'text-xl font-bold text-amber-400';
            crossoverBadge.innerText = '⚠️ Frontier API is Cheaper at low volume';
            crossoverBadge.className = 'text-xs px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-400 font-semibold border border-amber-500/30';
        }
    }

    slider.addEventListener('input', update);
    update();
}
