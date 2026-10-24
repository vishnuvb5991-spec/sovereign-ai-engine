// Preset sample congressional bills for instant testing
const PRESETS = {
    clean_energy: {
        title: "Clean Energy Innovation Act (H.R. 4821)",
        text: `SECTION 1. SHORT TITLE.
This Act may be cited as the 'Clean Energy Grid Innovation Act of 2026'.

SEC. 2. ESTABLISHMENT OF ADVANCED BATTERY STORAGE GRANT PROGRAM.
(a) In General.—The Secretary of Energy shall establish a competitive grant program to award funds to eligible entities for the deployment of advanced utility-scale energy storage and microgrid resilience systems.
(b) Priority.—In awarding grants under this section, the Secretary shall give priority to projects that—
(1) improve electrical grid reliability in regions experiencing frequent severe weather events;
(2) utilize non-lithium chemical formulations or long-duration gravity systems; and
(3) incorporate end-of-life battery recycling and domestic mineral extraction safeguards.

SEC. 3. AUTHORIZATION OF APPROPRIATIONS.
There is authorized to be appropriated to carry out this section $450,000,000 for each of fiscal years 2027 through 2031.`
    },
    ai_privacy: {
        title: "Federal Artificial Intelligence Privacy & Governance Act (S. 1104)",
        text: `SECTION 1. SHORT TITLE.
This Act may be cited as the 'Federal AI Algorithmic Transparency and Data Sovereignty Act'.

SEC. 2. REQUIREMENTS FOR HIGH-RISK FOUNDATION MODELS.
(a) Pre-Deployment Auditing.—Any entity deploying an automated decision system across healthcare, credit scoring, or criminal justice shall submit to an independent algorithmic audit certifying non-discrimination.
(b) Data Sovereignty Safeguards.—Entities maintaining sovereign citizen data must ensure personal biometric identifiers are processed on dedicated domestic infrastructure without exfiltration to third-party offshore endpoints.
(c) Civil Penalties.—Violations shall be enforceable by the Federal Trade Commission with fines not to exceed $45,000 per violation.`
    },
    healthcare: {
        title: "Rural Telehealth & Medical Access Act (H.R. 2910)",
        text: `SECTION 1. SHORT TITLE.
This Act may be cited as the 'Rural Healthcare Telehealth Connectivity Act'.

SEC. 2. EXPANSION OF BROADBAND CONNECTIVITY SUBSIDIES FOR COMMUNITY HEALTH CENTERS.
(a) Grants.—The Secretary of Health and Human Services shall provide low-interest loans and technical assistance to critical-access hospitals to install gigabit fiber networks and satellite telemetry backup.
(b) Metric Reporting.—Participating centers shall submit bi-annual quality reports detailing reduction in emergency transfer times for acute cardiology patients.`
    }
};

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-cyan-500', 'text-cyan-400');
        btn.classList.add('border-transparent', 'text-slate-400');
    });

    const activeTab = document.getElementById(tabId);
    if (activeTab) activeTab.classList.remove('hidden');

    const activeBtn = document.getElementById(`btn-${tabId}`);
    if (activeBtn) {
        activeBtn.classList.add('border-cyan-500', 'text-cyan-400');
        activeBtn.classList.remove('border-transparent', 'text-slate-400');
    }

    if (tabId === 'analytics-tab') {
        initCharts();
    }
}

// Load Preset Bills
function loadPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;

    // Load into Studio
    const studioInput = document.getElementById('studio-text');
    if (studioInput) studioInput.value = preset.text;

    // Load into Arena
    const arenaInput = document.getElementById('arena-text');
    if (arenaInput) arenaInput.value = preset.text;
}

// Studio: Run Single Summarization
async function runStudioSummarize() {
    const text = document.getElementById('studio-text').value.trim();
    if (!text) {
        alert("Please enter or select a congressional bill first.");
        return;
    }

    const provider = document.getElementById('studio-provider').value;
    const btn = document.getElementById('btn-studio-run');
    const outputContainer = document.getElementById('studio-output');
    const latencyBadge = document.getElementById('studio-latency');
    const tokensBadge = document.getElementById('studio-tokens');

    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-2">🔄</span> Synthesizing Summary...`;
    outputContainer.innerHTML = `<div class="text-slate-400 italic animate-pulse">Running ${provider} model inference...</div>`;

    try {
        const res = await fetch('/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                provider: provider,
                max_tokens: 512,
                temperature: 0.2
            })
        });

        const data = await res.json();
        outputContainer.innerText = data.summary;
        latencyBadge.innerText = `${data.latency_ms} ms`;
        tokensBadge.innerText = `${data.output_tokens} output tokens`;
    } catch (e) {
        outputContainer.innerHTML = `<div class="text-rose-400">Error connecting to model: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = `⚡ Generate Legislative Summary`;
    }
}

// Arena: Run Side-by-Side Battle
async function runArenaBattle() {
    const text = document.getElementById('arena-text').value.trim();
    if (!text) {
        alert("Please enter bill text for the arena battle.");
        return;
    }

    const btn = document.getElementById('btn-arena-run');
    const sovOutput = document.getElementById('arena-sov-output');
    const gemOutput = document.getElementById('arena-gem-output');
    const battleVerdict = document.getElementById('arena-verdict');

    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-2">⚔️</span> Fighting...`;

    sovOutput.innerHTML = `<div class="text-cyan-400 animate-pulse italic">Generating via Sovereign QLoRA...</div>`;
    gemOutput.innerHTML = `<div class="text-blue-400 animate-pulse italic">Generating via Gemini 3.8 Flash...</div>`;
    battleVerdict.classList.add('hidden');

    try {
        const res = await fetch('/api/arena', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await res.json();

        // Render Sovereign
        sovOutput.innerText = data.sovereign.summary;
        document.getElementById('arena-sov-lat').innerText = `${data.sovereign.latency_ms} ms`;
        document.getElementById('arena-sov-tokens').innerText = `${data.sovereign.output_tokens} tokens`;

        // Render Frontier
        gemOutput.innerText = data.frontier.summary;
        document.getElementById('arena-gem-lat').innerText = `${data.frontier.latency_ms} ms`;
        document.getElementById('arena-gem-tokens').innerText = `${data.frontier.output_tokens} tokens`;

        // Verdict banner
        battleVerdict.innerText = `🏆 Result: ${data.winner_summary} | Zero Cloud Exfiltration on Sovereign side.`;
        battleVerdict.classList.remove('hidden');
    } catch (e) {
        battleVerdict.innerText = `Battle execution error: ${e.message}`;
        battleVerdict.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `⚔️ Launch Side-by-Side Arena Battle`;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadPreset('clean_energy');
    initCharts();
});
