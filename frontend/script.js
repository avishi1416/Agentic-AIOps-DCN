const API_BASE = '/api';

// Elements
const runBtn = document.getElementById('run-btn');
const statusIndicator = document.getElementById('system-status');
const statServices = document.getElementById('stat-services');
const statAnomalies = document.getElementById('stat-anomalies');
const statRemediations = document.getElementById('stat-remediations');
const statLoss = document.getElementById('stat-loss');
const servicesContainer = document.getElementById('services-container');
const incidentsContainer = document.getElementById('incidents-container');

// Modal Elements
const modalOverlay = document.getElementById('remediation-modal');
const modalServiceId = document.getElementById('modal-service-id');
const btnCancelRemediation = document.getElementById('cancel-remediation');
const btnConfirmRemediation = document.getElementById('confirm-remediation');
const remediationResult = document.getElementById('remediation-result');

let currentServiceId = null;

// Initialize
async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        
        if (data.pipeline_initialized) {
            updateStatus(true);
            refreshDashboard();
        } else {
            updateStatus(false);
        }
    } catch (e) {
        console.error("Backend not reachable", e);
        updateStatus(false, "Backend Offline");
    }
}

function updateStatus(isReady, text = null) {
    statusIndicator.innerHTML = `<span class="dot ${isReady ? 'green' : 'red'}"></span> ${text || (isReady ? 'System Active' : 'Not Initialized')}`;
}

// Run Simulation
runBtn.addEventListener('click', async () => {
    runBtn.disabled = true;
    runBtn.innerHTML = 'Running...';
    
    try {
        const res = await fetch(`${API_BASE}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_services: 5, duration_seconds: 3600 })
        });
        
        if (res.ok) {
            updateStatus(true);
            await refreshDashboard();
        }
    } catch (e) {
        alert("Failed to run simulation");
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Re-run Simulation`;
    }
});

// Refresh Dashboard Data
async function refreshDashboard() {
    try {
        const res = await fetch(`${API_BASE}/pipeline/state`);
        const data = await res.json();
        
        updateSummary(data.pipeline_results);
        renderServices(data.service_states);
        renderIncidents(data.incidents);
        
    } catch (e) {
        console.error("Failed to load pipeline state", e);
    }
}

function updateSummary(results) {
    statServices.innerText = results.num_services || 0;
    statAnomalies.innerText = results.total_anomalies_detected || 0;
    statRemediations.innerText = results.total_remediations || 0;
    statLoss.innerText = (results.avg_cumulative_loss || 0).toFixed(2);
}

function renderServices(services) {
    servicesContainer.innerHTML = '';
    
    for (const [svcId, state] of Object.entries(services)) {
        const card = document.createElement('div');
        card.className = 'service-card';
        card.innerHTML = `
            <div class="service-header">
                <h3>${svcId}</h3>
                <button class="action-btn" onclick="openRemediation('${svcId}')">Remediate</button>
            </div>
            <div class="service-metrics">
                <div class="metric"><span>Avg Severity:</span> <strong>${state.avg_severity.toFixed(3)}</strong></div>
                <div class="metric"><span>Avg Failure Prob:</span> <strong>${state.avg_failure_prob.toFixed(3)}</strong></div>
                <div class="metric"><span>Stability:</span> <strong>${state.avg_stability.toFixed(3)}</strong></div>
                <div class="metric"><span>Loss:</span> <strong>${state.cumulative_loss.toFixed(3)}</strong></div>
                <div class="metric"><span>Anomalies:</span> <strong>${state.num_anomalies}</strong></div>
            </div>
        `;
        servicesContainer.appendChild(card);
    }
}

function renderIncidents(incidents) {
    if (!incidents || incidents.length === 0) {
        incidentsContainer.innerHTML = '<div class="empty-state">No incidents recorded.</div>';
        return;
    }
    
    incidentsContainer.innerHTML = incidents.slice(-5).reverse().map(inc => `
        <div class="anomaly-card ${inc.severity >= 0.8 ? 'critical' : 'warning'}">
            <strong>${inc.service}</strong> - ${inc.type}
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">
                Severity: ${inc.severity.toFixed(2)} | Time: ${inc.start.toFixed(0)}s
            </div>
        </div>
    `).join('');
}

// Remediation Flow
window.openRemediation = function(serviceId) {
    currentServiceId = serviceId;
    modalServiceId.innerText = serviceId;
    remediationResult.className = 'remediation-result hidden';
    modalOverlay.classList.add('active');
}

btnCancelRemediation.addEventListener('click', () => {
    modalOverlay.classList.remove('active');
});

btnConfirmRemediation.addEventListener('click', async () => {
    const selectedAction = document.querySelector('input[name="action"]:checked').value;
    
    btnConfirmRemediation.disabled = true;
    btnConfirmRemediation.innerText = 'Executing...';
    
    try {
        const body = { service_id: currentServiceId };
        if (selectedAction) body.action = selectedAction;
        
        const res = await fetch(`${API_BASE}/remediate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        
        const data = await res.json();
        
        remediationResult.classList.remove('hidden');
        if (data.result.error) {
            remediationResult.innerHTML = `<strong>Error:</strong> ${data.result.error}`;
            remediationResult.style.borderColor = 'var(--danger)';
            remediationResult.style.background = 'rgba(239, 68, 68, 0.1)';
        } else {
            remediationResult.innerHTML = `
                <strong>Action Taken:</strong> ${data.result.action}<br>
                <strong>Success:</strong> ${data.result.success ? 'Yes' : 'No'}<br>
                <strong>Loss Drop:</strong> ${data.result.loss_before.toFixed(2)} → ${data.result.loss_after.toFixed(2)}
            `;
            remediationResult.style.borderColor = 'var(--success)';
            remediationResult.style.background = 'rgba(16, 185, 129, 0.1)';
            
            // Refresh to get new metrics
            setTimeout(refreshDashboard, 1500);
        }
        
    } catch (e) {
        alert("Remediation request failed.");
    } finally {
        btnConfirmRemediation.disabled = false;
        btnConfirmRemediation.innerText = 'Execute';
    }
});

// Boot
checkStatus();
