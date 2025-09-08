document.addEventListener('DOMContentLoaded', function() {
    // Initialize health score circles
    document.querySelectorAll('.score-circle').forEach(circle => {
        const score = circle.dataset.score;
        circle.style.setProperty('--score', score + '%');
        circle.style.setProperty('--final-score', score + '%');
        
        // Add animation
        circle.style.animation = 'fillCircle 1s ease-out forwards';
    });
    
    // Initialize charts (Apex candlesticks with overlays; fallback to Chart.js if needed)
    document.querySelectorAll('[id^="chart-"]').forEach(node => {
        const series = node.dataset.series ? JSON.parse(node.dataset.series) : null;
        if (series && series.candles && window.ApexCharts) {
            initializeCandle(node, series);
            // Remove skeleton in the same container once chart renders
            const container = node.closest('.chart-container');
            const removeSkeleton = () => {
                const sk = container?.querySelector('.skeleton');
                if (sk) sk.remove();
            };
            // In case render is async, attempt delayed cleanup
            setTimeout(removeSkeleton, 300);
        } else if (series && series.labels && series.close) {
            initializeChart(node, series);
        }
    });

    // Initialize mini sparklines on reversal lists
    document.querySelectorAll('.rev-sparkline').forEach(node => {
        if (!window.ApexCharts) return;
        let series;
        try { series = JSON.parse(node.dataset.series || '[]'); } catch { series = []; }
        if (!series || series.length < 3) return;
        // Use list side color for consistency instead of slope
        const parentList = node.closest('.rev-list');
        const isBull = parentList && parentList.classList.contains('rev-bull');
        const color = isBull ? getComputedStyle(document.documentElement).getPropertyValue('--bull').trim() : getComputedStyle(document.documentElement).getPropertyValue('--bear').trim();
        const options = {
            chart: { type: 'line', height: 36, width: 130, sparkline: { enabled: true } },
            series: [{ data: series }],
            stroke: { width: 1.6, curve: 'smooth', colors: [color] },
            tooltip: { enabled: false },
            grid: { padding: { top: 0, bottom: 0, left: 0, right: 0 } }
        };
        const chart = new ApexCharts(node, options);
        chart.render().then(() => {
            node.classList.add('rendered');
        });
    });

    // Initialize mini sparklines on picks lists
    document.querySelectorAll('.pck-sparkline').forEach(node => {
        if (!window.ApexCharts) return;
        let series;
        try { series = JSON.parse(node.dataset.series || '[]'); } catch { series = []; }
        if (!series || series.length < 3) return;
        const up = series[series.length - 1] >= series[0];
        const options = {
            chart: { type: 'line', height: 36, width: 130, sparkline: { enabled: true } },
            series: [{ data: series }],
            stroke: { width: 1.6, curve: 'smooth', colors: [up ? '#22C55E' : '#EF4444'] },
            tooltip: { enabled: false },
            grid: { padding: { top: 0, bottom: 0, left: 0, right: 0 } }
        };
        const chart = new ApexCharts(node, options);
        chart.render().then(() => node.classList.add('rendered'));
    });

    // Picks filters + URL prefill
    const pMin = document.getElementById('pckMinScore');
    const pMinVal = document.getElementById('pckMinScoreVal');
    const pBuy = document.getElementById('pckShowBuy');
    const pSell = document.getElementById('pckShowSell');
    const pSigBtns = Array.from(document.querySelectorAll('.pck-sig-toggle'));
    const pReset = document.getElementById('pckResetFilters');
    const params = new URLSearchParams(window.location.search);
    const onPicksPage = !!document.getElementById('picks');

    if (onPicksPage) {
        const act = (params.get('action') || '').toLowerCase();
        if (act === 'buy') { if (pSell) pSell.checked = false; if (pBuy) pBuy.checked = true; }
        if (act === 'sell') { if (pBuy) pBuy.checked = false; if (pSell) pSell.checked = true; }
        const minS = params.get('minScore');
        if (minS && pMin) { pMin.value = String(parseInt(minS,10) || 0); if (pMinVal) pMinVal.textContent = pMin.value; }
    }

    function pGetSignals() { return pSigBtns.filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.signal.toLowerCase()); }
    function applyPicksFilters() {
        const minScore = parseInt(pMin?.value || '0', 10) || 0;
        const sigs = pGetSignals();
        document.querySelectorAll('#picks .rev-item').forEach(item => {
            const action = item.getAttribute('data-action');
            const score = parseInt(item.getAttribute('data-score') || '0', 10) || 0;
            const reasons = (item.getAttribute('data-signals') || '').toLowerCase();
            if ((action === 'buy' && !pBuy?.checked) || (action === 'sell' && !pSell?.checked)) { item.classList.add('d-none'); return; }
            if (score < minScore) { item.classList.add('d-none'); return; }
            if (sigs.length) {
                const hit = sigs.some(s => reasons.includes(s));
                if (!hit) { item.classList.add('d-none'); return; }
            }
            item.classList.remove('d-none');
        });
        if (pMinVal) pMinVal.textContent = String(minScore);
    }
    if (pMin) { pMin.addEventListener('input', applyPicksFilters); pMinVal.textContent = String(pMin.value); }
    pSigBtns.forEach(b => b.addEventListener('click', () => { const p = b.getAttribute('aria-pressed') === 'true'; b.setAttribute('aria-pressed', (!p).toString()); b.classList.toggle('active', !p); applyPicksFilters(); }));
    pBuy?.addEventListener('change', applyPicksFilters);
    pSell?.addEventListener('change', applyPicksFilters);
    pReset?.addEventListener('click', () => { if (pMin) { pMin.value = '0'; pMinVal.textContent = '0'; } if (pBuy) pBuy.checked = true; if (pSell) pSell.checked = true; pSigBtns.forEach(b => { b.setAttribute('aria-pressed','false'); b.classList.remove('active'); }); applyPicksFilters(); });
    applyPicksFilters();

    // Toggle LLM rationales on picks page
    const toggleWhy = document.getElementById('toggleRationales');
    if (toggleWhy && document.getElementById('picks')) {
        toggleWhy.addEventListener('click', () => {
            const container = document.getElementById('picks');
            container.classList.toggle('show-rationales');
        });
    }

    // Quick alert from Movers list
    document.querySelectorAll('.mvr-alert').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const symbol = btn.getAttribute('data-symbol');
            const price = parseFloat(btn.getAttribute('data-price') || '0');
            const dir = btn.getAttribute('data-dir');
            if (!symbol || !price) return;
            const def = dir === 'up' ? (price * 1.02).toFixed(2) : (price * 0.98).toFixed(2);
            const input = prompt(`Create ${dir === 'up' ? 'above' : 'below'} alert for ${symbol}. Trigger price ₹`, def);
            const trigger = parseFloat(input || '0');
            if (!trigger || trigger <= 0) return;
            try {
                await fetch('/api/alerts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, condition: dir === 'up' ? 'above' : 'below', price: trigger })
                });
                // Toast success
                const container = document.getElementById('toastContainer');
                if (container) {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'toast align-items-center text-white bg-success border-0';
                    wrapper.innerHTML = `<div class="d-flex"><div class="toast-body">Alert set for ${symbol} at ₹${trigger}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div>`;
                    container.appendChild(wrapper);
                    new bootstrap.Toast(wrapper, { delay: 2500 }).show();
                }
            } catch (err) { console.error(err); }
        });
    });

    // Reversal filters (score, signals, direction)
    const minScoreInput = document.getElementById('revMinScore');
    const minScoreVal = document.getElementById('revMinScoreVal');
    const sigButtons = Array.from(document.querySelectorAll('.rev-sig-toggle'));
    const showBull = document.getElementById('revShowBull');
    const showBear = document.getElementById('revShowBear');
    const resetBtn = document.getElementById('revResetFilters');
    const bullCount = document.getElementById('revBullCount');
    const bearCount = document.getElementById('revBearCount');

    function getActiveSignals() {
        return sigButtons.filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.signal);
    }

    function applyFilters() {
        const minScore = parseInt(minScoreInput?.value || '0', 10) || 0;
        const sigs = getActiveSignals();
        let bullVisible = 0, bearVisible = 0;
        document.querySelectorAll('.rev-item').forEach(item => {
            const dir = item.getAttribute('data-dir');
            const score = parseInt(item.getAttribute('data-score') || '0', 10) || 0;
            const reasons = (item.getAttribute('data-reasons') || '').toLowerCase();
            // Direction filter
            if ((dir === 'bull' && !showBull.checked) || (dir === 'bear' && !showBear.checked)) {
                item.classList.add('d-none');
                return;
            }
            // Score filter
            if (score < minScore) { item.classList.add('d-none'); return; }
            // Signals (any-of)
            if (sigs.length) {
                const hit = sigs.some(s => reasons.includes(s.toLowerCase()));
                if (!hit) { item.classList.add('d-none'); return; }
            }
            item.classList.remove('d-none');
            if (dir === 'bull') bullVisible++; else if (dir === 'bear') bearVisible++;
        });
        if (bullCount) bullCount.textContent = String(bullVisible);
        if (bearCount) bearCount.textContent = String(bearVisible);
        if (minScoreVal) minScoreVal.textContent = String(minScore);
    }

    if (minScoreInput) {
        minScoreInput.addEventListener('input', applyFilters);
        if (minScoreVal) minScoreVal.textContent = String(minScoreInput.value);
    }
    sigButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const pressed = btn.getAttribute('aria-pressed') === 'true';
            btn.setAttribute('aria-pressed', (!pressed).toString());
            btn.classList.toggle('active', !pressed);
            applyFilters();
        });
    });
    showBull?.addEventListener('change', applyFilters);
    showBear?.addEventListener('change', applyFilters);
    resetBtn?.addEventListener('click', () => {
        if (minScoreInput) { minScoreInput.value = '0'; if (minScoreVal) minScoreVal.textContent = '0'; }
        sigButtons.forEach(b => { b.setAttribute('aria-pressed','false'); b.classList.remove('active'); });
        if (showBull) showBull.checked = true; if (showBear) showBear.checked = true;
        applyFilters();
    });

    // Initial filter pass to compute visible counts
    applyFilters();
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize form submission with loading state
    const form = document.getElementById('analysis-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const submitButton = form.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.innerHTML = `
                <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                Analyzing...
            `;
        });
    }

    // Initialize price range progress bars
    document.querySelectorAll('.price-progress').forEach(container => {
        const progressBar = container.querySelector('.progress-bar');
        if (progressBar) {
            setTimeout(() => {
                const width = progressBar.getAttribute('aria-valuenow');
                progressBar.style.width = width + '%';
            }, 100);
        }
    });

    // Toast helper
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const wrapper = document.createElement('div');
        wrapper.className = `toast align-items-center text-white bg-${type} border-0`;
        wrapper.setAttribute('role', 'alert');
        wrapper.setAttribute('aria-live', 'assertive');
        wrapper.setAttribute('aria-atomic', 'true');
        wrapper.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>`;
        container.appendChild(wrapper);
        const toast = new bootstrap.Toast(wrapper, { delay: 2500 });
        toast.show();
        wrapper.addEventListener('hidden.bs.toast', () => wrapper.remove());
    }

    // Watchlist toggle with API + local state
    document.querySelectorAll('[data-action="watch"]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const symbol = btn.getAttribute('data-symbol');
            if (!symbol) return;
            const watched = btn.getAttribute('data-watched') === 'true';
            const action = watched ? 'remove' : 'add';
            try {
                await fetch('/api/watchlist', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, action })
                });
            } catch {}
            btn.setAttribute('data-watched', (!watched).toString());
            btn.classList.toggle('btn-primary', !watched);
            btn.classList.toggle('btn-outline-primary', watched);
            showToast(`${symbol} ${watched ? 'removed from' : 'added to'} watchlist`, 'primary');
        });
    });

    // Modal prefill: Alert
    const alertModal = document.getElementById('alertModal');
    if (alertModal) {
        alertModal.addEventListener('show.bs.modal', (event) => {
            const btn = event.relatedTarget;
            const symbol = btn?.getAttribute('data-symbol') || '';
            const price = btn?.getAttribute('data-price') || '';
            alertModal.querySelector('#alertSymbol').value = symbol;
            alertModal.querySelector('#alertPrice').value = price;
        });
        document.getElementById('saveAlertBtn')?.addEventListener('click', async () => {
            const symbol = alertModal.querySelector('#alertSymbol').value;
            const condition = alertModal.querySelector('#alertCondition').value;
            const price = parseFloat(alertModal.querySelector('#alertPrice').value || '0');
            try {
                await fetch('/api/alerts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, condition, price })
                });
                showToast(`Alert saved for ${symbol} at ₹${price}`, 'success');
                bootstrap.Modal.getInstance(alertModal)?.hide();
            } catch (e) {
                showToast('Failed to save alert', 'danger');
            }
        });
    }

    // Modal prefill + compute: Position Size
    const posModal = document.getElementById('positionModal');
    if (posModal) {
        posModal.addEventListener('show.bs.modal', (event) => {
            const btn = event.relatedTarget;
            const symbol = btn?.getAttribute('data-symbol') || '';
            const price = btn?.getAttribute('data-price') || '';
            const stop = btn?.getAttribute('data-stop') || '';
            posModal.querySelector('#posSymbol').value = symbol;
            posModal.querySelector('#posPrice').value = price;
            posModal.querySelector('#posStop').value = stop;
            posModal.querySelector('#posResult').value = '';
        });
        document.getElementById('calcPositionBtn')?.addEventListener('click', () => {
            const price = parseFloat(posModal.querySelector('#posPrice').value || '0');
            const stop = parseFloat(posModal.querySelector('#posStop').value || '0');
            const account = parseFloat(posModal.querySelector('#posAccount').value || '0');
            const riskPct = parseFloat(posModal.querySelector('#posRisk').value || '0');
            const riskAmt = account * (riskPct / 100);
            const perShareRisk = Math.abs(price - stop);
            const qty = perShareRisk > 0 ? Math.floor(riskAmt / perShareRisk) : 0;
            posModal.querySelector('#posResult').value = String(qty);
        });
    }
});

function initializeChart(canvas, series) {
    const ctx = canvas.getContext('2d');
    const data = {
        labels: series.labels,
        datasets: [
            // Bollinger band lower (hidden line)
            {
                label: 'BB Lower',
                data: series.bb_lower || [],
                borderColor: 'rgba(0,0,0,0)',
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                pointRadius: 0,
                borderWidth: 0,
                fill: false,
                order: 1
            },
            // Bollinger band upper (fills to previous)
            {
                label: 'BB Upper',
                data: series.bb_upper || [],
                borderColor: 'rgba(0,0,0,0)',
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                pointRadius: 0,
                borderWidth: 0,
                fill: '-1',
                order: 1
            },
            // Close price
            {
                label: 'Close',
                data: series.close,
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.05)',
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.25,
                order: 2
            },
            // SMA20 overlay
            {
                label: 'SMA20',
                data: series.sma20 || [],
                borderColor: '#22C55E',
                backgroundColor: 'transparent',
                borderDash: [3, 3],
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.25,
                order: 2
            }
        ]
    };

    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: false
                }
            },
            elements: {
                line: {
                    tension: 0.25
                }
            }
        }
    });
}

function initializeCandle(node, series) {
    // Build ApexCharts candlestick with SMA20 and Bollinger overlays
    const bounds = (() => {
        try {
            const lows = [], highs = [];
            (series.candles || []).forEach(c => {
                const y = c.y || [];
                if (Array.isArray(y) && y.length === 4) {
                    highs.push(Number(y[1]));
                    lows.push(Number(y[2]));
                }
            });
            // Include S/R levels so their annotations are visible
            const sr = [series.s1, series.s2, series.r1, series.r2]
                .map(v => Number(v))
                .filter(v => Number.isFinite(v) && v > 0);
            if (!highs.length && !sr.length) return null;
            const min = Math.min(...(lows.length ? lows : sr));
            const max = Math.max(...(highs.length ? highs : sr));
            const pad = Math.max((max - min) * 0.06, max * 0.003);
            return { min: min - pad, max: max + pad };
        } catch { return null; }
    })();
    const volMax = Math.max(1, ...((series.volume || []).map(p => Number(p.y) || 0)));

    const options = {
        chart: {
            type: 'candlestick',
            height: '100%',
            toolbar: { show: false },
            animations: { enabled: false },
            background: 'transparent',
            sparkline: { enabled: true },
            parentHeightOffset: 0,
            offsetX: 0,
            offsetY: 0
        },
        series: [
            { name: 'Price', type: 'candlestick', data: series.candles || [] },
            { name: 'Volume', type: 'column', data: series.volume || [], yAxisIndex: 1 },
            { name: 'SMA20', type: 'line', data: series.sma20 || [], stroke: { width: 1.5, curve: 'smooth' } },
            { name: 'BB Upper', type: 'line', data: series.bb_upper || [], stroke: { width: 1, dashArray: 3 } },
            { name: 'BB Lower', type: 'line', data: series.bb_lower || [], stroke: { width: 1, dashArray: 3 } }
        ],
        xaxis: { type: 'datetime', labels: { show: false }, axisBorder: { show: false }, axisTicks: { show: false }, tooltip: { enabled: false } },
        yaxis: [
            { labels: { show: false }, axisBorder: { show: false }, min: bounds?.min, max: bounds?.max, tickAmount: 2 },
            { labels: { show: false }, axisBorder: { show: false }, opposite: true, min: 0, max: volMax * 1.4, tickAmount: 2 }
        ],
        grid: { show: false, padding: { top: 0, right: 0, left: 0, bottom: -10 } },
        legend: { show: false },
        dataLabels: { enabled: false },
        tooltip: { theme: 'dark', shared: false, x: { show: false } },
        plotOptions: {
            candlestick: {
                colors: { upward: '#22C55E', downward: '#EF4444' },
                wick: { useFillColor: true }
            },
            bar: {
                columnWidth: '75%',
                borderRadius: 0,
                borderRadiusApplication: 'end'
            }
        },
        stroke: { colors: ['#3B82F6', 'rgba(148,163,184,0.5)', '#22C55E', '#60A5FA', '#60A5FA'], width: [1, 0, 1.2, 1, 1] },
        fill: { opacity: 1 },
        annotations: {
            yaxis: [
                ...(series.s1 ? [{ y: series.s1, borderColor: '#16A34A', label: { text: 'S1', style: { color: '#fff', background: '#16A34A' } } }] : []),
                ...(series.s2 ? [{ y: series.s2, borderColor: '#16A34A', label: { text: 'S2', style: { color: '#fff', background: '#16A34A' } } }] : []),
                ...(series.r1 ? [{ y: series.r1, borderColor: '#DC2626', label: { text: 'R1', style: { color: '#fff', background: '#DC2626' } } }] : []),
                ...(series.r2 ? [{ y: series.r2, borderColor: '#DC2626', label: { text: 'R2', style: { color: '#fff', background: '#DC2626' } } }] : [])
            ],
            texts: [{
                text: 'Source: Yahoo Finance',
                x: 10,
                y: 10,
                textAnchor: 'start',
                foreColor: '#94A3B8',
                opacity: 0.45
            }]
        }
    };
    const chart = new ApexCharts(node, options);
    chart.render();
}
