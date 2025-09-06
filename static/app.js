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
        } else if (series && series.labels && series.close) {
            initializeChart(node, series);
        }
    });
    
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
            if (!highs.length) return null;
            const min = Math.min(...lows);
            const max = Math.max(...highs);
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
            sparkline: { enabled: true }
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
            { labels: { show: false }, axisBorder: { show: false }, min: bounds?.min, max: bounds?.max },
            { labels: { show: false }, axisBorder: { show: false }, opposite: true, min: 0, max: volMax * 1.4 }
        ],
        grid: { show: false },
        legend: { show: false },
        dataLabels: { enabled: false },
        tooltip: { theme: 'dark', shared: false, x: { show: false } },
        plotOptions: {
            candlestick: {
                colors: { upward: '#22C55E', downward: '#EF4444' },
                wick: { useFillColor: true }
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
