document.addEventListener('DOMContentLoaded', function() {
    // Initialize health score circles
    document.querySelectorAll('.score-circle').forEach(circle => {
        const score = circle.dataset.score;
        circle.style.setProperty('--score', score + '%');
        circle.style.setProperty('--final-score', score + '%');
        
        // Add animation
        circle.style.animation = 'fillCircle 1s ease-out forwards';
    });
    
    // Initialize charts for each stock
    document.querySelectorAll('[id^="chart-"]').forEach(canvas => {
        const symbol = canvas.id.replace('chart-', '');
        initializeChart(canvas, symbol);
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
});

function initializeChart(canvas, symbol) {
    const ctx = canvas.getContext('2d');
    
    // Sample data - replace with actual stock data
    const data = {
        labels: Array.from({length: 30}, (_, i) => i + 1),
        datasets: [{
            label: symbol,
            data: Array.from({length: 30}, () => Math.random() * 100 + 1000),
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: true
        }]
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
                    tension: 0.4
                }
            }
        }
    });
}
