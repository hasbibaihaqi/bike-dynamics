
window.BikeCharts = {
    palette: {
        blue: '#0284c7',
        blueFill: 'rgba(2, 132, 199, 0.14)',
        teal: '#0d9488',
        tealFill: 'rgba(13, 148, 136, 0.14)',
        amber: '#d97706',
        amberFill: 'rgba(217, 119, 6, 0.16)',
        rose: '#e11d48',
        roseFill: 'rgba(225, 29, 72, 0.18)',
        slate: '#475569',
        indigo: '#4f46e5'
    },

    createLineChart: function (canvasId, labels, datasets, yLabel = '') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: 'line',
            data: { labels: labels, datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } }
                },
                scales: {
                    x: { ticks: { maxTicksLimit: 14, font: { size: 11 } }, grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        title: { display: Boolean(yLabel), text: yLabel },
                        grid: { color: '#e2e8f0' }
                    }
                }
            }
        });
    },

    createBarChart: function (canvasId, labels, datasets, yLabel = '') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: 'bar',
            data: { labels: labels, datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        title: { display: Boolean(yLabel), text: yLabel },
                        grid: { color: '#e2e8f0' }
                    }
                }
            }
        });
    }
};
