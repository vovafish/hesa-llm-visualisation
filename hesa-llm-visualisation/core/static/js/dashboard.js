// Chart instance
let mainChart = null;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    setupEventListeners();
});

// Initialize the main chart
function initializeChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'bar',  // Default type, will be updated based on data
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'HESA Data Visualization'
                }
            }
        }
    });
}

// Set up event listeners
function setupEventListeners() {
    const queryForm = document.getElementById('queryForm');
    const downloadButtons = {
        csv: document.getElementById('downloadCSV'),
        pdf: document.getElementById('downloadPDF'),
        excel: document.getElementById('downloadExcel')
    };

    // Form submission
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleQuerySubmission(e.target);
    });

    // Download buttons
    Object.entries(downloadButtons).forEach(([format, button]) => {
        button.addEventListener('click', () => handleDownload(format));
    });
}

// Handle query submission
async function handleQuerySubmission(form) {
    try {
        const formData = new FormData(form);
        
        // Show loading state
        showLoading();

        const response = await fetch(form.action, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': formData.get('csrfmiddlewaretoken')
            }
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Update UI with results
        updateQueryInterpretation(data.interpretation);
        updateVisualization(data.visualization);
        updateDataSummary(data.summary);

        // Hide loading state
        hideLoading();

    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while processing your query.');
        hideLoading();
    }
}

// Update the chart with new data
function updateVisualization(visualizationData) {
    if (!visualizationData) return;

    const {type, data, options} = visualizationData;

    // Destroy existing chart if it exists
    if (mainChart) {
        mainChart.destroy();
    }

    // Create new chart with updated data
    const ctx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: type || 'bar',
        data: data,
        options: {
            ...options,
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: visualizationData.title || 'HESA Data Visualization'
                }
            }
        }
    });
}

// Update query interpretation display
function updateQueryInterpretation(interpretation) {
    const interpretationCard = document.getElementById('interpretationCard');
    const interpretationResult = document.getElementById('interpretationResult');

    if (interpretation) {
        interpretationCard.classList.remove('hidden');
        interpretationResult.innerHTML = `
            <p class="text-gray-700">${interpretation}</p>
        `;
    } else {
        interpretationCard.classList.add('hidden');
    }
}

// Update data summary display
function updateDataSummary(summary) {
    const dataSummary = document.getElementById('dataSummary');
    
    if (!summary) {
        dataSummary.innerHTML = '<p class="text-gray-500">No summary available</p>';
        return;
    }

    let summaryHTML = '<div class="space-y-4">';
    
    // Add key statistics
    if (summary.stats) {
        summaryHTML += `
            <div class="grid grid-cols-2 gap-4">
                ${Object.entries(summary.stats).map(([key, value]) => `
                    <div class="bg-white p-3 rounded shadow-sm">
                        <div class="text-sm text-gray-500">${key}</div>
                        <div class="text-lg font-semibold">${value}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Add insights if available
    if (summary.insights) {
        summaryHTML += `
            <div class="mt-4">
                <h4 class="font-semibold mb-2">Key Insights:</h4>
                <ul class="list-disc list-inside space-y-1">
                    ${summary.insights.map(insight => `
                        <li class="text-gray-700">${insight}</li>
                    `).join('')}
                </ul>
            </div>
        `;
    }

    summaryHTML += '</div>';
    dataSummary.innerHTML = summaryHTML;
}

// Handle file downloads
async function handleDownload(format) {
    try {
        const response = await fetch(`/download/${format}`, {
            method: 'GET',
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
        });

        if (!response.ok) {
            throw new Error('Download failed');
        }

        // Handle the download based on format
        if (format === 'csv' || format === 'excel') {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `hesa_data.${format}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } else if (format === 'pdf') {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            window.open(url);
        }

    } catch (error) {
        console.error('Download error:', error);
        showError('Failed to download the file.');
    }
}

// UI Helper Functions
function showLoading() {
    // Add loading indicator logic here
    document.body.classList.add('cursor-wait');
}

function hideLoading() {
    document.body.classList.remove('cursor-wait');
}

function showError(message) {
    // Add error display logic here
    alert(message); // Replace with better error UI
} 