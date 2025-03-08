/**
 * HESA Data Visualization - Dashboard JavaScript
 * 
 * This file contains the JavaScript code for the dashboard,
 * handling query submission, visualization, and data interaction.
 */

document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const queryForm = document.getElementById('queryForm');
    const interpretationCard = document.getElementById('interpretationCard');
    const interpretationResult = document.getElementById('interpretationResult');
    const dataSummary = document.getElementById('dataSummary');
    const exampleQueriesBtn = document.getElementById('exampleQueries');
    const exampleQueriesModal = document.getElementById('exampleQueriesModal');
    const closeExampleQueriesBtn = document.getElementById('closeExampleQueries');
    const downloadCSVBtn = document.getElementById('downloadCSV');
    const downloadExcelBtn = document.getElementById('downloadExcel');
    const downloadPDFBtn = document.getElementById('downloadPDF');
    
    // Example queries handling
    exampleQueriesBtn.addEventListener('click', () => {
        exampleQueriesModal.classList.remove('hidden');
    });

    closeExampleQueriesBtn.addEventListener('click', () => {
        exampleQueriesModal.classList.add('hidden');
    });

    document.querySelectorAll('.example-query').forEach(query => {
        query.addEventListener('click', () => {
            document.getElementById('query').value = query.textContent.trim();
            exampleQueriesModal.classList.add('hidden');
        });
    });

    // Close modal when clicking outside
    exampleQueriesModal.addEventListener('click', (e) => {
        if (e.target === exampleQueriesModal) {
            exampleQueriesModal.classList.add('hidden');
        }
    });
    
    // Chart rendering
    let mainChart = null;
    
    // Form submission
    queryForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        showLoading();
        
        // Get the CSRF token
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        
        // Get the query text
        const queryText = document.getElementById('query').value;
        
        // Send the query to the backend
        fetch(queryForm.action, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrfToken,
            },
            body: new URLSearchParams({
                'query': queryText
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.status === 'error') {
                showError(data.error);
                return;
            }
            
            // Show the interpretation card
            interpretationCard.classList.remove('hidden');
            
            // Display the interpretation
            displayInterpretation(data.interpretation);
            
            // Display the visualization
            renderChart(data.visualization);
            
            // Display the summary
            displaySummary(data.summary, data.row_count);
            
            // Enable download buttons
            enableDownloadButtons();
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred while processing your query. Please try again.');
            console.error('Error:', error);
        });
    });
    
    function showLoading() {
        // Add a loading indicator to the page
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        loadingDiv.innerHTML = `
            <div class="bg-white p-6 rounded-lg">
                <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
                <p class="text-center mt-4">Processing your query...</p>
            </div>
        `;
        document.body.appendChild(loadingDiv);
    }
    
    function hideLoading() {
        // Remove the loading indicator
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }
    
    function showError(message) {
        // Display error message
        interpretationCard.classList.remove('hidden');
        interpretationResult.innerHTML = `
            <div class="bg-red-100 text-red-700 p-4 rounded">
                <p class="font-bold">Error</p>
                <p>${message}</p>
            </div>
        `;
        
        // Clear chart and summary
        if (mainChart) {
            mainChart.destroy();
            mainChart = null;
        }
        dataSummary.innerHTML = '';
    }
    
    function displayInterpretation(interpretation) {
        // Format the interpretation data
        let html = '<div class="space-y-2">';
        
        if (interpretation.metrics && interpretation.metrics.length > 0) {
            html += `<p><strong>Metrics:</strong> ${interpretation.metrics.join(', ')}</p>`;
        }
        
        if (interpretation.institutions && interpretation.institutions.length > 0) {
            html += `<p><strong>Institutions:</strong> ${interpretation.institutions.join(', ')}</p>`;
        }
        
        if (interpretation.time_period) {
            const timePeriod = [];
            if (interpretation.time_period.start) {
                timePeriod.push(`From ${interpretation.time_period.start}`);
            }
            if (interpretation.time_period.end) {
                timePeriod.push(`To ${interpretation.time_period.end}`);
            }
            if (timePeriod.length > 0) {
                html += `<p><strong>Time Period:</strong> ${timePeriod.join(' ')}</p>`;
            }
        }
        
        if (interpretation.comparison_type) {
            html += `<p><strong>Analysis Type:</strong> ${interpretation.comparison_type.charAt(0).toUpperCase() + interpretation.comparison_type.slice(1)}</p>`;
        }
        
        html += '</div>';
        interpretationResult.innerHTML = html;
    }
    
    function renderChart(visualization) {
        const chartCanvas = document.getElementById('mainChart');
        
        // Destroy existing chart if it exists
        if (mainChart) {
            mainChart.destroy();
        }
        
        // Create a new chart
        const ctx = chartCanvas.getContext('2d');
        mainChart = new Chart(ctx, {
            type: visualization.type,
            data: visualization.data,
            options: visualization.options
        });
    }
    
    function displaySummary(summary, rowCount) {
        let html = '<div class="space-y-4">';
        
        // Display row count
        html += `<p><strong>Total Records:</strong> ${rowCount}</p>`;
        
        // Display statistical summary
        if (summary && summary.statistics) {
            html += '<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200">';
            html += '<thead class="bg-gray-50"><tr>';
            html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>';
            html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>';
            html += '</tr></thead><tbody class="bg-white divide-y divide-gray-200">';
            
            Object.keys(summary.statistics).forEach(key => {
                html += '<tr>';
                html += `<td class="px-6 py-4 whitespace-nowrap">${key}</td>`;
                html += `<td class="px-6 py-4 whitespace-nowrap">${summary.statistics[key]}</td>`;
                html += '</tr>';
            });
            
            html += '</tbody></table></div>';
        }
        
        // Display other summary information
        if (summary && summary.top_values) {
            html += '<h4 class="font-semibold text-lg mt-4">Top Values</h4>';
            html += '<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200">';
            html += '<thead class="bg-gray-50"><tr>';
            html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>';
            html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>';
            html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>';
            html += '</tr></thead><tbody class="bg-white divide-y divide-gray-200">';
            
            Object.keys(summary.top_values).forEach(category => {
                const values = summary.top_values[category];
                if (values && values.length > 0) {
                    values.forEach((item, index) => {
                        html += '<tr>';
                        if (index === 0) {
                            html += `<td class="px-6 py-4 whitespace-nowrap" rowspan="${values.length}">${category}</td>`;
                        }
                        html += `<td class="px-6 py-4 whitespace-nowrap">${item.value}</td>`;
                        html += `<td class="px-6 py-4 whitespace-nowrap">${item.count}</td>`;
                        html += '</tr>';
                    });
                }
            });
            
            html += '</tbody></table></div>';
        }
        
        html += '</div>';
        dataSummary.innerHTML = html;
    }
    
    function enableDownloadButtons() {
        // Enable download buttons and set up event listeners
        // This would be implemented in a full version to download data in different formats
        downloadCSVBtn.disabled = false;
        downloadExcelBtn.disabled = false;
        downloadPDFBtn.disabled = false;
        
        // Example implementation for CSV download
        downloadCSVBtn.addEventListener('click', () => {
            window.location.href = '/download/csv/';
        });
        
        // Example implementation for Excel download
        downloadExcelBtn.addEventListener('click', () => {
            window.location.href = '/download/excel/';
        });
        
        // PDF download would require additional implementation
    }
}); 