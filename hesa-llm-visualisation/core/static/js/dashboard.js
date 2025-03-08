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
    const filterControlsDiv = document.getElementById('filterControls');
    const dateRangeDiv = document.getElementById('dateRangeControl');
    
    // State variables for interactive features
    let currentChartData = null;
    let currentChartConfig = null;
    let drillDownHistory = [];
    let filters = {};
    
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
        
        console.log("Form submitted");
        
        // Show loading state
        showLoading();
        
        // Get the CSRF token
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        console.log("CSRF Token:", csrfToken);
        
        // Get the query text
        const queryText = document.getElementById('query').value;
        console.log("Query:", queryText);
        
        // Log the form action
        console.log("Form action:", queryForm.action);
        
        // Reset drill-down history and filters when submitting a new query
        drillDownHistory = [];
        filters = {};
        
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
        .then(response => {
            console.log("Response status:", response.status);
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            
            if (data.status === 'error') {
                showError(data.error);
                return;
            }
            
            // Store chart data for interactive features
            currentChartData = data.chart.data;
            currentChartConfig = data.chart;
            
            // Show the interpretation card and interactive controls
            interpretationCard.classList.remove('hidden');
            document.getElementById('interactiveControls').classList.remove('hidden');
            
            // Display the interpretation
            displayInterpretation(data.query_interpretation);
            
            // Display the visualization
            renderChart(data.chart);
            
            // Setup interactive controls
            setupInteractiveControls(data.chart.data, data.query_interpretation);
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
            html += `<p><strong>Time Period:</strong> ${interpretation.time_period.start} to ${interpretation.time_period.end}</p>`;
        }
        
        if (interpretation.comparison_type) {
            html += `<p><strong>Analysis Type:</strong> ${interpretation.comparison_type}</p>`;
        }
        
        if (interpretation.visualization) {
            html += `<p><strong>Chart Type:</strong> ${interpretation.visualization.type}</p>`;
        }
        
        html += '</div>';
        interpretationResult.innerHTML = html;
    }
    
    function renderChart(chartConfig) {
        const chartCanvas = document.getElementById('mainChart');
        
        // Destroy existing chart if it exists
        if (mainChart) {
            mainChart.destroy();
        }
        
        // Create chart configuration
        const config = {
            type: chartConfig.type,
            data: {
                labels: chartConfig.data.map(d => d.year || d.institution),
                datasets: [{
                    label: chartConfig.data[0] ? Object.keys(chartConfig.data[0]).find(k => k !== 'year' && k !== 'institution') : '',
                    data: chartConfig.data.map(d => Object.values(d).find(v => typeof v === 'number')),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                onClick: handleChartClick, // Add click event for drill-down
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const dataPoint = chartConfig.data[context.dataIndex];
                                return `${context.dataset.label}: ${context.formattedValue}`;
                            }
                        }
                    }
                }
            }
        };
        
        // Create a new chart
        mainChart = new Chart(chartCanvas, config);
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
    
    // Setup interactive controls based on chart data
    function setupInteractiveControls(data, interpretation) {
        // Clear previous controls
        filterControlsDiv.innerHTML = '';
        dateRangeDiv.innerHTML = '';
        
        // Add drill-down navigation if we have history
        if (drillDownHistory.length > 0) {
            const drillUpBtn = document.createElement('button');
            drillUpBtn.className = 'btn-secondary text-sm px-3 py-1 mb-3';
            drillUpBtn.textContent = '← Back to Previous Level';
            drillUpBtn.addEventListener('click', handleDrillUp);
            filterControlsDiv.appendChild(drillUpBtn);
        }
        
        // Create filter controls based on data
        if (data && data.length > 0) {
            const firstDataPoint = data[0];
            
            // Filter by institution if available
            if (firstDataPoint.institution && !filters.institution) {
                // Get unique institutions
                const institutions = [...new Set(data.map(d => d.institution))];
                if (institutions.length > 1) {
                    addFilterDropdown('institution', 'Institution', institutions);
                }
            }
            
            // Add metric filters if multiple metrics
            const metrics = Object.keys(firstDataPoint).filter(k => 
                k !== 'year' && k !== 'institution' && typeof firstDataPoint[k] === 'number'
            );
            
            if (metrics.length > 1) {
                addFilterDropdown('metric', 'Metric', metrics);
            }
        }
        
        // Add date range control if time data is available
        if (interpretation && interpretation.time_period) {
            const startYear = parseInt(interpretation.time_period.start);
            const endYear = parseInt(interpretation.time_period.end);
            
            if (!isNaN(startYear) && !isNaN(endYear)) {
                addDateRangeControl(startYear, endYear);
            }
        }
    }
    
    function addFilterDropdown(filterKey, label, options) {
        const filterContainer = document.createElement('div');
        filterContainer.className = 'mb-3';
        
        const filterLabel = document.createElement('label');
        filterLabel.textContent = `Filter by ${label}: `;
        filterLabel.className = 'mr-2 text-sm font-medium';
        
        const select = document.createElement('select');
        select.className = 'border rounded px-2 py-1 text-sm';
        
        // Add "All" option
        const allOption = document.createElement('option');
        allOption.value = '';
        allOption.textContent = 'All';
        select.appendChild(allOption);
        
        // Add options
        options.forEach(option => {
            const optionEl = document.createElement('option');
            optionEl.value = option;
            optionEl.textContent = option;
            select.appendChild(optionEl);
        });
        
        // Set value from filters if exists
        if (filters[filterKey]) {
            select.value = filters[filterKey];
        }
        
        // Add event listener
        select.addEventListener('change', function() {
            if (this.value) {
                filters[filterKey] = this.value;
            } else {
                delete filters[filterKey];
            }
            applyFilters();
        });
        
        filterContainer.appendChild(filterLabel);
        filterContainer.appendChild(select);
        filterControlsDiv.appendChild(filterContainer);
    }
    
    function addDateRangeControl(minYear, maxYear) {
        const container = document.createElement('div');
        container.className = 'mb-3';
        
        const label = document.createElement('div');
        label.textContent = 'Date Range:';
        label.className = 'text-sm font-medium mb-1';
        
        const rangeContainer = document.createElement('div');
        rangeContainer.className = 'flex items-center space-x-2';
        
        // Start year input
        const startYearContainer = document.createElement('div');
        startYearContainer.className = 'flex items-center';
        
        const startLabel = document.createElement('label');
        startLabel.textContent = 'From:';
        startLabel.className = 'mr-1 text-sm';
        
        const startInput = document.createElement('input');
        startInput.type = 'number';
        startInput.className = 'border rounded px-2 py-1 w-20 text-sm';
        startInput.min = minYear;
        startInput.max = maxYear;
        startInput.value = minYear;
        startInput.id = 'startYearInput';
        
        startYearContainer.appendChild(startLabel);
        startYearContainer.appendChild(startInput);
        
        // End year input
        const endYearContainer = document.createElement('div');
        endYearContainer.className = 'flex items-center';
        
        const endLabel = document.createElement('label');
        endLabel.textContent = 'To:';
        endLabel.className = 'mr-1 text-sm';
        
        const endInput = document.createElement('input');
        endInput.type = 'number';
        endInput.className = 'border rounded px-2 py-1 w-20 text-sm';
        endInput.min = minYear;
        endInput.max = maxYear;
        endInput.value = maxYear;
        endInput.id = 'endYearInput';
        
        endYearContainer.appendChild(endLabel);
        endYearContainer.appendChild(endInput);
        
        // Apply button
        const applyBtn = document.createElement('button');
        applyBtn.textContent = 'Apply';
        applyBtn.className = 'btn-secondary text-sm px-3 py-1';
        applyBtn.addEventListener('click', function() {
            const startYear = parseInt(document.getElementById('startYearInput').value);
            const endYear = parseInt(document.getElementById('endYearInput').value);
            
            if (startYear > endYear) {
                alert('Start year cannot be greater than end year');
                return;
            }
            
            filters.startYear = startYear;
            filters.endYear = endYear;
            applyFilters();
        });
        
        rangeContainer.appendChild(startYearContainer);
        rangeContainer.appendChild(endYearContainer);
        rangeContainer.appendChild(applyBtn);
        
        container.appendChild(label);
        container.appendChild(rangeContainer);
        dateRangeDiv.appendChild(container);
    }
    
    function applyFilters() {
        if (!currentChartData) {
            return;
        }
        
        // Apply filters to current chart data
        let filteredData = [...currentChartData];
        
        // Apply institution filter
        if (filters.institution) {
            filteredData = filteredData.filter(d => d.institution === filters.institution);
        }
        
        // Apply year range filter
        if (filters.startYear && filters.endYear) {
            filteredData = filteredData.filter(d => {
                const year = parseInt(d.year);
                return !isNaN(year) && year >= filters.startYear && year <= filters.endYear;
            });
        }
        
        // Apply metric filter by transforming the data
        if (filters.metric && filteredData.length > 0 && filteredData[0].hasOwnProperty(filters.metric)) {
            filteredData = filteredData.map(d => {
                return {
                    year: d.year,
                    institution: d.institution,
                    [filters.metric]: d[filters.metric]
                };
            });
        }
        
        // Update chart with filtered data
        const updatedChartConfig = {
            ...currentChartConfig,
            data: filteredData
        };
        
        renderChart(updatedChartConfig);
    }
    
    // Handle chart click for drill-down functionality
    function handleChartClick(event, elements) {
        if (!elements || !elements.length || !currentChartData) {
            return;
        }
        
        const element = elements[0];
        const index = element.index;
        const dataPoint = currentChartData[index];
        
        // Save current state to history for drilling back up
        drillDownHistory.push({
            chartData: currentChartData,
            chartConfig: currentChartConfig,
            filters: {...filters}
        });
        
        // Determine drill-down type based on data structure
        if (dataPoint.institution) {
            // Drill down into institution
            filters.institution = dataPoint.institution;
            
            // Update query to show we're drilling down
            document.getElementById('query').value += ` (Drilling into ${dataPoint.institution})`;
        } else if (dataPoint.year) {
            // Drill down into year
            filters.startYear = parseInt(dataPoint.year);
            filters.endYear = parseInt(dataPoint.year);
            
            // Update query to show we're drilling down
            document.getElementById('query').value += ` (Drilling into ${dataPoint.year})`;
        }
        
        // Apply the new filters for drill-down
        applyFilters();
        
        // Update the filter controls to show the drill-down state
        setupInteractiveControls(currentChartData, null);
    }
    
    // Handle drilling back up
    function handleDrillUp() {
        if (drillDownHistory.length === 0) {
            return;
        }
        
        // Get the previous state
        const previousState = drillDownHistory.pop();
        
        // Restore the previous state
        currentChartData = previousState.chartData;
        currentChartConfig = previousState.chartConfig;
        filters = {...previousState.filters};
        
        // Render chart with previous state
        renderChart(currentChartConfig);
        
        // Update UI to reflect previous state
        setupInteractiveControls(currentChartData, null);
        
        // Update query text to remove drill-down suffix
        const queryText = document.getElementById('query').value;
        document.getElementById('query').value = queryText.replace(/ \(Drilling into .*\)$/, '');
    }
    
    // Setup download buttons
    downloadCSVBtn.addEventListener('click', function() {
        if (!currentChartData || currentChartData.length === 0) {
            alert('No data available to download');
            return;
        }
        
        // Convert data to CSV
        const headers = Object.keys(currentChartData[0]).join(',');
        const rows = currentChartData.map(row => Object.values(row).join(','));
        const csv = [headers, ...rows].join('\n');
        
        // Download CSV file
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hesa_data_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
    
    downloadPDFBtn.addEventListener('click', function() {
        if (!mainChart) {
            alert('No chart available to download');
            return;
        }
        
        // Use html2canvas and jsPDF to download chart as PDF
        alert('PDF export functionality will be implemented in future updates');
    });
    
    downloadExcelBtn.addEventListener('click', function() {
        alert('Excel export functionality will be implemented in future updates');
    });
}); 