/**
 * HESA Data Visualization - Dashboard JavaScript
 * 
 * Simplified version with basic query handling.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard JavaScript loaded');
    
    // Get chart buttons
    const lineChartBtn = document.getElementById('lineChartBtn');
    const barChartBtn = document.getElementById('barChartBtn');
    const pieChartBtn = document.getElementById('pieChartBtn');
    
    // Get result container
    const queryResultsContainer = document.getElementById('queryResults');
    
    // Get sample queries elements
    const sampleQueriesBtn = document.getElementById('sampleQueriesBtn');
    const sampleQueriesDropdown = document.getElementById('sampleQueriesDropdown');
    
    console.log('Elements found:', {
        lineChartBtn: !!lineChartBtn,
        barChartBtn: !!barChartBtn,
        pieChartBtn: !!pieChartBtn,
        queryResultsContainer: !!queryResultsContainer,
        sampleQueriesBtn: !!sampleQueriesBtn,
        sampleQueriesDropdown: !!sampleQueriesDropdown
    });
    
    // Sample Queries Button Click Handler
    if (sampleQueriesBtn && sampleQueriesDropdown) {
        sampleQueriesBtn.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event from bubbling up
            sampleQueriesDropdown.classList.toggle('hidden');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!sampleQueriesBtn.contains(e.target) && !sampleQueriesDropdown.contains(e.target)) {
                sampleQueriesDropdown.classList.add('hidden');
            }
        });
        
        // Handle sample query item clicks
        const sampleQueryItems = document.querySelectorAll('.sample-query-item');
        sampleQueryItems.forEach(item => {
            item.addEventListener('click', function() {
                const queryInput = document.getElementById('dashboardQuery');
                if (queryInput) {
                    const query = this.getAttribute('data-query');
                    queryInput.value = query;
                    sampleQueriesDropdown.classList.add('hidden');
                    
                    // Optional: Highlight the input to indicate it was changed
                    queryInput.classList.add('ring-2', 'ring-blue-500');
                    setTimeout(() => {
                        queryInput.classList.remove('ring-2', 'ring-blue-500');
                    }, 1000);
                }
            });
        });
    }
    
    // Line Chart Button Click Handler
    lineChartBtn.addEventListener('click', function() {
        console.log('Line Chart button clicked');
        
        // Get the query from the input field
        const queryInput = document.getElementById('dashboardQuery');
        if (!queryInput || !queryInput.value.trim()) {
            alert('Please enter a query first.');
            return;
        }
        
        const userQuery = queryInput.value.trim();
        console.log('Using query:', userQuery);
        
        // Show loading state
        showLoading();
        
        // Process the query
        processQuery(userQuery, 'line');
    });
    
    // Bar Chart Button Click Handler
    barChartBtn.addEventListener('click', function() {
        console.log('Bar Chart button clicked');
        
        // Get the query from the input field
        const queryInput = document.getElementById('dashboardQuery');
        if (!queryInput || !queryInput.value.trim()) {
            alert('Please enter a query first.');
            return;
        }
        
        alert('Bar chart functionality not implemented yet');
    });
    
    // Pie Chart Button Click Handler
    pieChartBtn.addEventListener('click', function() {
        console.log('Pie Chart button clicked');
        
        // Get the query from the input field
        const queryInput = document.getElementById('dashboardQuery');
        if (!queryInput || !queryInput.value.trim()) {
            alert('Please enter a query first.');
            return;
        }
        
        alert('Pie chart functionality not implemented yet');
    });
    
    // Function to process the query and fetch data
    function processQuery(query, chartType) {
        console.log('Processing query:', query, 'with chart type:', chartType);
        
        // Get the CSRF token for POST requests
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        console.log('CSRF token found:', !!csrfToken);
        
        // Send the query to the backend
        console.log('Sending fetch request to /process_hesa_query/');
        fetch('/process_hesa_query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrfToken,
            },
            body: new URLSearchParams({
                'query': query,
                'chart_type': chartType
            })
        })
        .then(response => {
            console.log('Received response with status:', response.status);
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            hideLoading();
            
            if (data.status === 'error') {
                console.error('Error from backend:', data.error);
                showError(data.error);
                return;
            }
            
            // Display the query and results
            displayQueryResults(query, data);
        })
        .catch(error => {
            console.error('Fetch error:', error);
            hideLoading();
            showError('An error occurred while processing your query. Please try again.');
            console.error('Error:', error);
        });
    }
    
    // Function to display query results
    function displayQueryResults(query, data) {
        console.log('Displaying query results');
        
        // Clear previous results
        queryResultsContainer.innerHTML = '';
        
        // Check if we have multiple matches that need a preview
        if (data.multiple_matches && data.preview_results) {
            console.log('Multiple file matches found, displaying previews');
            
            // Display warning about multiple matches
            let warningHTML = `
                <div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6" role="alert">
                    <p class="font-bold">Warning: Your query matched multiple data sources (${data.preview_results.length} dataset types)</p>
                    <p>Previews from each dataset type are shown below. Select one to view complete results.</p>
                    <p class="mt-2 text-sm">Note: You requested data for year(s): <strong>${data.query_info.years.join(", ")}</strong></p>
                </div>
            `;
            queryResultsContainer.innerHTML = warningHTML;
            
            // Display each dataset type preview
            data.preview_results.forEach(result => {
                // Get file years for display
                const fileYears = result.file_info.years || [];
                const yearsText = fileYears.length > 0 ? fileYears.join(", ") : 'Unknown years';
                
                // Format matched terms for better display
                let matchedTermsHtml = '';
                if (result.file_info.matched_terms && result.file_info.matched_terms.length > 0) {
                    matchedTermsHtml = result.file_info.matched_terms.join(', ');
                } else {
                    matchedTermsHtml = 'No specific terms matched';
                }
                
                // Format file names for display
                const fileNames = result.file_info.file_names || [];
                let fileNamesHtml = '';
                if (fileNames.length > 0) {
                    fileNamesHtml = `
                        <div class="mt-2 mb-3 text-sm">
                            <p class="font-medium">Source files (showing data for ${data.query_info.years.join(", ")} only):</p>
                            <ul class="list-disc pl-5 text-gray-600">
                                ${fileNames.map(name => `<li>${name}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                // Create a highlight class if any of the file years match requested years
                const yearMatch = fileYears.some(year => data.query_info.years.includes(year)) ? 
                    'border-green-500 bg-green-50' : 'border-gray-200';
                
                let previewHTML = `
                    <div class="rounded-lg shadow p-4 mb-6 file-preview border-l-4 ${yearMatch}" data-file-id="${result.file_id}">
                        <h3 class="text-lg font-semibold mb-2">Dataset: ${result.file_info.group_title}</h3>
                        <div class="grid grid-cols-2 gap-4 mb-3">
                            <p class="text-sm text-gray-700"><span class="font-medium">Years:</span> ${yearsText}</p>
                            <p class="text-sm text-gray-700"><span class="font-medium">Match score:</span> ${result.file_info.match_score}</p>
                        </div>
                        <p class="mb-2 text-sm text-gray-700"><span class="font-medium">Matched terms:</span> ${matchedTermsHtml}</p>
                        ${fileNamesHtml}
                        
                        <div class="overflow-x-auto mb-4">
                            <table class="min-w-full bg-white border border-gray-300">
                                <thead>
                                    <tr>
                                `;
                                
                                // Add table headers
                                if (result.columns && result.columns.length > 0) {
                                    result.columns.forEach(column => {
                                        previewHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left">${column}</th>`;
                                    });
                                }
                                
                                previewHTML += `
                                        </tr>
                                    </thead>
                                    <tbody>
                                `;
                                
                                // Add preview data rows
                                if (result.data && result.data.length > 0) {
                                    result.data.forEach(row => {
                                        previewHTML += '<tr>';
                                        result.columns.forEach(column => {
                                            previewHTML += `<td class="px-4 py-2 border-b border-gray-300">${row[column] || ''}</td>`;
                                        });
                                        previewHTML += '</tr>';
                                    });
                                } else {
                                    previewHTML += `
                                        <tr>
                                            <td colspan="${result.columns ? result.columns.length : 1}" class="px-4 py-2 text-center text-gray-500">
                                                No preview data available
                                            </td>
                                        </tr>
                                    `;
                                }
                                
                                previewHTML += `
                                    </tbody>
                                </table>
                            </div>
                            
                            ${result.has_more_rows ? `
                            <div class="bg-blue-100 text-blue-800 p-3 mb-4 rounded text-sm">
                                <p><i class="fas fa-info-circle mr-1"></i> Preview limited to 3 rows. Select this dataset to view all matching data.</p>
                            </div>
                            ` : ''}
                            
                            <button class="select-file-btn bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" 
                                    data-file-id="${result.file_id}">
                                Select This Dataset
                            </button>
                        </div>
                    </div>
                `;
                
                queryResultsContainer.innerHTML += previewHTML;
            });
            
            // Add event listeners to the select buttons
            document.querySelectorAll('.select-file-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const fileId = this.getAttribute('data-file-id');
                    selectFile(query, fileId);
                });
            });
            
            return; // Exit early as we're showing previews
        }
        
        // Regular result display (not multiple matches)
        // Create results HTML
        let resultsHTML = `
            <div class="bg-white rounded-lg shadow p-4 mb-6">
                <h3 class="text-lg font-semibold mb-2">Query:</h3>
                <p class="mb-4 text-gray-700">${query}</p>
                
                <h3 class="text-lg font-semibold mb-2">Results:</h3>
            `;
            
            // Add file info section if available
            if (data.file_info && data.file_info.length > 0) {
                resultsHTML += `
                    <div class="bg-blue-50 border-l-4 border-blue-500 p-3 mb-4">
                        <p class="font-medium">Source files (showing data for ${data.query_info.years.join(", ")} only):</p>
                        <ul class="list-disc pl-5 text-gray-600">
                `;
                
                data.file_info.forEach(fileInfo => {
                    resultsHTML += `<li>${fileInfo.file_name} (${fileInfo.year || 'Unknown year'})</li>`;
                });
                
                resultsHTML += `
                        </ul>
                    </div>
                `;
            }
            
            resultsHTML += `
                <div class="overflow-x-auto">
                    <table class="min-w-full bg-white border border-gray-300">
                        <thead>
                            <tr>
            `;
            
            // Add table headers
            if (data.columns && data.columns.length > 0) {
                console.log('Table columns:', data.columns);
                data.columns.forEach(column => {
                    resultsHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left">${column}</th>`;
                });
            } else {
                console.warn('No columns found in data');
            }
            
            resultsHTML += `
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            // Add table rows
            if (data.data && data.data.length > 0) {
                console.log('Table data rows:', data.data.length);
                data.data.forEach(row => {
                    resultsHTML += '<tr>';
                    data.columns.forEach(column => {
                        resultsHTML += `<td class="px-4 py-2 border-b border-gray-300">${row[column] || ''}</td>`;
                    });
                    resultsHTML += '</tr>';
                });
            } else {
                console.warn('No data rows found');
                resultsHTML += `
                    <tr>
                        <td colspan="${data.columns ? data.columns.length : 1}" class="px-4 py-2 text-center text-gray-500">
                            No data found
                        </td>
                    </tr>
                `;
            }
            
            resultsHTML += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        // Add to the page
        queryResultsContainer.innerHTML = resultsHTML;
        console.log('Results HTML added to page');
    }
    
    // Function to select a specific file and get complete results
    function selectFile(query, fileId) {
        console.log(`Selecting file with ID: ${fileId} for query: ${query}`);
        showLoading();
        
        // Get the CSRF token for POST requests
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        
        // Get the currently selected chart type
        const chartType = getCurrentChartType();
        
        // Send selection to backend
        fetch('/select_file_source/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrfToken,
            },
            body: new URLSearchParams({
                'query': query,
                'file_id': fileId,
                'chart_type': chartType
            })
        })
        .then(response => {
            console.log('Received response with status:', response.status);
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data after file selection:', data);
            hideLoading();
            
            if (data.status === 'error') {
                console.error('Error from backend:', data.error);
                showError(data.error);
                return;
            }
            
            // Display the complete results for the selected file
            displayQueryResults(query, data);
        })
        .catch(error => {
            console.error('Error selecting file:', error);
            hideLoading();
            showError('An error occurred while processing your selection. Please try again.');
        });
    }
    
    // Helper function to get the current chart type
    function getCurrentChartType() {
        // This is a placeholder - in a real implementation, you would get this from
        // the currently selected chart button or a variable tracking the current selection
        return 'line'; // Default to line chart for now
    }
    
    // Function to show loading indicator
    function showLoading() {
        console.log('Showing loading indicator');
        // Create loading indicator if it doesn't exist
        if (!document.getElementById('loadingIndicator')) {
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
            console.log('Loading indicator created and added to page');
        }
    }
    
    // Function to hide loading indicator
    function hideLoading() {
        console.log('Hiding loading indicator');
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
            console.log('Loading indicator removed');
        }
    }
    
    // Function to show error message
    function showError(message) {
        console.error('Showing error message:', message);
        queryResultsContainer.innerHTML = `
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                <strong class="font-bold">Error:</strong>
                <span class="block sm:inline"> ${message}</span>
            </div>
        `;
    }
});

function handleHesaResponse(response) {
    // DIRECT ALERT - This will definitely show up
    if (response && response.file_info) {
        let fileNames = [];
        if (Array.isArray(response.file_info)) {
            // Multiple files case
            fileNames = response.file_info.map(fi => fi.file_name || "Unknown file");
        } else {
            // Single file case (legacy)
            let fileName = "Unknown file";
            if (response.file_info.file_name) {
                fileName = response.file_info.file_name;
            } else if (response.file_info.cleaned_file_path) {
                fileName = response.file_info.cleaned_file_path.split('\\').pop();
            } else if (response.file_info.raw_file) {
                fileName = response.file_info.raw_file.split('\\').pop();
            }
            fileNames.push(fileName);
        }
        
        // Show an alert that can't be missed
        window.alert("FILES USED: " + fileNames.join(", "));
    }
    
    // Rest of the function continues...
    // Basic console log to ensure this function is being called
    console.log("⭐⭐⭐ HANDLE RESPONSE FUNCTION STARTED ⭐⭐⭐");
    
    if (response.status === 'success') {
        // Clear previous results
        document.getElementById('results-container').innerHTML = '';
        
        // Get raw file_info for debugging
        console.log("RAW FILE INFO:", JSON.stringify(response.file_info));
        
        // Extract filename(s) with fallbacks
        let fileInfo = "";
        if (Array.isArray(response.file_info)) {
            // Multiple files case
            fileInfo = response.file_info.map(fi => {
                return `${fi.year || 'Unknown year'}: ${fi.file_name || "Unknown file"}`;
            }).join("<br>");
        } else {
            // Single file case (legacy)
            let fileName = "Unknown file";
            if (response.file_info.file_name) {
                fileName = response.file_info.file_name;
            } else if (response.file_info.cleaned_file_path) {
                const parts = response.file_info.cleaned_file_path.split('\\');
                fileName = parts[parts.length - 1];
            } else if (response.file_info.raw_file) {
                const parts = response.file_info.raw_file.split('\\');
                fileName = parts[parts.length - 1];
            }
            fileInfo = fileName;
        }
        
        // Very visible console logs
        console.log("✅✅✅ FILES USED: " + fileInfo + " ✅✅✅");
        
        // Create a container for results
        const resultsContainer = document.getElementById('results-container');
        
        // 1. Create query section
        const querySection = document.createElement('div');
        querySection.className = 'mb-4';
        querySection.innerHTML = `
            <h3 class="text-lg font-bold mb-2">Query:</h3>
            <p>${response.query_info ? response.query_info.original_query || 'N/A' : 'N/A'}</p>
        `;
        resultsContainer.appendChild(querySection);
        
        // 2. Create a VERY visible filename section
        const fileNameSection = document.createElement('div');
        fileNameSection.style.backgroundColor = 'red';
        fileNameSection.style.color = 'white';
        fileNameSection.style.padding = '10px';
        fileNameSection.style.margin = '10px 0';
        fileNameSection.style.fontWeight = 'bold';
        fileNameSection.style.fontSize = '16px';
        fileNameSection.innerHTML = `FILES USED:<br>${fileInfo}`;
        resultsContainer.appendChild(fileNameSection);
        
        // 3. Results heading
        const resultsHeading = document.createElement('h3');
        resultsHeading.className = 'text-lg font-bold mb-2';
        resultsHeading.textContent = 'Results:';
        resultsContainer.appendChild(resultsHeading);
        
        // 4. Create table
        const table = document.createElement('table');
        table.className = 'min-w-full divide-y divide-gray-200';
        
        // Create header
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50';
        const headerRow = document.createElement('tr');
        
        if (response.columns && response.columns.length > 0) {
            response.columns.forEach(col => {
                const th = document.createElement('th');
                th.scope = 'col';
                th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
                th.textContent = col;
                headerRow.appendChild(th);
            });
        }
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white divide-y divide-gray-200';
        
        if (response.data && response.data.length > 0) {
            response.data.forEach(row => {
                const dataRow = document.createElement('tr');
                response.columns.forEach(col => {
                    const td = document.createElement('td');
                    td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
                    td.textContent = row[col] || '';
                    dataRow.appendChild(td);
                });
                tbody.appendChild(dataRow);
            });
        }
        
        table.appendChild(tbody);
        resultsContainer.appendChild(table);
        
        // 5. Create chart if chart data is available
        if (response.chart_data) {
            const chartSection = document.createElement('div');
            chartSection.className = 'mt-8';
            chartSection.innerHTML = `
                <h3 class="text-lg font-bold mb-2">Chart:</h3>
                <div class="h-96">
                    <canvas id="dataChart"></canvas>
                </div>
            `;
            resultsContainer.appendChild(chartSection);
            
            // Render chart
            const ctx = document.getElementById('dataChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: response.chart_data.labels,
                    datasets: response.chart_data.datasets || [{
                        label: response.chart_data.label || 'Value',
                        data: response.chart_data.values || [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        // Show the results container
        resultsContainer.classList.remove('hidden');
        
        // Final confirmation log
        console.log("⭐⭐⭐ HANDLE RESPONSE FUNCTION COMPLETED SUCCESSFULLY ⭐⭐⭐");
        
    } else {
        // Show error message
        showAlert('error', `Error: ${response.error}`);
    }
} 